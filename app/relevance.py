"""④a Which parts of a long SIIS article answer this complaint?

Official articles mix a step-by-step procedure with feature tours, glossaries and tips for other problems
(a "Screen mirroring" article answers "my screen looks small" with one tip about the aspect ratio). Every
paragraph block that contains instructions is scored against the complaint with the cross-encoder and the
dense embedder:

  * a numbered procedure (Step 1..N) or an article with at most `rel_keep_all` instruction blocks is kept whole:
    the SIIS system already chose it for this complaint and every step is part of the fix;
  * otherwise a block is kept when its cross-encoder score is within `rel_ce_delta` of the best block, or its
    dense similarity is within `rel_dense_delta` of the best; the best block is always kept.

Nothing is rewritten here; blocks are only kept or dropped, and the decision is recorded as evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app import models
from app.config import SETTINGS
from app.siis_text import Article, Block


@dataclass
class BlockScore:
    block: int
    heading: str
    snippet: str
    ce: float
    dense: float
    kept: bool

    def evidence(self) -> dict:
        return {"block": self.block, "heading": self.heading, "text": self.snippet, "rerank": round(self.ce, 3),
                "dense": round(self.dense, 3), "kept": self.kept}


def score_blocks(query: str, blocks: list[Block]) -> tuple[np.ndarray, np.ndarray]:
    if not blocks:
        return np.zeros(0, dtype="float32"), np.zeros(0, dtype="float32")
    texts = [b.scoring_text[:1500] for b in blocks]
    ce = models.rerank_pairs([(query, t) for t in texts])
    dense = models.embed(texts) @ models.embed([query])[0]
    return ce, dense


def select_blocks(query: str, article: Article, actionable: list[int]) -> list[BlockScore]:
    """Score the instruction-bearing blocks (indices into article.blocks()) and decide which to keep."""
    blocks = article.blocks()
    if not actionable:
        return []
    ce, dense = score_blocks(query, [blocks[i] for i in actionable])
    keep_all = article.procedural or len(actionable) <= SETTINGS.rel_keep_all
    best_ce, best_dense = float(ce.max()), float(dense.max())
    out = []
    for j, bi in enumerate(actionable):
        kept = keep_all or ce[j] >= best_ce - SETTINGS.rel_ce_delta or dense[j] >= best_dense - SETTINGS.rel_dense_delta
        b = blocks[bi]
        out.append(BlockScore(bi, b.heading, b.text[:90], float(ce[j]), float(dense[j]), bool(kept)))
    return out
