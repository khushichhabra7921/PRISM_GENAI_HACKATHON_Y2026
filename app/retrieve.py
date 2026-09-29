"""③ SIIS retrieval: BM25 + dense (article and sentence level) -> RRF -> cross-encoder gate.

The winning article's text is the grounding boundary for extraction; sentence spans are
kept so every step can be traced back to its source sentence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from app import models
from app.config import SETTINGS
from app.data_loader import SiisDoc
from app.deeplinks import rrf
from app.textutils import split_sentences


@dataclass
class SiisHit:
    doc: SiisDoc
    bm25: float
    dense: float
    rrf_rank: int
    rerank: float

    def evidence(self) -> dict:
        return {"siis_id": self.doc.id, "title": self.doc.title, "bm25": round(self.bm25, 3),
                "dense": round(self.dense, 3), "rrf_rank": self.rrf_rank, "rerank": round(self.rerank, 3)}


class SiisIndex:
    def __init__(self, docs: list[SiisDoc]):
        self.docs = docs
        self.full = [f"{d.title}. {d.text}" for d in docs]
        self.bm25 = BM25Okapi([models.tokenize(t) for t in self.full])
        self.doc_emb = models.embed(self.full)
        self.sentences: list[tuple[int, str]] = [(i, s) for i, d in enumerate(docs) for s in split_sentences(d.text)]
        self.sent_emb = models.embed([s for _, s in self.sentences])
        self.sent_doc = np.array([i for i, _ in self.sentences])
        self.title_emb = models.embed([d.title for d in docs])

    def _dense(self, qv: np.ndarray) -> np.ndarray:
        doc = self.doc_emb @ qv
        sent = np.full(len(self.docs), -1.0, dtype="float32")
        np.maximum.at(sent, self.sent_doc, self.sent_emb @ qv)
        return np.maximum.reduce([doc, sent, self.title_emb @ qv])

    def search(self, query: str, k: Optional[int] = None, qv: Optional[np.ndarray] = None) -> list[SiisHit]:
        k = k or SETTINGS.siis_top_k
        qv = qv if qv is not None else models.embed([query])[0]
        bm = np.asarray(self.bm25.get_scores(models.tokenize(query)), dtype="float32")
        dense = self._dense(qv)
        rank = lambda s: np.argsort(np.argsort(-s, kind="stable"), kind="stable")  # noqa: E731
        fused_rank = rank(rrf([rank(bm), rank(dense)], SETTINGS.rrf_k))
        top = np.argsort(-dense, kind="stable")[:k]
        ce = models.rerank_pairs([(query, self.full[i]) for i in top])
        return [SiisHit(self.docs[i], float(bm[i]), float(dense[i]), int(fused_rank[i]) + 1, float(c))
                for i, c in zip(top, ce)]

    @staticmethod
    def passes(hit: SiisHit, best_ce: float) -> bool:
        """No-match gate: very close semantically, or close AND confirmed by the cross-encoder."""
        if hit.dense >= SETTINGS.siis_dense_strong:
            return True
        return hit.dense >= SETTINGS.siis_min_dense and best_ce >= SETTINGS.siis_min_ce

    def best(self, query: str, qv: Optional[np.ndarray] = None) -> tuple[Optional[SiisHit], list[SiisHit]]:
        hits = self.search(query, qv=qv)
        if not hits:
            return None, hits
        return (hits[0] if self.passes(hits[0], max(h.rerank for h in hits)) else None), hits
