"""⑥ Deeplink engine: BM25 + dense over catalog METADATA -> RRF -> cross-encoder -> tie-break -> gate.

The masked URI string is never searched, generated or edited: it is only copied from the
winning catalog entry.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from app import models
from app.config import SETTINGS
from app.data_loader import CatalogEntry

MODES = ("hybrid", "bm25", "dense", "rules", "llm")


@dataclass
class ActionQuery:
    name: str
    steps: list[str]
    path: Optional[tuple[str, ...]] = None

    @property
    def leaf(self) -> str:
        return self.path[-1] if self.path else ""

    @property
    def targets(self) -> list[str]:
        """Most specific screen/toggle terms first: toggle objects, then the deepest menu term."""
        out = []
        for s in self.steps:
            m = re.match(r"(?:Optionally\s+)?(?:turn|toggle|switch) (?:on|off) ([A-Z][^.,]*)", s, re.I)
            if m and not m.group(1).lower().startswith(("it ", "it.", "them")):
                out.append(re.split(r" and | if | when ", m.group(1))[0].strip())
            # official phrasing: "tap the switch next to Touch sensitivity", "search for and select Screen timeout"
            m = re.search(r"\bswitch(?:es)? next to ([A-Z][^.,]*?)(?: to | or |[.,]|$)", s) or \
                re.search(r"\b[Ss]earch for and select ([A-Z][^.,]*?)(?: and |[.,]|$)", s)
            if m:
                out.append(m.group(1).strip())
        if self.leaf:
            out.append(self.leaf)
        return out

    def text(self) -> str:
        action_steps = [s for s in self.steps if not re.match(r"(Tap on|Navigate to|Open the|Swipe down)", s)]
        menu = " > ".join(self.path) if self.path else ""
        return f"{self.leaf}. {self.name}. {menu}. {' '.join(action_steps)}".strip(". ")

    def ce_text(self) -> str:
        action_steps = [s for s in self.steps if not re.match(r"(Tap on|Navigate to|Open the|Swipe down)", s)]
        return f"{self.name}. {self.leaf}. {' '.join(action_steps[:2])}"


@dataclass
class DeeplinkMatch:
    entry: Optional[CatalogEntry]
    uri: Optional[str]
    tier: str  # high | medium | low
    evidence: dict = field(default_factory=dict)

    def as_schema(self) -> Optional[dict]:
        """actionableDeeplink in the official key order (deeplink, description, message, originalType)."""
        if self.uri is None:
            return None
        if self.entry is None:  # dummy_positive
            return {"deeplink": self.uri, "description": "Open the matching Settings screen", "message": ""}
        out = {"deeplink": self.entry.uri, "description": self.entry.description, "message": self.entry.message}
        if self.entry.original_type:
            out["originalType"] = self.entry.original_type
        return out

    def validation_schema(self, steps: list[str]) -> Optional[dict]:
        """validationDeeplink: read the entry's key after the action. Only emitted when the expected value is
        stated by the steps themselves (a toggle turned on or off); otherwise None, never a guess."""
        v = self.entry.validation if self.entry else None
        if not v or not v.get("deeplink") or not v.get("key"):
            return None
        if v.get("condition") and v.get("value") is not None:  # fixed expectation from the catalog (official entry)
            expected = str(v["value"])
        elif v.get("resultType", "boolean") == "boolean":
            expected = toggle_target(steps, self.entry)
            if expected is None:
                return None
        else:
            return None
        return {"deeplink": v["deeplink"], "key": v["key"], "resultType": v.get("resultType", "boolean"),
                "condition": v.get("condition") or "equal", "value": expected}


OFF_RE = re.compile(r"\b(?:turn|toggle|switch) (?:it |them )?off\b|\bdisable\b|\bto disable it\b", re.I)
ON_RE = re.compile(r"\b(?:turn|toggle|switch) (?:it |them )?on\b|\benable\b", re.I)


def toggle_target(steps: list[str], entry: CatalogEntry) -> Optional[str]:
    """'True' / 'False' when the steps switch this toggle on / off, else None."""
    text = " ".join(s for s in steps if not s.startswith(("Navigate to and open", "Tap on ")))
    if OFF_RE.search(text):
        return "False"
    if ON_RE.search(text):
        return "True"
    if entry.leaf and re.search(rf"\bselect {re.escape(entry.leaf)}\b", text, re.I):
        return "True"
    return None


def _subject_match(entry: CatalogEntry, term: str) -> bool:
    """Is `term` the screen this entry opens (not merely mentioned in it)?"""
    t = term.lower().strip()
    if not t:
        return False
    if entry.path:
        return entry.leaf.lower() == t
    head = " ".join(entry.description.lower().split()[:6])
    return t in head


def rrf(ranks: list[np.ndarray], k: int) -> np.ndarray:
    """ranks[i][j] = rank (0-based) of doc j in list i."""
    return sum(1.0 / (k + r + 1) for r in ranks)


class DeeplinkEngine:
    def __init__(self, catalog: list[CatalogEntry], llm=None):
        self.catalog = catalog
        self.uris = {e.uri for e in catalog}
        self.docs = [e.text for e in catalog]  # description + message + cna only
        self.bm25 = BM25Okapi([models.tokenize(d) for d in self.docs])
        self.emb = models.embed(self.docs)
        self.llm = llm

    # -------------------------------------------------------------- retrieval
    def _scores(self, q: ActionQuery) -> tuple[np.ndarray, np.ndarray]:
        bm = np.asarray(self.bm25.get_scores(models.tokenize(q.text())), dtype="float32")
        dense = self.emb @ models.embed([q.text()])[0]
        return bm, dense

    @staticmethod
    def _rank(scores: np.ndarray) -> np.ndarray:
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order))
        return ranks

    def candidates(self, q: ActionQuery, mode: str = "hybrid", k: Optional[int] = None) -> list[dict]:
        k = k or SETTINGS.deeplink_top_k
        bm, dense = self._scores(q)
        bm_r, de_r = self._rank(bm), self._rank(dense)
        fused = {"hybrid": rrf([bm_r, de_r], SETTINGS.rrf_k), "bm25": -bm_r.astype(float),
                 "dense": -de_r.astype(float), "rules": -bm_r.astype(float)}[mode]
        top = np.argsort(-fused, kind="stable")[:k]
        return [{"idx": int(i), "bm25": round(float(bm[i]), 3), "dense": round(float(dense[i]), 3),
                 "bm25_rank": int(bm_r[i]) + 1, "dense_rank": int(de_r[i]) + 1, "rrf_rank": r + 1,
                 "rrf": round(float(fused[i]), 5)} for r, i in enumerate(top)]

    # -------------------------------------------------------------- matching
    def match(self, q: ActionQuery, mode: str = "hybrid") -> DeeplinkMatch:
        return self.match_many([q], mode)[0]

    def match_many(self, qs: list[ActionQuery], mode: str = "hybrid") -> list[DeeplinkMatch]:
        """Match several actions; the cross-encoder runs once over all (query, candidate) pairs."""
        if mode == "llm":
            return [self._match_llm(q) for q in qs]
        all_cands = [self.candidates(q, mode) for q in qs]
        if mode in ("bm25", "dense", "rules"):
            return [self._gate_simple(q, c, mode) for q, c in zip(qs, all_cands)]
        pairs = [(q.ce_text(), self.docs[c["idx"]]) for q, cs in zip(qs, all_cands) for c in cs]
        scores = models.rerank_pairs(pairs)
        out, i = [], 0
        for q, cs in zip(qs, all_cands):
            out.append(self._decide(q, cs, scores[i:i + len(cs)]))
            i += len(cs)
        return out

    def _decide(self, q: ActionQuery, cands: list[dict], ce: np.ndarray) -> DeeplinkMatch:
        objects = [t for t in q.targets if t != q.leaf]
        parents = [p.lower() for p in (q.path[1:-1] if q.path else ())]
        for c, s in zip(cands, ce):
            e = self.catalog[c["idx"]]
            text = e.text.lower()
            c["rerank"] = round(float(s), 3)
            # specificity: 2 = the toggle/child named in the steps, 1 = the deepest menu term, 0 = neither
            spec = 0
            if any(_subject_match(e, t) for t in objects) and (not q.leaf or q.leaf.lower() in text):
                spec = 2
            elif q.leaf and _subject_match(e, q.leaf):
                spec = 1
            c["specificity"] = spec
            c["parent_hits"] = sum(p in text for p in parents)
        # tie-break on the deepest, most specific menu term, then parent terms, then rerank score
        cands.sort(key=lambda c: (-c["specificity"], -c["parent_hits"], -c["rerank"]))
        best = cands[0]
        peers = [c for c in cands[1:] if (c["specificity"], c["parent_hits"]) == (best["specificity"], best["parent_hits"])]
        runner = peers[0] if peers else (cands[1] if len(cands) > 1 else None)
        margin = best["rerank"] - peers[0]["rerank"] if peers else 99.0
        e_best = self.catalog[best["idx"]]
        if best["specificity"] >= 1:
            tier = "high" if margin >= SETTINGS.dl_high_margin and best["rerank"] >= SETTINGS.dl_medium_score else "medium"
        elif q.leaf and q.leaf.lower() not in e_best.text.lower():
            tier = "low"  # the screen named in the text is not in the catalog
        elif best["rerank"] >= SETTINGS.dl_high_score:
            tier = "medium"  # never "high" without confirming the exact screen term
        else:
            tier = "low"
        ev = {"query": q.ce_text(), "bm25": best["bm25"], "dense": best["dense"], "rrf_rank": best["rrf_rank"],
              "rerank": best["rerank"], "specificity": best["specificity"], "parent_hits": best["parent_hits"],
              "runner_up": self.catalog[runner["idx"]].description if runner else None,
              "runner_up_rerank": runner["rerank"] if runner else None,
              "margin": round(margin, 3) if margin != 99.0 else None, "tier": tier}
        if tier == "low":
            return self._low(q, ev)
        e = self.catalog[best["idx"]]
        ev["matched"] = e.description
        return DeeplinkMatch(e, e.uri, tier, ev)

    def _low(self, q: ActionQuery, ev: dict) -> DeeplinkMatch:
        opens_settings = bool(q.path) and q.path[0] == "Settings"
        ev["decision"] = "dummy_positive" if opens_settings else "no_deeplink"
        return DeeplinkMatch(None, SETTINGS.dummy_deeplink if opens_settings else None, "low", ev)

    def _gate_simple(self, q: ActionQuery, cands: list[dict], mode: str) -> DeeplinkMatch:
        best = cands[0]
        score = best["bm25"] if mode in ("bm25", "rules") else best["dense"]
        floor = 1.0 if mode in ("bm25", "rules") else 0.3
        ev = {"mode": mode, "score": score, "bm25": best["bm25"], "dense": best["dense"]}
        if score < floor:
            return self._low(q, ev)
        e = self.catalog[best["idx"]]
        return DeeplinkMatch(e, e.uri, "medium", ev)

    def _match_llm(self, q: ActionQuery) -> DeeplinkMatch:
        """Ablation baseline: the LLM picks an entry from the whole catalog (by index, never a URI)."""
        listing = "\n".join(f"{i}: {e.description} | {e.cna}" for i, e in enumerate(self.catalog))
        prompt = (f"Pick the single settings screen that best matches this troubleshooting action.\n"
                  f"Action: {q.ce_text()}\nMenu path: {' > '.join(q.path or [])}\n\nScreens:\n{listing}\n\n"
                  'Reply as JSON: {"index": <number or -1 if none fits>}')
        res = self.llm.complete_json(prompt, {"type": "object", "properties": {"index": {"type": "integer"}},
                                              "required": ["index"]}, max_tokens=20, num_ctx=8192)
        idx = (res.data or {}).get("index", -1) if res.ok else -1
        ev = {"mode": "llm", "index": idx, "latency_ms": res.latency_ms, "cost_usd": res.cost_usd}
        if not isinstance(idx, int) or not 0 <= idx < len(self.catalog):
            return self._low(q, ev)
        e = self.catalog[idx]
        return DeeplinkMatch(e, e.uri, "medium", ev)
