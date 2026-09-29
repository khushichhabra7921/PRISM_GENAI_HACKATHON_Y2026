"""② Semantic cache: FAISS inner-product index over normalised embeddings, persisted to disk.

Keys = original query + canonical query + all variations, each pointing to ONE validated plan.
Only plans that passed validation are ever written.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import faiss
import numpy as np


@dataclass
class CacheHit:
    plan_id: str
    plan: dict
    score: float
    key: str


class SemanticCache:
    def __init__(self, directory: Path, dim: int, tau: float):
        self.dir = Path(directory)
        self.dim = dim
        self.tau = tau
        self._lock = threading.Lock()
        self.index = faiss.IndexFlatIP(dim)
        self.keys: list[str] = []
        self.key_plan: list[str] = []
        self.plans: dict[str, dict] = {}
        self.load()

    # -------------------------------------------------------------- persistence
    @property
    def _files(self) -> tuple[Path, Path]:
        return self.dir / "index.faiss", self.dir / "store.json"

    def load(self) -> None:
        idx_f, store_f = self._files
        if idx_f.exists() and store_f.exists():
            store = json.loads(store_f.read_text(encoding="utf-8"))
            index = faiss.read_index(str(idx_f))
            if index.d == self.dim and index.ntotal == len(store["keys"]):
                self.index, self.keys, self.key_plan, self.plans = index, store["keys"], store["key_plan"], store["plans"]

    def save(self) -> None:
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            idx_f, store_f = self._files
            faiss.write_index(self.index, str(idx_f))
            store_f.write_text(json.dumps({"keys": self.keys, "key_plan": self.key_plan, "plans": self.plans},
                                          ensure_ascii=False), encoding="utf-8")

    def clear(self) -> None:
        with self._lock:
            self.index = faiss.IndexFlatIP(self.dim)
            self.keys, self.key_plan, self.plans = [], [], {}

    # -------------------------------------------------------------- read / write
    def top(self, vec: np.ndarray) -> Optional[CacheHit]:
        if self.index.ntotal == 0:
            return None
        scores, ids = self.index.search(vec.reshape(1, -1).astype("float32"), 1)
        i = int(ids[0][0])
        if i < 0:
            return None
        pid = self.key_plan[i]
        return CacheHit(pid, self.plans[pid], float(scores[0][0]), self.keys[i])

    def lookup(self, vec: np.ndarray, tau: Optional[float] = None) -> Optional[CacheHit]:
        hit = self.top(vec)
        return hit if hit and hit.score >= (self.tau if tau is None else tau) else None

    def best_plan_for_article(self, vec: np.ndarray, article_id: str, k: int = 64) -> Optional[CacheHit]:
        """Among cached single-article plans grounded in `article_id`, the one whose key is closest to `vec`."""
        if self.index.ntotal == 0:
            return None
        scores, ids = self.index.search(vec.reshape(1, -1).astype("float32"), min(k, self.index.ntotal))
        for sc, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            pid = self.key_plan[int(i)]
            if self.plans[pid].get("siis_ids") == [article_id]:
                return CacheHit(pid, self.plans[pid], float(sc), self.keys[int(i)])
        for pid, plan in self.plans.items():  # article has a plan but none of its keys is in the top-k
            if plan.get("siis_ids") == [article_id]:
                return CacheHit(pid, plan, 0.0, plan["query"])
        return None

    def add(self, plan_id: str, plan: dict, keys: list[str], vecs: np.ndarray) -> None:
        with self._lock:
            self.plans[plan_id] = plan
            seen = {k.lower() for k, p in zip(self.keys, self.key_plan) if p == plan_id}
            rows = []
            for k, v in zip(keys, vecs):
                if k.lower() not in seen:
                    seen.add(k.lower())
                    self.keys.append(k)
                    self.key_plan.append(plan_id)
                    rows.append(v)
            if rows:
                self.index.add(np.vstack(rows).astype("float32"))

    def __len__(self) -> int:
        return len(self.plans)
