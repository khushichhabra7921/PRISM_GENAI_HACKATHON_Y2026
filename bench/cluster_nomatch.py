"""Offline clustering of logged no-match queries -> knowledge-gap themes for the SIIS content team.

    python bench/cluster_nomatch.py [--log logs/no_match.jsonl] [--threshold 0.55]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import models  # noqa: E402


def load(log: Path) -> list[dict]:
    rows, seen = [], set()
    if not log.exists():
        return rows
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = r.get("canonical_query", r.get("query", "")).lower().strip()
        if key and key not in seen and r.get("reason") == "no_match":
            seen.add(key)
            rows.append(r)
    return rows


def cluster(texts: list[str], threshold: float) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering
    if len(texts) == 1:
        return np.zeros(1, dtype=int)
    X = models.embed(texts)
    return AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                   distance_threshold=threshold).fit_predict(X)


def themes(rows: list[dict], labels: np.ndarray) -> list[dict]:
    out = []
    texts = [r.get("canonical_query") or r["query"] for r in rows]
    X = models.embed(texts)
    for lab in sorted(set(labels), key=lambda l: -int((labels == l).sum())):
        idx = np.where(labels == lab)[0]
        centroid = X[idx].mean(0)
        rep = texts[idx[int(np.argmax(X[idx] @ centroid))]]
        kw = Counter(t for i in idx for t in set(models.tokenize(texts[i])))
        out.append({"cluster": int(lab), "size": int(len(idx)), "keywords": [w for w, _ in kw.most_common(4)],
                    "representative": rep, "topics": dict(Counter(rows[i].get("topic", "") for i in idx)),
                    "queries": [texts[i] for i in idx]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(ROOT / "logs" / "no_match.jsonl"))
    ap.add_argument("--threshold", type=float, default=0.55, help="cosine distance for merging clusters")
    ap.add_argument("--out", default=str(ROOT / "bench" / "nomatch_clusters.json"))
    args = ap.parse_args()
    rows = load(Path(args.log))
    if not rows:
        print("no genuine no-match queries logged yet")
        return
    th = themes(rows, cluster([r.get("canonical_query") or r["query"] for r in rows], args.threshold))
    Path(args.out).write_text(json.dumps(th, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(rows)} distinct no-match queries -> {len(th)} knowledge-gap themes (largest first)\n")
    for t in th:
        print(f"[{t['size']}] {' / '.join(t['keywords'])}  e.g. \"{t['representative']}\"")
        for q in t["queries"][:5]:
            print(f"      - {q}")


if __name__ == "__main__":
    main()
