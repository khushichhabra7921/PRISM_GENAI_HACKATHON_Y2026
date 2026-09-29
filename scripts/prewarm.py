"""Pre-warm the semantic cache from queries.json + gold samples (never from held-out paraphrases).

    python scripts/prewarm.py                 # uses the configured extractor (LLM if reachable)
    SGTE_LLM_PROVIDER=none python scripts/prewarm.py   # deterministic (used at Docker build time)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def prewarm(engine, verbose: bool = False) -> dict:
    queries = [q.query for q in engine.kit.queries]
    queries += [s["input"]["query"] for s in engine.kit.samples.values() if s.get("input", {}).get("query")]
    stats = {"queries": 0, "cached": 0, "no_match": 0, "already": 0}
    t0 = time.time()
    for q in dict.fromkeys(queries):  # de-duplicated, order kept
        rec = engine.troubleshoot(q)
        stats["queries"] += 1
        m = rec["meta"]
        if m["cache_hit"]:
            stats["already"] += 1
        elif m["fallback"]:
            stats["no_match"] += 1
        else:
            stats["cached"] += 1
        if verbose:
            print(f"  {m['latency_ms']:>8.0f} ms  hit={m['cache_hit']!s:5} fb={m['fallback']}  {q[:70]}")
    engine.cache.save()
    stats["seconds"] = round(time.time() - t0, 1)
    stats["plans"] = len(engine.cache)
    stats["keys"] = engine.cache.index.ntotal
    return stats


if __name__ == "__main__":
    from app.pipeline import Engine
    eng = Engine()
    if "--fresh" in sys.argv:
        eng.cache.clear()
    print(prewarm(eng, verbose="-v" in sys.argv))
