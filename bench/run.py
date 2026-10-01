"""Benchmarks -> results.jsonl + metrics.md. Every number is measured on this machine, never typed in.

    python bench/run.py                    # full run (uses the LLM if reachable)
    python bench/run.py --quick            # fewer cold / ablation samples
    SGTE_LLM_PROVIDER=none python bench/run.py   # deterministic path only

Runs on the synthetic regression kit (data/synthetic/: labelled queries, held-out paraphrases, gold samples);
the official kit (data/input.txt + SIIS payloads) is evaluated by bench/official.py.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import models  # noqa: E402
from app.config import SETTINGS  # noqa: E402
from app.data_loader import load_synthetic_kit  # noqa: E402
from app.deeplinks import ActionQuery  # noqa: E402
from app.enrich import enrich  # noqa: E402
from app.llm_client import LLMClient, Usage  # noqa: E402
from app.pipeline import Engine  # noqa: E402
from app.validate import CHECKS, check_record  # noqa: E402
from bench.deeplink_gold import build_cases, relevance  # noqa: E402
from bench.scoring import deeplink_relevance, step_accuracy  # noqa: E402

TAUS = (0.80, 0.85, 0.90)


def pct(xs, p):
    return round(float(np.percentile(xs, p)), 1) if xs else None


def env_info() -> dict:
    info = {"os": f"{platform.system()} {platform.release()}", "python": platform.python_version(),
            "cpu": platform.processor() or platform.machine(), "logical_cpus": os.cpu_count(),
            "torch_threads": SETTINGS.torch_threads}
    try:
        import psutil  # optional
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        b = psutil.sensors_battery()
        info["power"] = ("on battery" if b and not b.power_plugged else "plugged in") if b else "desktop"
    except Exception:
        pass
    return info


# ---------------------------------------------------------------- 1. results.jsonl + compliance
def run_queries(eng: Engine, kit) -> tuple[list[dict], dict]:
    rows, routing = [], Counter()
    for q in kit.queries:
        rec = eng.troubleshoot(q.query)
        rows.append({"query": rec["query"], "query_variations": rec["query_variations"],
                     "response": rec["response"], "meta": rec["meta"]})
        got = set(_siis_ids(rec))
        if q.type == "no_match":
            routing["no_match_correct" if not rec["response"]["contexts"] else "no_match_false_plan"] += 1
        elif q.type == "multi":
            routing["multi_both" if set(q.siis_ids) <= got else
                    "multi_partial" if got & set(q.siis_ids) else "multi_wrong"] += 1
        else:
            routing["single_correct" if q.siis_ids[0] in got else
                    "single_no_match" if not got else "single_wrong"] += 1
    eng.cache.save()
    return rows, dict(routing)


def _siis_ids(rec: dict) -> list[str]:
    ev = rec["meta"].get("evidence") or {}
    goals = ev.get("goals") or (ev.get("plan_evidence") or {}).get("goals") or []
    return [g["siis"]["siis_id"] for g in goals if g.get("siis") and g["siis"].get("siis_id")]


def compliance(rows: list[dict], uris: set[str], dummy: str) -> dict:
    n = len(rows)
    schema_ok = rules_ok = leaks = 0
    per_check = Counter()
    dl_total = dl_catalog = dl_dummy = auto_total = auto_with = 0
    for r in rows:
        v = check_record(r, uris)
        per_check.update(x.check for x in v)
        schema_ok += not any(x.check == "schema" for x in v)
        rules_ok += not v
        leaks += sum(x.check == "zero_urls" for x in v)
        for g in r["response"]["contexts"]:
            for a in g["actions"]:
                has = False
                for sg in a["stepGroups"]:
                    d = sg.get("actionableDeeplink")
                    if d:
                        dl_total += 1
                        dl_catalog += d["deeplink"] in uris
                        dl_dummy += d["deeplink"] == dummy
                        has = True
                if a["category"] == "auto":
                    auto_total += 1
                    auto_with += has
    return {"lines": n, "schema_valid_pct": round(100 * schema_ok / n, 1), "rule_compliance_pct": round(100 * rules_ok / n, 1),
            "violations_by_check": {c: per_check.get(c, 0) for c in CHECKS}, "url_leaks": leaks,
            "deeplinks": dl_total, "catalog_validity_pct": round(100 * (dl_catalog + dl_dummy) / dl_total, 1) if dl_total else 100.0,
            "dummy_positive": dl_dummy, "auto_actions": auto_total,
            "auto_with_deeplink_pct": round(100 * auto_with / auto_total, 1) if auto_total else None}


# ---------------------------------------------------------------- 2. latency
def latency(eng: Engine, kit, n_cold: int, n_canonical: int) -> dict:
    cached = [q.query for q in kit.queries if q.type != "no_match"]
    exact = [eng.troubleshoot(q)["meta"] for q in cached[:max(30, len(cached))]]
    exact_ms = [m["latency_ms"] for m in exact if m["cache_hit"]]
    para = [p for h in kit.heldout for p in h["paraphrases"]]
    raw = [(p, eng.troubleshoot(p, cache_only=True)) for p in para]
    para_meta = [r["meta"] for _, r in raw if r]
    para_hit_ms = [m["latency_ms"] for m in para_meta]
    article_ms = [m["latency_ms"] for m in para_meta if m["evidence"]["lookup"] == "article"]
    key_ms = [m["latency_ms"] for m in para_meta if m["evidence"]["lookup"] == "raw"]
    misses = [p for p, r in raw if r is None][:n_canonical]
    canon_meta = [eng.troubleshoot(p, write_cache=False)["meta"] for p in misses]
    para_canon_ms = [m["latency_ms"] for m in canon_meta if m["cache_hit"]]
    cold_meta = [eng.troubleshoot(p, use_cache=False)["meta"] for p in para[:n_cold]]
    cold_ms = [m["latency_ms"] for m in cold_meta]
    stage = defaultdict(list)
    for m in cold_meta:
        for k, v in m["timings_ms"].items():
            stage[k].append(v)
    return {
        "exact_hit": {"n": len(exact_ms), "p50": pct(exact_ms, 50), "p95": pct(exact_ms, 95)},
        "paraphrase_hit_raw": {"n": len(para_hit_ms), "p50": pct(para_hit_ms, 50), "p95": pct(para_hit_ms, 95),
                               "key_tier": {"n": len(key_ms), "p95": pct(key_ms, 95)},
                               "article_tier": {"n": len(article_ms), "p95": pct(article_ms, 95)}},
        "paraphrase_hit_canonical": {"n": len(para_canon_ms), "tried": len(misses), "p50": pct(para_canon_ms, 50),
                                     "p95": pct(para_canon_ms, 95)},
        "cold": {"n": len(cold_ms), "p50": pct(cold_ms, 50), "p95": pct(cold_ms, 95),
                 "llm_calls_mean": round(statistics.mean(m["llm_calls"] for m in cold_meta), 2) if cold_meta else 0,
                 "cost_usd_mean": round(statistics.mean(m["cost_usd"] for m in cold_meta), 6) if cold_meta else 0,
                 "tokens_in_mean": round(statistics.mean(m["tokens_in"] for m in cold_meta), 1) if cold_meta else 0,
                 "tokens_out_mean": round(statistics.mean(m["tokens_out"] for m in cold_meta), 1) if cold_meta else 0,
                 "models": sorted({m["model"] for m in cold_meta}),
                 "stage_p50_ms": {k: pct(v, 50) for k, v in stage.items()}},
        "cache_hit_llm_calls": sum(m["llm_calls"] for m in exact) + sum(m["llm_calls"] for m in para_meta),
    }


# ---------------------------------------------------------------- 3. accuracy on gold samples
def accuracy(eng: Engine, kit, extract_mode: str | None = None) -> dict:
    out = []
    for name, s in sorted(kit.samples.items()):
        gold_ctx = s["output"]["response"]["contexts"]
        for grounded in (True, False):
            rec = eng.troubleshoot(s["input"]["query"], s["input"]["siis_response"] if grounded else None,
                                   use_cache=False, extract_mode=extract_mode)
            pred_ctx = rec["response"]["contexts"]
            if not gold_ctx:
                ok = not pred_ctx and rec["meta"]["fallback"] in ("no_match", "no_siis_context")
                out.append({"sample": name, "grounded": grounded, "step_accuracy": 3.0 if ok else 0.0,
                            "deeplink_relevance": 2.0 if ok else 0.0, "note": f"fallback={rec['meta']['fallback']}",
                            "latency_ms": rec["meta"]["latency_ms"], "cost_usd": rec["meta"]["cost_usd"]})
                continue
            pred = pred_ctx[0] if pred_ctx else None
            acc = step_accuracy(pred, gold_ctx[0])
            rel = deeplink_relevance(pred, gold_ctx[0], kit.catalog, SETTINGS.dummy_deeplink)
            wrong = [f"{n['gold_action']}: gold '{n['gold']}' vs pred '{n['pred']}'" for n in rel["per_action"] if n["score"] < 2]
            out.append({"sample": name, "grounded": grounded, "step_accuracy": acc["score"],
                        "completeness": acc["completeness"], "correctness": acc["correctness"],
                        "ordering": acc["ordering"], "deeplink_relevance": rel["score"],
                        "note": "; ".join(wrong) or "all deeplinks exact",
                        "latency_ms": rec["meta"]["latency_ms"], "cost_usd": rec["meta"]["cost_usd"],
                        "model": rec["meta"]["model"]})
    return {"rows": out,
            "mean_step_accuracy": round(statistics.mean(r["step_accuracy"] for r in out), 3),
            "mean_deeplink_relevance": round(statistics.mean(r["deeplink_relevance"] for r in out), 3)}


# ---------------------------------------------------------------- 4. cache efficacy
def cache_sweep(eng: Engine, kit, embed_model: str | None = None) -> dict:
    keys, key_plan, plans = eng.cache.keys, eng.cache.key_plan, eng.cache.plans
    if embed_model:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(embed_model, device="cpu")
        enc = lambda xs: m.encode(xs, normalize_embeddings=True, batch_size=64, convert_to_numpy=True)  # noqa: E731
    else:
        enc = models.embed_cache
    K = enc(keys)
    pos = [(p, h["siis_id"]) for h in kit.heldout for p in h["paraphrases"]]
    queries = [p for p, _ in pos] + list(kit.negatives)
    Q = enc(queries)
    sims = Q @ K.T
    best = sims.argmax(1)
    best_s = sims.max(1)
    res = {}
    for tau in TAUS:
        true_hits = false_hits = neg_hits = 0
        for i, (p, sid) in enumerate(pos):
            if best_s[i] >= tau:
                ok = sid in plans[key_plan[best[i]]].get("siis_ids", [])
                true_hits += ok
                false_hits += not ok
        for j in range(len(pos), len(queries)):
            neg_hits += best_s[j] >= tau
        hits = true_hits + false_hits + neg_hits
        res[f"{tau:.2f}"] = {"hit_rate_pct": round(100 * true_hits / len(pos), 1),
                             "any_hit_pct": round(100 * (true_hits + false_hits) / len(pos), 1),
                             "false_hits": int(false_hits + neg_hits),
                             "false_hit_rate_pct": round(100 * (false_hits + neg_hits) / hits, 1) if hits else 0.0,
                             "negatives_hit": int(neg_hits)}
    return {"embed_model": embed_model or SETTINGS.cache_embed_model, "paraphrases": len(pos),
            "negatives": len(kit.negatives), "keys": len(keys), "plans": len(plans), "by_tau": res}


def dual_lookup(eng: Engine, kit, limit: int) -> dict:
    """End-to-end tiered lookup at the configured τ: raw key → article-anchored (both zero-LLM) → canonical key
    (after enrichment). Correct = the served plan is grounded in the paraphrase's expected article."""
    pos = [(p, h["siis_id"]) for h in kit.heldout for p in h["paraphrases"]][:limit]
    tiers = Counter()
    wrong = Counter()
    for p, sid in pos:
        rec = eng.troubleshoot(p, cache_only=True)
        kind, plan_ids = None, []
        if rec:
            kind, plan_ids = rec["meta"]["evidence"]["lookup"], rec["meta"]["evidence"]["siis_ids"]
        else:
            mode = eng.enrich_mode()
            e = enrich(p, eng.llm, Usage(), "rules" if mode == "background" else mode)
            hit = eng.cache.lookup(models.embed_cache([e.canonical_query])[0])
            if hit:
                kind, plan_ids = "canonical", hit.plan.get("siis_ids", [])
        if kind:
            (tiers if sid in plan_ids else wrong)[kind] += 1
    neg_hits = sum(1 for q in kit.negatives if eng.troubleshoot(q, cache_only=True))
    n = len(pos)
    zero_llm = tiers["raw"] + tiers["article"]
    hits_all = sum(tiers.values()) + sum(wrong.values()) + neg_hits
    return {"tau": eng.cache.tau, "n": n, "raw_hit_pct": round(100 * tiers["raw"] / n, 1),
            "zero_llm_hit_pct": round(100 * zero_llm / n, 1),
            "dual_hit_pct": round(100 * sum(tiers.values()) / n, 1),
            "correct_by_tier": dict(tiers), "wrong_by_tier": dict(wrong), "wrong_hits": sum(wrong.values()),
            "negatives": len(kit.negatives), "negatives_hit": neg_hits,
            "false_hit_rate_pct": round(100 * (sum(wrong.values()) + neg_hits) / hits_all, 1) if hits_all else 0.0}


# ---------------------------------------------------------------- 5. ablations
def deeplink_ablation(eng: Engine, kit, llm_cases: int, with_llm: bool = True) -> dict:
    cases = build_cases(kit)
    out = {}
    modes = ["hybrid", "bm25", "dense"] + (["llm"] if with_llm and eng.llm and eng.llm.available() else [])
    for mode in modes:
        cs = cases if mode != "llm" else cases[:llm_cases]
        rels, lat, cost = [], [], []
        for c in cs:
            t = time.perf_counter()
            m = eng.deeplinks.match(c.query, mode)
            lat.append((time.perf_counter() - t) * 1000)
            cost.append(m.evidence.get("cost_usd", 0.0))
            rels.append(relevance(m.uri, c, SETTINGS.dummy_deeplink))
        rels = np.array(rels)
        out[mode] = {"n": len(cs), "exact_screen_pct": round(100 * float(np.mean(rels == 2)), 1),
                     "parent_or_dummy_pct": round(100 * float(np.mean(rels == 1)), 1),
                     "mean_relevance": round(float(rels.mean()), 3), "p95_ms_per_action": pct(lat, 95),
                     "cost_usd_per_action": round(float(np.mean(cost)), 6)}
    return out


def extraction_ablation(eng: Engine, kit, with_llm: bool = True) -> dict:
    out = {}
    modes = ["rules"] + (["pointer", "generative"] if with_llm and eng.llm and eng.llm.available() else [])
    # note: the LLM modes here use whichever provider is first reachable (see report['llm'])
    for mode in modes:
        acc = accuracy(eng, kit, extract_mode=mode)
        rows = [r for r in acc["rows"] if r["grounded"]]
        out[mode] = {"mean_step_accuracy": round(statistics.mean(r["step_accuracy"] for r in rows), 3),
                     "mean_deeplink_relevance": round(statistics.mean(r["deeplink_relevance"] for r in rows), 3),
                     "p95_latency_ms": pct([r["latency_ms"] for r in rows], 95),
                     "mean_cost_usd": round(statistics.mean(r["cost_usd"] for r in rows), 6)}
    return out


def hosted_vs_local(kit, n: int) -> dict:
    out = {}
    for prov in ("ollama", "anthropic", "openai"):
        client = LLMClient(SETTINGS, provider=prov)
        if not client.available():
            out[prov] = {"measured": False, "reason": "provider not reachable / no API key"}
            continue
        s = replace(SETTINGS, llm_provider=prov, llm_fallback="")
        eng = Engine(s, kit=kit, llm=client, cache_dir=ROOT / "bench" / ".cache_provider")
        para = [p for h in kit.heldout for p in h["paraphrases"]][:n]
        metas = [eng.troubleshoot(p, use_cache=False, enrich_mode="llm", extract_mode="pointer")["meta"]
                 for p in para]
        ok_calls = sum(m["llm_ok"] for m in metas)
        if ok_calls == 0:
            probe = client.complete_json('Reply with JSON {"ok": true}', {"type": "object"}, max_tokens=10)
            out[prov] = {"measured": False, "reason": f"every call failed ({probe.error[:120]})"}
            continue
        lat = [m["latency_ms"] for m in metas]
        acc = accuracy(eng, kit, extract_mode="pointer") if prov != "ollama" else None
        out[prov] = {"measured": True, "model": client.model_name, "n": len(metas), "p50_ms": pct(lat, 50),
                     "p95_ms": pct(lat, 95), "mean_cost_usd": round(statistics.mean(m["cost_usd"] for m in metas), 6),
                     "llm_ok_pct": round(100 * ok_calls / max(1, sum(m["llm_calls"] for m in metas)), 1),
                     "fallback_rate_pct": round(100 * sum(bool(m["fallback"]) for m in metas) / len(metas), 1),
                     "step_accuracy": acc["mean_step_accuracy"] if acc else None}
    return out


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cold", type=int, default=30)
    ap.add_argument("--llm-cases", type=int, default=10)
    ap.add_argument("--profile-n", type=int, default=10, help="cold queries per LLM profile")
    ap.add_argument("--canonical", type=int, default=15, help="paraphrase misses sent through enrichment")
    ap.add_argument("--dual", type=int, default=1000, help="max paraphrases for the dual-lookup hit rate")
    ap.add_argument("--compare-embed", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", default=str(ROOT))
    ap.add_argument("--reuse-llm-from", default="", help="copy the cache-independent LLM ablation sections "
                    "(LLM deeplink baseline, LLM extraction modes, providers) from an earlier report.json")
    args = ap.parse_args()
    if args.quick:
        args.cold, args.llm_cases, args.canonical, args.dual, args.profile_n = 10, 3, 5, 30, 3
    t_start = time.time()
    kit = load_synthetic_kit()
    cache_dir = ROOT / "bench" / ".cache"
    s = replace(SETTINGS, cache_dir=cache_dir, log_dir=ROOT / "logs")
    eng = Engine(s, kit=kit)
    eng.cache.clear()
    report: dict = {"generated": datetime.now().isoformat(timespec="seconds"), "env": env_info(),
                    "synthetic_kit": kit.synthetic, "llm": eng.llm.model_name if eng.llm else "none",
                    "embed_model": SETTINGS.embed_model, "cache_embed_model": SETTINGS.cache_embed_model,
                    "rerank_model": SETTINGS.rerank_model,
                    "extract_mode": eng.extract_mode(), "enrich_mode": eng.enrich_mode(),
                    "tau": SETTINGS.cache_tau}
    print(f"[bench] llm={report['llm']}  queries={len(kit.queries)}")

    rows, routing = run_queries(eng, kit)
    t_up = time.time()
    pending = eng.pending_upgrades
    eng.drain()
    report["background_upgrade"] = {"mode": eng.enrich_mode(), "queued_at_end_of_run": pending,
                                    "wait_s": round(time.time() - t_up, 1),
                                    "plans_with_llm_variations": sum(p.get("variations_source", "rules") != "rules"
                                                                     for p in eng.cache.plans.values())}
    print("[bench] background upgrade", report["background_upgrade"])
    out = Path(args.out)
    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report["results_cold_latency"] = {"p50": pct([r["meta"]["latency_ms"] for r in rows], 50),
                                      "p95": pct([r["meta"]["latency_ms"] for r in rows], 95)}
    report["routing"] = routing
    report["compliance"] = compliance(rows, eng.uris, SETTINGS.dummy_deeplink)
    print("[bench] compliance", report["compliance"])
    report["accuracy"] = accuracy(eng, kit)
    print("[bench] accuracy", report["accuracy"]["mean_step_accuracy"], report["accuracy"]["mean_deeplink_relevance"])
    report["cache"] = {"primary": cache_sweep(eng, kit)}
    if args.compare_embed:
        report["cache"]["compare"] = cache_sweep(eng, kit, args.compare_embed)
    report["cache"]["dual_lookup"] = dual_lookup(eng, kit, args.dual)
    print("[bench] cache", json.dumps(report["cache"]["primary"]["by_tau"]), report["cache"]["dual_lookup"])
    report["latency"] = latency(eng, kit, args.cold, args.canonical)
    print("[bench] latency", json.dumps({k: v for k, v in report["latency"].items() if k != "cold"}),
          report["latency"]["cold"]["p50"], report["latency"]["cold"]["p95"])
    prev = json.loads(Path(args.reuse_llm_from).read_text(encoding="utf-8")) if args.reuse_llm_from else None
    report["ablation_deeplink"] = deeplink_ablation(eng, kit, args.llm_cases, with_llm=prev is None)
    report["ablation_extraction"] = extraction_ablation(eng, kit, with_llm=prev is None)
    if prev:
        for k in ("llm",):
            if k in prev.get("ablation_deeplink", {}):
                report["ablation_deeplink"][k] = prev["ablation_deeplink"][k]
        for k in ("pointer", "generative"):
            if k in prev.get("ablation_extraction", {}):
                report["ablation_extraction"][k] = prev["ablation_extraction"][k]
        report["providers"] = prev["providers"]
        report["reused_llm_sections_from"] = {"file": args.reuse_llm_from, "generated": prev["generated"]}
    else:
        report["providers"] = hosted_vs_local(kit, args.profile_n)
    report["bench_seconds"] = round(time.time() - t_start, 1)
    (out / "bench" / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    from bench.metrics_md import write_metrics
    write_metrics(report, out / "metrics.md")
    print(f"[bench] done in {report['bench_seconds']} s -> results.jsonl, metrics.md, bench/report.json")


if __name__ == "__main__":
    main()
