"""Official-kit run: every data/input.txt complaint POSTed with its SIIS payload -> outputs/.

    python bench/official.py                 # uses the configured LLM routing (auto)
    SGTE_LLM_PROVIDER=none python bench/official.py

Writes
  outputs/<row id>.json   one response per complaint, exactly the data/sample_output.json shape {"query", "response"}
  outputs/all.json        the same 20 responses in input.txt order
  outputs/report.json     contract / rule / grounding / deeplink / latency / cache numbers (all measured here)
  outputs/README.md       the same, as a table

Each plan is built cold (cache bypassed) so it reflects its own complaint; latency is then measured cold, as a
repeat (payload-scoped cache), and for the other complaints that arrive with the same article.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import SETTINGS  # noqa: E402
from app.data_loader import load_kit  # noqa: E402
from app.extract import ground_score  # noqa: E402
from app.pipeline import Engine, contract_view  # noqa: E402
from app.siis_text import article_from_payload  # noqa: E402
from app.textutils import split_sentences  # noqa: E402
from app.validate import CHECKS, check_record  # noqa: E402
from bench.run import env_info, pct  # noqa: E402
from bench.scoring import deeplink_relevance, step_accuracy  # noqa: E402
from data.schema import ContextDeeplinkResponse  # noqa: E402

OUT = ROOT / "outputs"


def _steps(resp: dict):
    for g in resp["contexts"]:
        for a in g["actions"]:
            for sg in a["stepGroups"]:
                yield a, sg


def summarise_case(c, rec: dict, uris: set[str]) -> dict:
    resp = rec["response"]
    article = article_from_payload(c.siis_response)
    sentences = split_sentences(article.text)
    violations = check_record(rec, uris)
    try:
        ContextDeeplinkResponse.model_validate(resp)
        schema_ok = True
    except Exception:  # noqa: BLE001
        schema_ok = False
    steps = [s for _, sg in _steps(resp) for s in sg["steps"]]
    grounded = [ground_score(s, article.text, sentences) >= SETTINGS.grounding_min for s in steps]
    dls = [sg["actionableDeeplink"] for _, sg in _steps(resp) if sg["actionableDeeplink"]]
    vals = [sg["validationDeeplink"] for _, sg in _steps(resp) if sg["validationDeeplink"]]
    goals = resp["contexts"]
    ev = (rec["meta"].get("evidence") or {}).get("goals") or [{}]
    rel = ev[0].get("relevance") or []
    return {
        "id": c.id, "query": c.query, "siis_title": c.siis_response.get("title"), "siis_id": c.siis_id,
        "fallback": rec["meta"]["fallback"], "schema_valid": schema_ok, "violations": [str(v) for v in violations],
        "goal": goals[0]["goal"] if goals else None, "title": goals[0]["title"] if goals else None,
        "score": goals[0]["score"] if goals else None,
        "actions": [f"[{a['category']}] {a['actionName']}" for g in goals for a in g["actions"]],
        "steps": len(steps), "grounded_steps": sum(grounded), "deeplinks": len(dls),
        "dummy_deeplinks": sum(d["deeplink"] == SETTINGS.dummy_deeplink for d in dls),
        "validation_deeplinks": len(vals),
        "blocks_kept": sum(b["kept"] for b in rel), "blocks_scored": len(rel),
        "cold_ms": rec["meta"]["latency_ms"],
    }


def main() -> None:
    t_start = time.time()
    kit = load_kit()
    if not kit.cases:
        raise SystemExit("no official cases: data/input.txt + data/siis_responses.json are required")
    s = replace(SETTINGS, cache_dir=ROOT / "bench" / ".cache_official", log_dir=ROOT / "logs")
    eng = Engine(s, kit=kit)
    eng.cache.clear()
    OUT.mkdir(exist_ok=True)

    rows, views = [], []
    for c in kit.cases:  # 1) cold plan per complaint -> outputs/
        rec = eng.troubleshoot(c.query, c.siis_response, use_cache=False)
        view = contract_view(rec)
        (OUT / f"{c.id}.json").write_text(json.dumps(view, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        views.append(view)
        rows.append(summarise_case(c, rec, eng.uris))
    (OUT / "all.json").write_text(json.dumps(views, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # 2) cache: write each plan (cold path with cache on), then repeat it; then the other complaints that
    #    arrive with the same article (natural paraphrases in the official kit)
    first_ms, repeat_ms, repeat_hits = [], [], 0
    seen_article: dict[str, str] = {}
    cross = Counter()
    for c in kit.cases:
        before = seen_article.get(c.siis_id)
        rec = eng.troubleshoot(c.query, c.siis_response)
        if before is not None:
            cross["hit" if rec["meta"]["cache_hit"] else "miss"] += 1
        else:
            first_ms.append(rec["meta"]["latency_ms"])
        seen_article.setdefault(c.siis_id, c.id)
        again = eng.troubleshoot(c.query, c.siis_response)
        repeat_hits += again["meta"]["cache_hit"]
        repeat_ms.append(again["meta"]["latency_ms"])

    # 3) the reference query of sample_output.json (no payload ships with it -> free-text retrieval path)
    ref = kit.reference_outputs[0] if kit.reference_outputs else None
    ref_block = None
    if ref:
        rec = eng.troubleshoot(ref["query"], use_cache=False)
        view = contract_view(rec)
        (OUT / "sample_query.json").write_text(json.dumps(view, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        gold = ref["response"]["contexts"][0]
        pred = view["response"]["contexts"][0] if view["response"]["contexts"] else None
        ref_block = {"query": ref["query"], "reference_check_violations": [str(v) for v in check_record(ref, eng.uris)],
                     "our_goal": pred["goal"] if pred else None, "our_actions": [a["actionName"] for a in (pred or {}).get("actions", [])],
                     "step_accuracy_vs_reference": step_accuracy(pred, gold),
                     "deeplink_relevance_vs_reference": deeplink_relevance(pred, gold, kit.catalog, SETTINGS.dummy_deeplink)["score"],
                     "siis": [g.get("siis") for g in rec["meta"]["evidence"].get("goals", [])],
                     "note": "sample_output.json ships without its SIIS payload, so this plan is retrieved from the 11 "
                             "official articles; its back-up step comes from an article that is not in the kit."}

    n = len(rows)
    steps = sum(r["steps"] for r in rows)
    viol = Counter(v.split(" @ ")[0] for r in rows for v in r["violations"])
    report = {
        "generated": datetime.now().isoformat(timespec="seconds"), "env": env_info(),
        "llm": eng.llm.model_name if eng.llm else "none", "extract_mode": eng.extract_mode(),
        "enrich_mode": eng.enrich_mode(), "catalog_entries": len(kit.catalog),
        "catalog_synthetic": kit.catalog_synthetic, "cases": n, "distinct_articles": len(kit.siis),
        "schema_valid_pct": round(100 * sum(r["schema_valid"] for r in rows) / n, 1),
        "rule_compliance_pct": round(100 * sum(not r["violations"] for r in rows) / n, 1),
        "violations_by_check": {c: viol.get(c, 0) for c in CHECKS},
        "plans": sum(r["fallback"] is None for r in rows), "fallbacks": dict(Counter(r["fallback"] for r in rows if r["fallback"])),
        "steps": steps, "grounded_steps_pct": round(100 * sum(r["grounded_steps"] for r in rows) / steps, 1) if steps else None,
        "actions": sum(len(r["actions"]) for r in rows),
        "actions_by_category": dict(Counter(a.split("]")[0][1:] for r in rows for a in r["actions"])),
        "deeplinks": sum(r["deeplinks"] for r in rows), "dummy_deeplinks": sum(r["dummy_deeplinks"] for r in rows),
        "validation_deeplinks": sum(r["validation_deeplinks"] for r in rows),
        "latency_ms": {"cold": {"p50": pct([r["cold_ms"] for r in rows], 50), "p95": pct([r["cold_ms"] for r in rows], 95)},
                       "first_with_cache_write": {"p50": pct(first_ms, 50), "p95": pct(first_ms, 95)},
                       "repeat_cache_hit": {"n": len(repeat_ms), "hits": repeat_hits, "p50": pct(repeat_ms, 50),
                                            "p95": pct(repeat_ms, 95)}},
        "same_article_other_complaint": dict(cross),
        "score": {"mean": round(statistics.mean(r["score"] for r in rows if r["score"] is not None), 3),
                  "min": min(r["score"] for r in rows if r["score"] is not None),
                  "max": max(r["score"] for r in rows if r["score"] is not None)},
        "reference_sample": ref_block,
        "rows": rows,
        "bench_seconds": round(time.time() - t_start, 1),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_readme(report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "env", "reference_sample")}, indent=1))


def write_readme(r: dict) -> None:
    L = ["# Official-kit outputs", "",
         f"Generated {r['generated']} by `bench/official.py` ({r['bench_seconds']} s) · LLM routing: `{r['llm']}`, "
         f"extraction `{r['extract_mode']}`, enrichment `{r['enrich_mode']}`.", "",
         "One file per `data/input.txt` complaint, each the response to `POST /v1/troubleshoot` with that row's "
         "`siis_response` payload, in exactly the `data/sample_output.json` shape (`?view=contract`).", "",
         "| Metric | Value |", "|---|---|",
         f"| Complaints / distinct SIIS articles | {r['cases']} / {r['distinct_articles']} |",
         f"| Schema-valid (`data/schema.py`) | {r['schema_valid_pct']}% |",
         f"| All 11 rule checks pass | {r['rule_compliance_pct']}% |",
         f"| Plans / fallbacks | {r['plans']} / {r['fallbacks'] or 0} |",
         f"| Steps grounded in the payload (fuzzy >= {SETTINGS.grounding_min:.0f}) | {r['grounded_steps_pct']}% of {r['steps']} |",
         f"| Actions (auto / manual / critical) | {r['actions']} ({', '.join(f'{k} {v}' for k, v in r['actions_by_category'].items())}) |",
         f"| Actionable deeplinks (dummy_positive) / validation deeplinks | {r['deeplinks']} ({r['dummy_deeplinks']}) / {r['validation_deeplinks']} |",
         f"| Cold latency P50 / P95 | {r['latency_ms']['cold']['p50']} / {r['latency_ms']['cold']['p95']} ms |",
         f"| Repeat (payload-scoped cache) hits, P95 | {r['latency_ms']['repeat_cache_hit']['hits']}/{r['latency_ms']['repeat_cache_hit']['n']}, "
         f"{r['latency_ms']['repeat_cache_hit']['p95']} ms |",
         f"| Other complaint, same article: cache hit / miss | {r['same_article_other_complaint'].get('hit', 0)} / "
         f"{r['same_article_other_complaint'].get('miss', 0)} |",
         f"| Goal score mean (min-max) | {r['score']['mean']} ({r['score']['min']}-{r['score']['max']}) |", "",
         "| Row | SIIS article | Goal | Actions | Steps | Deeplinks | Score |", "|---|---|---|---|---|---|---|"]
    for x in r["rows"]:
        goal = (x["goal"] or x["fallback"] or "").replace("Follow these steps to perform this ", "")
        L.append(f"| [{x['id']}]({x['id']}.json) | {x['siis_title']} | {goal} | {len(x['actions'])} | {x['steps']} | "
                 f"{x['deeplinks']} | {x['score']} |")
    if r.get("reference_sample"):
        rs = r["reference_sample"]
        L += ["", "## Reference sample", "",
              f"`data/sample_output.json` passes every check ({len(rs['reference_check_violations'])} violations). Its query "
              f"has no SIIS payload in the kit, so [sample_query.json](sample_query.json) is our free-text answer "
              f"(retrieval over the official articles): {rs['our_goal']}; step accuracy vs the reference "
              f"{rs['step_accuracy_vs_reference']['score']}/3, deeplink relevance {rs['deeplink_relevance_vs_reference']}/2. "
              f"{rs['note']}"]
    L += ["", "The deeplink catalog (`data/deeplinks.json`) is synthetic except the Back up data (TechCorp Cloud) "
          "entry copied from `data/sample_output.json`: the official kit ships no catalog."]
    (OUT / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
