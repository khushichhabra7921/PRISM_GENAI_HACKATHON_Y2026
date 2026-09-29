"""Orchestration: ① enrich → ② cache (raw, then canonical) → ③ SIIS RAG → ④ extract → ⑤ order →
⑥ deeplinks → ⑦ validate + fix (+ one retry) → cache → plan. Multi-symptom split, evidence, timing."""
from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from app import models
from app.cache import CacheHit, SemanticCache
from app.config import SETTINGS, Settings
from app.data_loader import Kit, SiisDoc, load_kit
from app.deeplinks import ActionQuery, DeeplinkEngine
from app.enrich import Enrichment, enrich, llm_enrich, normalise, rule_variations, split_symptoms
from app.extract import Extraction, extract
from app.llm_client import LLMClient, Usage
from app.retrieve import SiisHit, SiisIndex
from app.textutils import scrub_urls
from app.validate import check_record, validate_and_fix

TIER_VALUE = {"high": 1.0, "medium": 0.75, "low": 0.4}
RULES_MODEL = "rules-v1 (deterministic, no LLM)"


@dataclass
class Grounding:
    query: str
    text: str
    hit: Optional[SiisHit]  # None when the caller supplied siis_response
    doc: Optional[SiisDoc]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Engine:
    def __init__(self, s: Settings = SETTINGS, kit: Optional[Kit] = None, llm: Optional[LLMClient] = None,
                 cache_dir=None):
        self.s = s
        self.kit = kit or load_kit()
        self.known = {" > ".join(e.path) for e in self.kit.catalog if e.path}
        self.uris = {e.uri for e in self.kit.catalog}
        self.llm = llm if llm is not None else (LLMClient(s) if s.llm_provider != "none" else None)
        self.deeplinks = DeeplinkEngine(self.kit.catalog, llm=self.llm)
        self.siis = SiisIndex(self.kit.siis)
        self.cache = SemanticCache(cache_dir or s.cache_dir, dim=int(models.embed_cache(["x"]).shape[1]),
                                   tau=s.cache_tau)
        models.warm()
        self._bg = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sgte-bg")  # serialises SLM calls
        self._pending: list[Future] = []
        self._pending_lock = threading.Lock()
        self._queued: set[str] = set()
        self.ready = True

    # ============================================================== LLM routing (auto modes)
    def llm_is_fast(self) -> bool:
        """True when the first reachable provider is hosted (seconds per call); a CPU-local SLM is not."""
        return bool(self.llm) and self.llm.model_name.split("/")[0] in ("anthropic", "openai")

    def enrich_mode(self, override: Optional[str] = None) -> str:
        m = override or self.s.enrich_mode
        if m != "auto":
            return m
        if not self.llm or not self.llm.available():
            return "rules"
        return "llm" if self.llm_is_fast() else "background"

    def extract_mode(self, override: Optional[str] = None) -> str:
        m = override or self.s.extract_mode
        if m != "auto":
            return m
        return "pointer" if self.llm and self.llm.available() and self.llm_is_fast() else "rules"

    # ============================================================== background cache upgrade
    def _upgrade_job(self, plan_id: str, query: str) -> None:
        """SLM paraphrases for an already-validated plan become extra cache keys (the plan's steps never change)."""
        e = llm_enrich(query, self.llm, Usage())
        plan = self.cache.plans.get(plan_id)
        if not e or not plan:
            return
        rec = {"query": plan["query"], "query_variations": e.query_variations, "response": plan["response"]}
        if check_record(rec, self.uris):  # 8-10 distinct variations, no URLs
            return
        keys = [e.canonical_query] + e.query_variations
        plan["query_variations"] = e.query_variations
        plan["variations_source"] = self.llm.model_name
        self.cache.add(plan_id, plan, keys, models.embed_cache(keys))

    def schedule_upgrade(self, plan_id: str, query: str) -> None:
        with self._pending_lock:
            self._pending = [f for f in self._pending if not f.done()]
            if plan_id in self._queued:  # already waiting for its SLM paraphrases
                return
            self._queued.add(plan_id)
            fut = self._bg.submit(self._upgrade_job, plan_id, query)
            fut.add_done_callback(lambda _f, pid=plan_id: self._queued.discard(pid))
            self._pending.append(fut)

    def upgrade_all(self) -> int:
        """Queue SLM paraphrases for every cached plan whose variations are still template-generated."""
        todo = [(pid, p["query"]) for pid, p in list(self.cache.plans.items())
                if p.get("variations_source", "rules") == "rules"]
        for pid, q in todo:
            self.schedule_upgrade(pid, q)
        return len(todo)

    def drain(self, save: bool = True) -> None:
        with self._pending_lock:
            pending = list(self._pending)
        for f in pending:
            f.exception()
        if save:
            self.cache.save()

    @property
    def pending_upgrades(self) -> int:
        with self._pending_lock:
            return sum(not f.done() for f in self._pending)

    # ============================================================== public
    def troubleshoot(self, query: str, siis_response: Optional[str] = None, use_cache: bool = True,
                     dl_mode: str = "hybrid", extract_mode: Optional[str] = None, write_cache: bool = True,
                     cache_only: bool = False, enrich_mode: Optional[str] = None) -> Optional[dict]:
        """Full request path. cache_only=True returns None on a raw-lookup miss (benchmarking the fast path)."""
        t0 = time.perf_counter()
        usage, timings = Usage(), {}
        query = scrub_urls((query or "").strip())
        tick = time.perf_counter()
        qv = models.embed_cache([query])[0]
        timings["embed_ms"] = _ms(tick)
        if siis_response is None and use_cache:
            hit = self.cache.lookup(qv)
            multi = len(split_symptoms(normalise(query))) > 1
            if hit and _compatible(hit, multi):
                rec = self._from_cache(query, hit, "raw", t0, timings)
                if rec:
                    return rec
            anchored = self._anchor_lookup(query, qv)
            if anchored:
                hit, anchor_ev = anchored
                rec = self._from_cache(query, hit, "article", t0, timings, extra=anchor_ev)
                if rec:
                    return rec
        if cache_only:
            return None

        tick = time.perf_counter()
        e_mode = self.enrich_mode(enrich_mode)
        enr = enrich(query, self.llm, usage, "rules" if e_mode == "background" else e_mode)
        timings["enrich_ms"] = _ms(tick)
        if enr.canonical_query.lower() != query.lower():
            if siis_response is None and use_cache:
                hit = self.cache.lookup(models.embed_cache([enr.canonical_query])[0])
                if hit and _compatible(hit, len(enr.symptoms) > 1):
                    rec = self._from_cache(query, hit, "canonical", t0, timings, usage, enr)
                    if rec:
                        return rec

        if siis_response is not None and not siis_response.strip():
            return self._fallback(query, enr, "no_siis_context", t0, usage, timings, {})
        tick = time.perf_counter()
        groundings = ([Grounding(enr.canonical_query, siis_response, None, self._known_article(siis_response))]
                      if siis_response is not None else self._resolve(enr))
        timings["retrieve_ms"] = _ms(tick)
        if not groundings:
            return self._fallback(query, enr, "no_match", t0, usage, timings, {"reason": "no SIIS article passed the gate"})

        extract_mode = self.extract_mode(extract_mode)
        record, evidence, llm_extract = self._plan(query, enr, groundings, usage, timings, dl_mode, extract_mode)
        fixed, errors = validate_and_fix(record, self.uris, pad=lambda q: rule_variations(q, enr.canonical_query),
                                         intent=_intent(enr))
        if errors and llm_extract:  # one retry with the error list fed back
            record, evidence, _ = self._plan(query, enr, groundings, usage, timings, dl_mode, extract_mode,
                                             feedback=sorted({str(e) for e in errors})[:6])
            fixed, errors = validate_and_fix(record, self.uris, pad=lambda q: rule_variations(q, enr.canonical_query),
                                             intent=_intent(enr))
            evidence["retried"] = True
        if errors or not fixed["response"]["contexts"]:
            evidence["validation_errors"] = [str(e) for e in errors][:10]
            return self._fallback(query, enr, "no_match", t0, usage, timings, evidence)

        model = ", ".join(usage.models) or RULES_MODEL
        if siis_response is None and use_cache and write_cache:
            plan_id = self._store(query, enr, fixed, evidence, model, groundings)
            if e_mode == "background" and enr.source == "rules":
                self.schedule_upgrade(plan_id, query)
        return self._finish(fixed, t0, usage, timings, cache_hit=False, model=model, evidence=evidence, enr=enr)

    def _anchor_lookup(self, query: str, qv: np.ndarray):
        """Zero-LLM tier ②b: reuse the validated plan of the article this query confidently maps to."""
        if not self.s.anchor_cache or len(split_symptoms(normalise(query))) > 1:
            return None
        dense = self.siis._dense(models.embed([query])[0])
        order = np.argsort(-dense)
        top, runner = float(dense[order[0]]), float(dense[order[1]]) if len(order) > 1 else 0.0
        if top < self.s.anchor_min_dense or top - runner < self.s.anchor_margin:
            return None
        doc = self.siis.docs[int(order[0])]
        hit = self.cache.best_plan_for_article(qv, doc.id)
        if not hit:
            return None
        return hit, {"anchor_article": doc.id, "anchor_dense": round(top, 4), "anchor_margin": round(top - runner, 4)}

    # ============================================================== retrieval / multi-symptom
    def _known_article(self, text: str) -> Optional[SiisDoc]:
        """If caller-supplied text is (nearly) a known SIIS article, borrow its title for the goal topic.
        The supplied text itself remains the only grounding boundary."""
        sims = self.siis.doc_emb @ models.embed([text])[0]
        i = int(np.argmax(sims))
        return self.siis.docs[i] if float(sims[i]) >= 0.85 else None

    def _resolve(self, enr: Enrichment) -> list[Grounding]:
        if len(enr.symptoms) > 1:
            parts: list[Grounding] = []
            for sym in enr.symptoms:
                hit, _ = self.siis.best(sym)
                if hit and all(p.doc.id != hit.doc.id for p in parts):
                    parts.append(Grounding(sym, hit.doc.text, hit, hit.doc))
            if len(parts) > 1:
                return parts
        hit, _ = self.siis.best(enr.canonical_query)
        if hit is None and enr.canonical_query != enr.query:
            hit, _ = self.siis.best(enr.query)
        return [Grounding(enr.canonical_query, hit.doc.text, hit, hit.doc)] if hit else []

    # ============================================================== plan construction
    def _plan(self, query: str, enr: Enrichment, groundings: list[Grounding], usage: Usage, timings: dict,
              dl_mode: str, extract_mode: Optional[str], feedback: Optional[list[str]] = None):
        goals, ev_goals, used_llm = [], [], False
        for g in groundings:
            tick = time.perf_counter()
            ex = extract(g.text, g.query, self.known, g.doc.title if g.doc else "", self.llm, usage,
                         extract_mode or "rules", feedback)
            timings["extract_ms"] = timings.get("extract_ms", 0) + _ms(tick)
            if ex is None or not ex.viable or not ex.actions:
                ev_goals.append({"query": g.query, "siis": g.hit.evidence() if g.hit else None,
                                 "dropped": ex.note if ex else "no actionable sentences"})
                continue
            used_llm = used_llm or ex.mode != "rules"
            tick = time.perf_counter()
            goal, ev = self._goal(g, ex, enr, dl_mode)
            timings["deeplink_ms"] = timings.get("deeplink_ms", 0) + _ms(tick)
            if goal:
                goals.append(goal)
            ev_goals.append(ev)
        record = {"query": query, "query_variations": enr.query_variations, "response": {"contexts": goals}}
        evidence = {"enrichment": {"source": enr.source, "canonical_query": enr.canonical_query, "topic": enr.topic,
                                   "intent": enr.intent, "symptoms": enr.symptoms},
                    "goals": ev_goals, "multi_symptom": len(groundings) > 1}
        return record, evidence, used_llm

    def _goal(self, g: Grounding, ex: Extraction, enr: Enrichment, dl_mode: str) -> tuple[Optional[dict], dict]:
        linkable = [a for a in ex.actions if a.path and a.category in ("auto", "critical")]
        matches = dict(zip(map(id, linkable), self.deeplinks.match_many(
            [ActionQuery(a.name, a.steps, a.path) for a in linkable], dl_mode))) if linkable else {}
        actions, ev_actions, tiers = [], [], []
        for a in ex.actions:
            m = matches.get(id(a))
            dl = m.as_schema() if m else None
            if m:
                tiers.append(TIER_VALUE[m.tier] if m.uri else 0.3)
            actions.append({"actionName": a.name, "description": a.description, "category": a.category,
                            "stepGroups": [{"steps": a.steps, "validationDeeplink": None, "actionableDeeplink": dl}]})
            ev_actions.append({"actionName": a.name, "category": a.category, "source_sentences": a.sources,
                               "menu_path": " > ".join(a.path) if a.path else None,
                               "deeplink": m.evidence if m else None})
        ret = g.hit.rerank if g.hit else float(models.rerank_pairs([(g.query, g.text)])[0])
        dense = g.hit.dense if g.hit else 0.75
        s_ret = 0.5 * _sigmoid(ret / 3.0) + 0.5 * min(1.0, max(0.0, (dense - 0.35) / 0.45))
        s_ground = float(np.mean(ex.grounding)) / 100.0 if ex.grounding else 0.0
        s_dl = float(np.mean(tiers)) if tiers else 0.8
        raw = 0.45 * s_ret + 0.25 * s_ground + 0.30 * s_dl
        score = round(min(1.0, max(0.0, self.s.score_a * raw + self.s.score_b)), 2)
        kind = "Configuration" if enr.intent == "configuration" else "Troubleshooting"
        goal = {"goal": f"Follow these steps to perform this {ex.topic} {kind}", "title": ex.title,
                "actions": actions, "score": score}
        ev = {"query": g.query, "siis": g.hit.evidence() if g.hit else {"source": "request.siis_response"},
              "extract_mode": ex.mode, "dropped_ungrounded_steps": ex.dropped_steps,
              "score_parts": {"retrieval": round(s_ret, 3), "grounding": round(s_ground, 3),
                              "deeplink": round(s_dl, 3), "raw": round(raw, 3)},
              "actions": ev_actions}
        return goal, ev

    # ============================================================== cache / output
    def _store(self, query: str, enr: Enrichment, record: dict, evidence: dict, model: str,
               groundings: list[Grounding]) -> str:
        keys = [query, enr.canonical_query] + record["query_variations"]
        plan_id = hashlib.sha1(enr.canonical_query.lower().encode()).hexdigest()[:12]
        plan = {"query": query, "canonical_query": enr.canonical_query, "query_variations": record["query_variations"],
                "response": record["response"], "evidence": evidence, "model": model,
                "siis_ids": [g.doc.id for g in groundings if g.doc], "variations_source": enr.source}
        self.cache.add(plan_id, plan, keys, models.embed_cache(keys))
        return plan_id

    def _from_cache(self, query: str, hit: CacheHit, kind: str, t0: float, timings: dict,
                    usage: Optional[Usage] = None, enr: Optional[Enrichment] = None,
                    extra: Optional[dict] = None) -> Optional[dict]:
        plan = hit.plan
        record = {"query": query, "query_variations": plan["query_variations"], "response": plan["response"]}
        if check_record(record, self.uris):  # never serve a plan that fails validation
            return None
        ev = {"from_cache": True, "lookup": kind, "cache_score": round(hit.score, 4), "cache_key": hit.key,
              "cached_query": plan["query"], "siis_ids": plan.get("siis_ids", []), **(extra or {}),
              "plan_evidence": plan.get("evidence")}
        return self._finish(record, t0, usage or Usage(), timings, cache_hit=True, model=plan.get("model", ""),
                            evidence=ev, enr=enr)

    def _fallback(self, query: str, enr: Enrichment, reason: str, t0: float, usage: Usage, timings: dict,
                  evidence: dict) -> dict:
        record = {"query": query, "query_variations": [], "response": {"contexts": []}}
        fixed, _ = validate_and_fix(record, self.uris, pad=lambda q: rule_variations(q, enr.canonical_query))
        fixed["query_variations"] = (enr.query_variations or fixed["query_variations"])[:10]
        self._log_no_match(query, enr, reason, evidence)
        return self._finish(fixed, t0, usage, timings, cache_hit=False, model=", ".join(usage.models) or RULES_MODEL,
                            evidence=evidence, enr=enr, fallback=reason)

    def _log_no_match(self, query: str, enr: Enrichment, reason: str, evidence: dict) -> None:
        self.s.log_dir.mkdir(parents=True, exist_ok=True)
        row = {"ts": datetime.now(timezone.utc).isoformat(), "query": query, "canonical_query": enr.canonical_query,
               "topic": enr.topic, "reason": reason, "detail": evidence.get("reason") or evidence.get("validation_errors")}
        with open(self.s.log_dir / "no_match.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _finish(self, record: dict, t0: float, usage: Usage, timings: dict, cache_hit: bool, model: str,
                evidence: dict, enr: Optional[Enrichment], fallback: Optional[str] = None) -> dict:
        record["meta"] = {"latency_ms": _ms(t0), "cache_hit": cache_hit, "model": model, "cost_usd": usage.cost_usd,
                          "fallback": fallback, "timings_ms": timings, **usage.summary(), "evidence": evidence}
        return record


def _compatible(hit: CacheHit, multi_symptom: bool) -> bool:
    """A multi-symptom complaint must not be answered by a single-goal plan, nor vice versa."""
    return (len(hit.plan.get("siis_ids", [])) > 1) == multi_symptom


def _ms(t: float) -> float:
    return round((time.perf_counter() - t) * 1000, 1)


def _intent(enr: Enrichment) -> str:
    return "Configuration" if enr.intent == "configuration" else "Troubleshooting"
