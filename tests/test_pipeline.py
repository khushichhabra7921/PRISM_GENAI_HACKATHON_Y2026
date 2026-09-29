"""End-to-end pipeline tests (deterministic: no LLM). Loads the embedding + reranker models once."""
import copy

import pytest

from app.config import Settings
from app.data_loader import load_kit
from app.pipeline import Engine
from app.validate import GOAL_RE, check_record
from bench.scoring import deeplink_relevance, step_accuracy

KIT = load_kit()


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    s = Settings(llm_provider="none", cache_dir=tmp_path_factory.mktemp("cache"),
                 log_dir=tmp_path_factory.mktemp("logs"))
    return Engine(s, kit=KIT)


def _uris():
    return {e.uri for e in KIT.catalog}


GOLD_WITH_PLAN = [n for n, s in sorted(KIT.samples.items()) if s["output"]["response"]["contexts"]]


@pytest.mark.parametrize("name", GOLD_WITH_PLAN)
def test_gold_sample_reproduced(engine, name):
    sample = KIT.samples[name]
    rec = engine.troubleshoot(sample["input"]["query"], sample["input"]["siis_response"])
    assert check_record(rec, _uris()) == []
    assert rec["meta"]["fallback"] is None
    gold = sample["output"]["response"]["contexts"][0]
    pred = rec["response"]["contexts"][0]
    acc = step_accuracy(pred, gold)
    rel = deeplink_relevance(pred, gold, KIT.catalog, "bixby://dummy_positive")
    assert acc["score"] >= 2.7, acc
    assert rel["score"] >= 1.8, rel
    m, gm = GOAL_RE.match(pred["goal"]), GOAL_RE.match(gold["goal"])
    assert m and m.group(2) == gm.group(2)
    assert set(m.group(1).lower().split()) & set(gm.group(1).lower().split())  # same topic, wording may differ


def test_gold_no_match_sample(engine):
    s = KIT.samples["sample_05"]
    rec = engine.troubleshoot(s["input"]["query"])
    assert rec["response"]["contexts"] == [] and rec["meta"]["fallback"] == "no_match"
    assert 8 <= len(rec["query_variations"]) <= 10


def test_empty_siis_is_no_siis_context(engine):
    rec = engine.troubleshoot("my battery drains fast", siis_response="   ")
    assert rec["response"]["contexts"] == [] and rec["meta"]["fallback"] == "no_siis_context"


def test_siis_without_instructions_is_no_match(engine):
    rec = engine.troubleshoot("battery drains fast", siis_response="Batteries age over time. Heat matters.")
    assert rec["response"]["contexts"] == [] and rec["meta"]["fallback"] == "no_match"


def test_url_in_siis_never_leaks(engine):
    text = KIT.samples["sample_01"]["input"]["siis_response"] + \
        " Visit https://www.samsung.com/support or [help](http://x.y) for more. Go to samsung.com/in/support."
    rec = engine.troubleshoot("swipe navigation broken", siis_response=text)
    assert check_record(rec, _uris()) == []
    blob = str(rec["response"]) + str(rec["query_variations"])
    assert "http" not in blob and "www." not in blob and "samsung.com" not in blob


def test_critical_last_and_manual_without_deeplink(engine):
    s = KIT.samples["sample_04"]
    rec = engine.troubleshoot(s["input"]["query"], s["input"]["siis_response"])
    acts = rec["response"]["contexts"][0]["actions"]
    cats = [a["category"] for a in acts]
    assert cats == sorted(cats, key={"auto": 0, "manual": 1, "critical": 2}.get) and cats[-1] == "critical"
    for a in acts:
        if a["category"] == "manual":
            assert all(sg["actionableDeeplink"] is None for sg in a["stepGroups"])


def test_every_deeplink_in_catalog(engine):
    for q in [q for q in KIT.queries if q.type == "single"][:15]:
        rec = engine.troubleshoot(q.query, write_cache=False)
        for g in rec["response"]["contexts"]:
            for a in g["actions"]:
                for sg in a["stepGroups"]:
                    d = sg["actionableDeeplink"]
                    assert d is None or d["deeplink"] in _uris() or d["deeplink"] == "bixby://dummy_positive"


def test_multi_symptom_returns_two_goals(engine):
    rec = engine.troubleshoot("my screen flickers and the battery dies fast", write_cache=False)
    assert len(rec["response"]["contexts"]) == 2, rec["meta"]["evidence"]
    assert rec["meta"]["evidence"]["multi_symptom"] is True


def test_repeat_query_served_from_cache_fast(engine):
    q = "phone keeps freezing"
    first = engine.troubleshoot(q)
    assert first["meta"]["cache_hit"] is False
    second = engine.troubleshoot(q)
    assert second["meta"]["cache_hit"] is True and second["meta"]["llm_calls"] == 0
    assert second["meta"]["latency_ms"] < 300
    assert second["response"] == first["response"]


def test_evidence_does_not_break_schema(engine):
    rec = engine.troubleshoot("screen dim", write_cache=False)
    assert "evidence" in rec["meta"]
    ev = rec["meta"]["evidence"]["goals"][0]["actions"]
    assert all("source_sentences" in a for a in ev)
    assert check_record(copy.deepcopy(rec), _uris()) == []


def test_multi_symptom_not_answered_by_single_plan(engine):
    engine.troubleshoot("my screen keeps flickering")  # caches a single-goal plan
    rec = engine.troubleshoot("my screen keeps flickering and the battery drains fast", write_cache=False)
    assert len(rec["response"]["contexts"]) == 2, rec["meta"]["evidence"]
