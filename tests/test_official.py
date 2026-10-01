"""End-to-end tests on the OFFICIAL kit: the 20 input.txt complaints, each POSTed with its SIIS payload
({"title", "content"}), deterministic (no LLM)."""
import json

import pytest

from app.config import Settings
from app.data_loader import load_kit
from app.extract import ground_score
from app.pipeline import Engine, contract_view
from app.siis_text import article_from_payload
from app.textutils import split_sentences
from app.validate import check_record
from data.schema import ContextDeeplinkResponse

KIT = load_kit()
DUMMY = Settings().dummy_deeplink
CASES = {c.id: c for c in KIT.cases}
REF = KIT.reference_outputs[0]


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    s = Settings(llm_provider="none", cache_dir=tmp_path_factory.mktemp("cache"),
                 log_dir=tmp_path_factory.mktemp("logs"))
    return Engine(s, kit=KIT)


def _keys(obj, path=()):
    """Key order of every dict in the plan, e.g. ('contexts', 0) -> ['goal', 'title', 'score', 'actions']."""
    if isinstance(obj, dict):
        yield path, list(obj)
        for k, v in obj.items():
            yield from _keys(v, path + (k,))
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v, path + ("*",))


@pytest.mark.parametrize("cid", sorted(CASES))
def test_official_case_is_valid_and_grounded(engine, cid):
    c = CASES[cid]
    rec = engine.troubleshoot(c.query, c.siis_response, write_cache=False)
    assert rec["meta"]["fallback"] is None, rec["meta"]["evidence"]
    assert rec["query"] == c.query
    ContextDeeplinkResponse.model_validate(rec["response"])
    assert check_record(rec, engine.uris) == []
    article = article_from_payload(c.siis_response)
    sentences = split_sentences(article.text)
    for g in rec["response"]["contexts"]:
        for a in g["actions"]:
            for sg in a["stepGroups"]:
                for step in sg["steps"]:  # never invented: every step is supported by the payload text
                    assert ground_score(step, article.text, sentences) >= 80, step
                for key in ("actionableDeeplink", "validationDeeplink"):
                    d = sg[key]
                    assert d is None or d["deeplink"] in engine.uris or d["deeplink"] == DUMMY
                if a["category"] == "manual":
                    assert sg["actionableDeeplink"] is None and sg["validationDeeplink"] is None
    assert "http" not in json.dumps(rec["response"])


def test_key_order_matches_official_sample(engine):
    c = CASES["row_21"]  # has auto actions with deeplinks
    view = contract_view(engine.troubleshoot(c.query, c.siis_response, write_cache=False))
    ref_keys = {tuple(p for p in path if p != "*"): ks for path, ks in _keys(REF)}
    got_keys = {tuple(p for p in path if p != "*"): ks for path, ks in _keys(view)}
    assert list(view) == list(REF) == ["query", "response"]
    for path in [("response", "contexts"), ("response", "contexts", "actions"),
                 ("response", "contexts", "actions", "stepGroups")]:
        assert got_keys[path] == ref_keys[path], path
    dl = next(sg["actionableDeeplink"] for g in view["response"]["contexts"] for a in g["actions"]
              for sg in a["stepGroups"] if sg["actionableDeeplink"])
    assert list(dl) == list(REF["response"]["contexts"][0]["actions"][0]["stepGroups"][0]["actionableDeeplink"])


def test_multi_topic_article_keeps_only_the_relevant_tip(engine):
    """'Screen looks small' + a long screen-mirroring article -> the aspect-ratio tip, not the PC/TV tours."""
    c = CASES["row_8"]
    rec = engine.troubleshoot(c.query, c.siis_response, write_cache=False)
    steps = " ".join(s for g in rec["response"]["contexts"] for a in g["actions"] for sg in a["stepGroups"]
                     for s in sg["steps"])
    assert "Phone aspect ratio" in steps
    assert "Windows" not in steps and "Bluetooth and other device settings" not in steps


def test_numbered_procedure_kept_whole_and_critical_last(engine):
    c = CASES["row_21"]
    rec = engine.troubleshoot(c.query, c.siis_response, write_cache=False)
    acts = rec["response"]["contexts"][0]["actions"]
    cats = [a["category"] for a in acts]
    assert cats == sorted(cats, key={"auto": 0, "manual": 1, "critical": 2}.get)
    names = [a["actionName"] for a in acts]
    assert "Start Safe Mode" in names and "Perform a Factory Data Reset" in names and cats[-1] == "critical"
    reset = next(a for a in acts if a["actionName"] == "Perform a Factory Data Reset")
    assert reset["stepGroups"][0]["steps"][:4] == ["Navigate to and open Settings.", "Tap on General management.",
                                                   "Tap on Reset.", "Tap on Factory data reset."]
    assert reset["stepGroups"][0]["actionableDeeplink"]["description"].startswith("Open factory data reset")


def test_two_navigations_become_two_step_groups(engine):
    c = CASES["row_1"]
    rec = engine.troubleshoot(c.query, c.siis_response, write_cache=False)
    clear = next(a for g in rec["response"]["contexts"] for a in g["actions"] if "Cache" in a["actionName"])
    assert len(clear["stepGroups"]) == 2
    assert clear["stepGroups"][0]["steps"][-1] == "Tap Clear cache." and "Tap Clear data." in clear["stepGroups"][1]["steps"]
    assert all(sg["actionableDeeplink"] for sg in clear["stepGroups"])


def test_payload_cache_is_scoped_to_the_article(engine):
    c, other = CASES["row_14"], CASES["row_20"]
    first = engine.troubleshoot(c.query, c.siis_response)
    again = engine.troubleshoot(c.query, c.siis_response)
    assert first["meta"]["cache_hit"] is False and again["meta"]["cache_hit"] is True
    assert again["meta"]["evidence"]["lookup"] == "payload" and again["meta"]["llm_calls"] == 0
    assert again["response"] == first["response"]
    swapped = engine.troubleshoot(c.query, other.siis_response, write_cache=False)
    assert swapped["meta"]["cache_hit"] is False  # same complaint, different article: never the cached plan
    assert swapped["response"] != first["response"]


def test_numbered_quoted_complaint_is_cleaned_but_echoed(engine):
    c = CASES["row_19"]
    rec = engine.troubleshoot(c.query, c.siis_response, write_cache=False)
    assert rec["query"] == c.query and rec["query"].startswith('1. "')
    assert '"' not in rec["meta"]["evidence"]["enrichment"]["canonical_query"]


def test_empty_payload_is_no_siis_context(engine):
    rec = engine.troubleshoot("screen is black", {"title": "Blank screen", "content": "   "})
    assert rec["response"]["contexts"] == [] and rec["meta"]["fallback"] == "no_siis_context"


def test_payload_without_instructions_is_no_match(engine):
    payload = {"title": "Bleeding pixels", "content": "## Bleeding Pixels\nBleeding pixels are caused by an impact."}
    rec = engine.troubleshoot("lines on my screen", payload)
    assert rec["response"]["contexts"] == [] and rec["meta"]["fallback"] == "no_match"


def test_free_text_query_retrieves_an_official_article(engine):
    rec = engine.troubleshoot("my touchscreen is laggy and slow to respond", write_cache=False)
    assert rec["meta"]["fallback"] is None
    ids = [g["siis"].get("siis_id") for g in rec["meta"]["evidence"]["goals"]]
    assert "row_21" in ids
