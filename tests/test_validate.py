"""Contract, rule checks, auto-fixers, data loader (synthetic gold samples + the official sample output)."""
import copy
import json

import pytest

from app.data_loader import load_kit, load_synthetic_kit
from app.textutils import fit_description, has_url, is_title_case, scrub_urls, title_case
from app.validate import CHECKS, check_record, fix_record, validate_and_fix
from data.schema import ContextDeeplinkResponse

KIT = load_synthetic_kit()
URIS = {e.uri for e in KIT.catalog} | {e.val_uri for e in KIT.catalog if e.val_uri}
OFFICIAL = load_kit()


def _gold(name):
    return copy.deepcopy(KIT.samples[name]["output"])


def test_eleven_named_checks():
    assert len(CHECKS) == 11


@pytest.mark.parametrize("name", sorted(KIT.samples))
def test_gold_samples_pass_every_check(name):
    rec = _gold(name)
    ContextDeeplinkResponse.model_validate(rec["response"])
    assert check_record(rec, URIS) == []


def test_no_match_sample_is_empty_with_flag():
    rec = _gold("sample_05")
    assert rec["response"]["contexts"] == [] and rec["meta"]["fallback"] == "no_match"


def test_goal_and_title_fixed():
    rec = _gold("sample_01")
    g = rec["response"]["contexts"][0]
    g["goal"] = "follow these steps to perform this swipe navigation troubleshooting."
    g["title"] = "Swipe Navigation Settings For Gestures"
    fixed, errs = validate_and_fix(rec, URIS)
    g2 = fixed["response"]["contexts"][0]
    assert g2["goal"] == "Follow these steps to perform this Swipe Navigation Troubleshooting"
    assert g2["title"] == "Swipe navigation settings"
    assert errs == []


def test_description_forced_to_5_7_words():
    for raw in ["Lets you pick", "It will let you choose the navigation type and the gesture hint options",
                "will open settings", ""]:
        d = fit_description(raw)
        assert d.startswith("It will") and 5 <= len(d.split()) <= 7, d


def test_urls_scrubbed_everywhere():
    rec = _gold("sample_01")
    a = rec["response"]["contexts"][0]["actions"][0]
    a["stepGroups"][0]["steps"].append("Visit https://www.samsung.com/support for help.")
    a["description"] = "It will see [the guide](http://x.y) now"
    rec["query_variations"][0] = "see www.example.com for swipe"
    assert any(v.check == "zero_urls" for v in check_record(rec, URIS))
    fixed, errs = validate_and_fix(rec, URIS)
    assert errs == [], errs
    assert "samsung.com" not in json.dumps(fixed)
    for bad in ["Visit samsung.com/support", "go to http://a.b/c", "[x](https://y.z)", "www.foo.org"]:
        assert has_url(bad) and not has_url(scrub_urls(bad))


def test_manual_deeplink_stripped_and_critical_last():
    rec = _gold("sample_03")
    acts = rec["response"]["contexts"][0]["actions"]
    acts[1]["stepGroups"][0]["actionableDeeplink"] = acts[0]["stepGroups"][0]["actionableDeeplink"]
    acts.insert(0, acts.pop())  # critical first
    errs = {v.check for v in check_record(rec, URIS)}
    assert {"manual_no_deeplink", "category_order"} <= errs
    fixed, errs2 = validate_and_fix(rec, URIS)
    assert errs2 == []
    cats = [a["category"] for a in fixed["response"]["contexts"][0]["actions"]]
    assert cats == sorted(cats, key={"auto": 0, "manual": 1, "critical": 2}.get)


def test_unknown_deeplink_removed_never_invented():
    rec = _gold("sample_01")
    sg = rec["response"]["contexts"][0]["actions"][0]["stepGroups"][0]
    sg["actionableDeeplink"]["deeplink"] = "voiceassist://masked/act/zzzzzzzzzz"
    assert any(v.check == "catalog_deeplinks" for v in check_record(rec, URIS))
    fixed, errs = validate_and_fix(rec, URIS)
    assert fixed["response"]["contexts"][0]["actions"][0]["stepGroups"][0]["actionableDeeplink"] is None
    assert errs == []


def test_variations_count_enforced():
    rec = _gold("sample_01")
    rec["query_variations"] = rec["query_variations"][:3] + rec["query_variations"][:3]
    assert any(v.check == "query_variations" for v in check_record(rec, URIS))
    fixed, errs = validate_and_fix(rec, URIS, pad=lambda q: [f"{q} variant {i}" for i in range(10)])
    assert 8 <= len(fixed["query_variations"]) <= 10 and errs == []


def test_title_case_helper():
    assert title_case("turn on put unused apps to sleep") == "Turn On Put Unused Apps to Sleep"
    assert is_title_case("Configure Navigation Bar Settings")
    assert not is_title_case("Configure navigation bar settings")


def test_loader_accepts_raw_lists(tmp_path):
    """Official-kit style: bare lists and alternative field names (qna_description)."""
    (tmp_path / "deeplinks.json").write_text(json.dumps([
        {"deeplink": "voiceassist://masked/act/abc", "description": "Open display", "message": "m",
         "qna_description": "display menu"}]))
    (tmp_path / "siis_responses.json").write_text(json.dumps({"k1": "Open Settings > Display."}))
    (tmp_path / "queries.json").write_text(json.dumps(["screen issue"]))
    kit = load_kit(str(tmp_path))
    assert kit.catalog[0].cna == "display menu" and kit.catalog[0].path == ()
    assert kit.siis[0].id == "k1" and kit.queries[0].type == "no_match"


# ------------------------------------------------------------------ official kit
def _official_uris():
    return {e.uri for e in OFFICIAL.catalog} | {e.val_uri for e in OFFICIAL.catalog if e.val_uri}


def test_official_sample_output_passes_every_check():
    """data/sample_output.json (official) is the reference shape: it must pass the contract and all checks."""
    ref = copy.deepcopy(OFFICIAL.reference_outputs[0])
    ContextDeeplinkResponse.model_validate(ref["response"])
    assert "query_variations" not in ref
    assert check_record(ref, _official_uris()) == []


def test_official_sample_deeplinks_are_in_the_catalog():
    """The only real deeplink pair the official kit reveals is part of the catalog, unchanged."""
    sg = OFFICIAL.reference_outputs[0]["response"]["contexts"][0]["actions"][0]["stepGroups"][0]
    entry = next(e for e in OFFICIAL.catalog if e.uri == sg["actionableDeeplink"]["deeplink"])
    assert entry.description == sg["actionableDeeplink"]["description"]
    assert entry.message == sg["actionableDeeplink"]["message"]
    assert entry.original_type == sg["actionableDeeplink"]["originalType"]
    assert entry.val_uri == sg["validationDeeplink"]["deeplink"]
    assert entry.validation["key"] == sg["validationDeeplink"]["key"]


def test_official_kit_loaded():
    assert OFFICIAL.official and len(OFFICIAL.cases) == 20
    assert all(c.siis_response["title"] and c.siis_response["content"] for c in OFFICIAL.cases)
    assert len(OFFICIAL.siis) == 11  # 20 rows share 11 distinct articles
    assert all(e.uri.startswith("voiceassist://masked/act/") for e in OFFICIAL.catalog)
    assert {c.siis_id for c in OFFICIAL.cases} <= {d.id for d in OFFICIAL.siis}


def test_official_payload_parsing():
    from app.siis_text import article_from_payload
    c = OFFICIAL.cases[0]
    a = article_from_payload(c.siis_response)
    assert not a.text.startswith("Smartphone,")  # product-category prefix removed
    assert a.procedural and [s.heading for s in a.sections][1] == "Check Email Access on a PC"
    garbled = article_from_payload(next(x.siis_response for x in OFFICIAL.cases if x.id == "row_3"))
    assert "enteryourcurrentpin" not in garbled.text and "Improve accuracy" in garbled.text


def test_description_length_follows_official_sample():
    assert check_record({"query": "q", "response": {"contexts": [{
        "goal": "Follow these steps to perform this Screen Damage Troubleshooting", "title": "Screen display damage",
        "score": 0.9, "actions": [{"actionName": "Schedule Screen Repair Service",
                                   "description": "It will help you locate the nearest TechCorp service center and schedule",
                                   "stepGroups": [{"steps": ["Contact Customer Support."]}], "category": "manual"}]}]}},
        set()) == []
