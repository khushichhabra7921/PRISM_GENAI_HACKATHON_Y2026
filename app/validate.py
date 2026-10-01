"""⑦ Validation: Pydantic contract + 11 named rule checks, deterministic auto-fixers.

Rules live here, in code, not in prompts. An invalid plan is never cached.
"""
from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from pydantic import ValidationError

from app.config import SETTINGS
from app.textutils import (fit_description, has_url, is_sentence_case, is_title_case, is_valid_description,
                           scrub_step_text, scrub_urls, sentence_case, title_case, words)
from data.schema import ContextDeeplinkResponse

CHECKS = ["schema", "goal_format", "title_format", "score_range", "action_name_case", "description_format",
          "manual_no_deeplink", "category_order", "catalog_deeplinks", "zero_urls", "query_variations"]
GOAL_PREFIX = "Follow these steps to perform this "
GOAL_RE = re.compile(r"^Follow these steps to perform this (.+) (Troubleshooting|Configuration)$")
CAT_ORDER = {"auto": 0, "manual": 1, "critical": 2}


@dataclass(frozen=True)
class Violation:
    check: str
    where: str
    detail: str

    def __str__(self) -> str:
        return f"{self.check} @ {self.where}: {self.detail}"


def _strings(obj) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k != "deeplink":  # the masked URI itself is checked against the catalog instead
                yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def check_record(record: dict, catalog_uris: set[str]) -> list[Violation]:
    """Run all 11 checks on {"query", "query_variations", "response": {"contexts": [...]}}.
    `catalog_uris` holds every actionable and validation deeplink of the catalog."""
    v: list[Violation] = []
    resp = record.get("response", {})
    try:
        ContextDeeplinkResponse.model_validate(resp)
    except ValidationError as e:
        return [Violation("schema", "response", str(e.errors()[:3]))]
    for gi, g in enumerate(resp.get("contexts", [])):
        at = f"contexts[{gi}]"
        m = GOAL_RE.match(g["goal"])
        if not m or not is_title_case(m.group(1)):
            v.append(Violation("goal_format", at, g["goal"]))
        if not (2 <= len(words(g["title"])) <= 3 and is_sentence_case(g["title"])):
            v.append(Violation("title_format", at, g["title"]))
        s = g["score"]
        if not isinstance(s, (int, float)) or math.isnan(s) or not 0.0 <= s <= 1.0:
            v.append(Violation("score_range", at, str(s)))
        if not g["actions"]:
            v.append(Violation("schema", at, "no actions"))
        last = -1
        for ai, a in enumerate(g["actions"]):
            aat = f"{at}.actions[{ai}]"
            if not is_title_case(a["actionName"]):
                v.append(Violation("action_name_case", aat, a["actionName"]))
            if not is_valid_description(a["description"], SETTINGS.desc_max_words):
                v.append(Violation("description_format", aat, a["description"]))
            cat = a.get("category") or "manual"
            if cat not in CAT_ORDER:
                v.append(Violation("schema", aat, f"category {cat}"))
                continue
            if CAT_ORDER[cat] < last:
                v.append(Violation("category_order", aat, f"{cat} after a later category"))
            last = max(last, CAT_ORDER[cat])
            if not a["stepGroups"] or any(not sg["steps"] for sg in a["stepGroups"]):
                v.append(Violation("schema", aat, "empty steps"))
            for sg in a["stepGroups"]:
                dl = sg.get("actionableDeeplink")
                if cat == "manual" and dl:
                    v.append(Violation("manual_no_deeplink", aat, dl["deeplink"]))
                for key in ("actionableDeeplink", "validationDeeplink"):
                    d = sg.get(key)
                    if d and d["deeplink"] not in catalog_uris and d["deeplink"] != SETTINGS.dummy_deeplink:
                        v.append(Violation("catalog_deeplinks", aat, d["deeplink"]))
    for s in _strings({"q": record.get("query_variations", []), "r": resp}):
        if has_url(s):
            v.append(Violation("zero_urls", "record", s[:80]))
    if "query_variations" not in record:  # the official contract (sample_output.json) has no variations field
        return v
    qv = record.get("query_variations", [])
    if not (8 <= len(qv) <= 10) or len({q.strip().lower() for q in qv}) != len(qv):
        v.append(Violation("query_variations", "record", f"{len(qv)} variations"))
    return v


# ------------------------------------------------------------------ fixers
def fix_goal(goal: str, fallback_topic: str, intent: str = "Troubleshooting") -> str:
    m = GOAL_RE.match(goal.strip())
    if m:
        topic, kind = m.group(1), m.group(2)
    else:
        topic = re.sub(r"^(?:follow these steps to perform this\s+)?", "", goal.strip(), flags=re.I)
        topic = re.sub(r"\s*(troubleshooting|configuration)\.?$", "", topic, flags=re.I)
        kind = intent
    topic = scrub_urls(re.sub(r"[^\w\- ]", " ", topic)).strip() or fallback_topic
    return f"{GOAL_PREFIX}{title_case(topic)} {kind}"


def fix_title(title: str, fallback: str) -> str:
    ws = words(scrub_urls(re.sub(r"[^\w\- ]", " ", title))) or words(fallback)
    ws = ws[:3]
    while len(ws) > 2 and ws[-1].lower() in {"and", "or", "the", "a", "of", "to", "for"}:
        ws = ws[:-1]
    if len(ws) < 2:
        ws.append("settings")
    return sentence_case(" ".join(ws))


def fix_variations(query: str, variations: list[str], pad: Optional[Callable[[str], list[str]]]) -> list[str]:
    seen, out = set(), []
    for q in variations + (pad(query) if pad else []):
        q = scrub_urls(q).strip()
        key = q.lower().rstrip("?.! ")
        if q and key not in seen and key != query.lower().rstrip("?.! "):
            seen.add(key)
            out.append(q)
        if len(out) == 10:
            break
    return out


def fix_record(record: dict, catalog_uris: set[str], pad: Optional[Callable[[str], list[str]]] = None,
               intent: str = "Troubleshooting") -> dict:
    """Apply every deterministic fix. Never invents content: it only trims, rewrites casing,
    reorders, scrubs or removes."""
    r = copy.deepcopy(record)
    resp = r.setdefault("response", {"contexts": []})
    goals = []
    for g in resp.get("contexts", []):
        g["title"] = fix_title(g.get("title", ""), g.get("goal", "Troubleshooting fix"))
        g["goal"] = fix_goal(g.get("goal", ""), g["title"], intent)
        s = g.get("score", 0.5)
        g["score"] = round(min(1.0, max(0.0, float(s) if isinstance(s, (int, float)) and not math.isnan(s) else 0.5)), 2)
        actions = []
        for a in g.get("actions", []):
            a["actionName"] = title_case(scrub_urls(a.get("actionName", "")).strip(" .")) or "Follow These Steps"
            a["description"] = fit_description(a.get("description", ""))
            cat = a.get("category") or "manual"
            a["category"] = cat if cat in CAT_ORDER else "manual"
            groups = []
            for sg in a.get("stepGroups", []):
                sg["steps"] = [t for t in (scrub_step_text(x) for x in sg.get("steps", [])) if t]
                dl = sg.get("actionableDeeplink")
                if dl and (a["category"] == "manual" or
                           (dl["deeplink"] not in catalog_uris and dl["deeplink"] != SETTINGS.dummy_deeplink)):
                    sg["actionableDeeplink"] = None
                vd = sg.get("validationDeeplink")
                if vd and (vd["deeplink"] not in catalog_uris or a["category"] == "manual"):
                    sg["validationDeeplink"] = None
                for key in ("actionableDeeplink",):
                    d = sg.get(key)
                    if d:
                        d["description"] = scrub_urls(d.get("description", ""))
                        d["message"] = scrub_urls(d.get("message") or "")
                if sg["steps"]:
                    groups.append(sg)
            a["stepGroups"] = groups
            if groups:
                actions.append(a)
        actions.sort(key=lambda x: CAT_ORDER[x["category"]])  # stable: keeps order within a category
        g["actions"] = actions
        if actions:
            goals.append(g)
    resp["contexts"] = goals
    r["query_variations"] = fix_variations(r.get("query", ""), r.get("query_variations", []), pad)
    return r


def validate_and_fix(record: dict, catalog_uris: set[str], pad=None, intent="Troubleshooting") -> tuple[dict, list[Violation]]:
    fixed = fix_record(record, catalog_uris, pad, intent)
    return fixed, check_record(fixed, catalog_uris)


def to_json(obj: dict) -> str:
    """Pure JSON for the wire: no fences, no preamble."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
