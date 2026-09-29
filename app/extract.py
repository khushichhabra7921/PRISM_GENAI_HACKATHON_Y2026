"""④ Grounded extraction: SIIS text -> actions -> step groups -> steps.

Modes (SGTE_EXTRACT_MODE):
  pointer     (default) the SLM sees numbered instruction units parsed from the SIIS text and
              returns only groupings + names + descriptions (unit ids). Step text is copied
              from the source deterministically, so it cannot be hallucinated. ~100 output tokens.
  generative  the SLM writes the steps itself; every step is then fuzzy-matched against the
              source and ungrounded steps are dropped.
  rules       no LLM: deterministic grouping and naming (also the fallback when no LLM is up).
Neither LLM schema has a deeplink field, so the model cannot emit a URI.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz import fuzz

from app.config import SETTINGS
from app.llm_client import LLMClient, Usage
from app.order import order_actions
from app.steps import (DraftAction, Unit, categorise, find_paths, group_units, name_action, parse_units)
from app.textutils import as_step, scrub_step_text, sentence_case, split_sentences, title_case

NAV_RE = re.compile(r"^(?:Navigate to and open (.+)|Tap on (.+)|Open the (.+) app|Swipe down .* open the (.+))\.$")


@dataclass
class Extraction:
    topic: str
    title: str
    actions: list[DraftAction]
    mode: str
    grounding: list[float] = field(default_factory=list)
    dropped_steps: int = 0
    viable: bool = True
    note: str = ""


# ------------------------------------------------------------------ grounding
def ground_score(step: str, source: str, sentences: list[str]) -> float:
    """0-100: how well a step is supported by the source text."""
    m = NAV_RE.match(step)
    if m:
        term = next(g for g in m.groups() if g)
        return 100.0 if term.lower() in source.lower() else 0.0
    s = step.rstrip(".").lower()
    if s in source.lower():
        return 100.0
    return max((fuzz.token_set_ratio(s, x.lower()) for x in sentences), default=0.0)


_TRAILING_FILLER = {"in", "on", "of", "the", "a", "an", "to", "for", "with", "and", "or", "after", "while", "not",
                    "at", "by", "from"}


def _topic_title(doc_title: str, actions: list[DraftAction]) -> tuple[str, str]:
    base = doc_title or (actions[0].path[-1] if actions and actions[0].path else "Device")
    ws = base.split()[:3]
    while len(ws) > 1 and ws[-1].lower() in _TRAILING_FILLER:  # "Earbuds sound in" -> "Earbuds sound"
        ws = ws[:-1]
    topic = title_case(" ".join(ws))
    suffix = "settings" if actions and all(a.category == "auto" for a in actions) else "fix"
    title = sentence_case(" ".join(ws if len(ws) >= 3 else ws + [suffix]))
    return topic, title


# ------------------------------------------------------------------ pointer mode
POINTER_SCHEMA = {
    "type": "object",
    "properties": {
        "viable": {"type": "boolean"},
        "topic": {"type": "string"},
        "title": {"type": "string"},
        "actions": {"type": "array", "items": {
            "type": "object",
            "properties": {"ids": {"type": "array", "items": {"type": "integer"}},
                           "name": {"type": "string"}, "description": {"type": "string"}},
            "required": ["ids", "name", "description"]}},
    },
    "required": ["viable", "topic", "title", "actions"],
}

POINTER_PROMPT = """User complaint: "{query}"

Numbered instructions from the trusted support article:
{units}

Build a troubleshooting plan using ONLY these instructions (refer to them by number).
- Put instructions that happen on the same screen into one action; one action = one screen or one physical task.
- Leave out instructions that do not help with this complaint. If none help, set "viable" to false.
- "name": Title Case, e.g. "Configure Navigation Bar Settings".
- "description": starts with "It will", 5 to 7 words, says the benefit.
- "topic": 1-3 words naming the problem, e.g. "Swipe Navigation". "title": 2-3 words, sentence case.
Return JSON: {{"viable": true, "topic": "...", "title": "...", "actions": [{{"ids": [1], "name": "...", "description": "..."}}]}}{feedback}"""


def _unit_line(n: int, u: Unit) -> str:
    screen = f"(screen: {' > '.join(u.path)}) " if u.path else ""
    return f"[{n}] {screen}{u.sentence}"


def _pointer(query: str, units: list[Unit], llm: LLMClient, usage: Usage, feedback: str) -> Optional[dict]:
    listing = "\n".join(_unit_line(i + 1, u) for i, u in enumerate(units))
    res = llm.complete_json(POINTER_PROMPT.format(query=query.replace('"', "'"), units=listing, feedback=feedback),
                            POINTER_SCHEMA, max_tokens=300)
    usage.add(res)
    return res.data if res.ok else None


def _from_pointer(data: dict, units: list[Unit]) -> list[DraftAction]:
    used: set[int] = set()
    actions: list[tuple[int, DraftAction]] = []
    for spec in data.get("actions", []) or []:
        ids = [i - 1 for i in spec.get("ids", []) if isinstance(i, int) and 0 < i <= len(units) and i - 1 not in used]
        if not ids:
            continue
        used.update(ids)
        groups = group_units([units[i] for i in sorted(ids)])  # rules split mixed screens/categories safely
        for gi, g in enumerate(groups):
            g.name, g.description = name_action(g)
            if gi == 0 and len(groups) == 1:
                g.name = str(spec.get("name") or g.name)
                g.description = str(spec.get("description") or g.description)
            actions.append((g.units[0].idx, g))
    return [a for _, a in sorted(actions, key=lambda t: t[0])]


# ------------------------------------------------------------------ generative mode
GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "viable": {"type": "boolean"}, "topic": {"type": "string"}, "title": {"type": "string"},
        "actions": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "description": {"type": "string"},
                           "steps": {"type": "array", "items": {"type": "string"}}},
            "required": ["name", "description", "steps"]}},
    },
    "required": ["viable", "topic", "title", "actions"],
}

GEN_PROMPT = """User complaint: "{query}"

Trusted support article:
{text}

Turn the article into a troubleshooting plan. Use ONLY information from the article; never add steps or links.
One action = one screen or one physical task; each step = one tap or one physical interaction, imperative.
"name" Title Case; "description" starts with "It will", 5-7 words. "topic" 1-3 words; "title" 2-3 words, sentence case.
If the article does not help with the complaint set "viable" to false.{feedback}"""


def _from_generative(data: dict, units: list[Unit], source: str, known: set[str]) -> tuple[list[DraftAction], int]:
    sentences = split_sentences(source)
    dropped = 0
    out = []
    for spec in data.get("actions", []) or []:
        steps, best_units = [], []
        for raw in spec.get("steps", []) or []:
            st = as_step(scrub_step_text(str(raw)))
            if not st or ground_score(st, source, sentences) < SETTINGS.grounding_min:
                dropped += 1
                continue
            steps.append(st)
            best = max(units, key=lambda u: fuzz.token_set_ratio(st.lower(), u.sentence.lower()), default=None)
            if best:
                best_units.append(best)
        if not steps:
            continue
        path = next((u.path for u in best_units if u.path), None)
        cat = "critical" if any(u.category == "critical" for u in best_units) else categorise(" ".join(steps), path)
        u = Unit(idx=min((b.idx for b in best_units), default=0), sentence=" ".join(b.sentence for b in best_units),
                 steps=steps, path=path, category=cat)
        a = DraftAction(cat, path, [u], str(spec.get("name", "")), str(spec.get("description", "")))
        out.append(a)
    return out, dropped


# ------------------------------------------------------------------ entry point
def extract(text: str, query: str, known: set[str], doc_title: str = "", llm: Optional[LLMClient] = None,
            usage: Optional[Usage] = None, mode: str = "rules", feedback: list[str] | None = None
            ) -> Optional[Extraction]:
    usage = usage or Usage()
    units = parse_units(text, known)
    if not units:
        return None  # nothing actionable in the source -> caller returns no_match
    fb = ("\nFix these problems from your previous answer: " + "; ".join(feedback)) if feedback else ""
    used_mode, actions, dropped, topic, title = "rules", [], 0, "", ""
    if mode in ("pointer", "generative") and llm is not None and llm.available():
        if mode == "pointer":
            data = _pointer(query, units, llm, usage, fb)
            if data is not None:
                if data.get("viable") is False:
                    return Extraction("", "", [], "pointer", viable=False, note="LLM judged article not viable")
                actions, used_mode = _from_pointer(data, units), "pointer"
        else:
            res = llm.complete_json(GEN_PROMPT.format(query=query, text=text, feedback=fb), GEN_SCHEMA, max_tokens=900)
            usage.add(res)
            if res.ok and res.data is not None:
                if res.data.get("viable") is False:
                    return Extraction("", "", [], "generative", viable=False, note="LLM judged article not viable")
                actions, dropped = _from_generative(res.data, units, text, known)
                used_mode = "generative"
        if used_mode != "rules":
            topic, title = str((data if mode == "pointer" else res.data).get("topic", "")), \
                str((data if mode == "pointer" else res.data).get("title", ""))
    if not actions:  # rules mode, LLM unreachable, or LLM returned nothing usable
        actions = group_units(units)
        for a in actions:
            a.name, a.description = name_action(a)
        used_mode = "rules" if used_mode == "rules" else f"{used_mode}+rules"
    actions = order_actions(actions)
    rt, rtitle = _topic_title(doc_title, actions)
    topic = topic if topic and len(topic.split()) <= 4 else rt
    title = title if title and 2 <= len(title.split()) <= 3 else rtitle
    sentences = split_sentences(text)
    grounding = [ground_score(s, text, sentences) for a in actions for s in a.steps]
    return Extraction(topic=topic, title=title, actions=actions, mode=used_mode, grounding=grounding,
                      dropped_steps=dropped)
