"""① Enrichment: canonical query, topic, intent, symptom list, 8-10 paraphrases.

Never produces troubleshooting steps. LLM when reachable; deterministic rules otherwise.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from rapidfuzz import fuzz, process

from app.llm_client import LLMClient, Usage
from app.textutils import scrub_urls, words

TOPICS = ["Battery", "Display", "Camera", "Performance", "Connectivity", "Sound", "Notifications", "Navigation",
          "Software", "Other"]
TOPIC_KEYWORDS = {
    "Battery": r"batter|charg|drain|power|percent|dies",
    "Display": r"screen|display|bright|dim|flicker|colou?r|touch|rotat|text|font|dark mode|refresh|always on",
    "Camera": r"camera|photo|picture|pic|selfie|video|flash|lens|shutter",
    "Performance": r"slow|lag|freez|hang|crash|storage|memory|ram|restart|reboot|overheat|hot|app",
    "Connectivity": r"wi-?fi|bluetooth|data|signal|network|hotspot|gps|location|internet|pair",
    "Sound": r"sound|audio|volume|ringtone|speaker|vibrat|hear|earbud",
    "Notifications": r"notif|do not disturb|alert|silenc",
    "Navigation": r"swipe|gesture|navigation|home button|back button",
    "Software": r"update|reset|time|date|permission",
}
SLANG = {"u": "you", "ur": "your", "r": "are", "pls": "please", "plz": "please", "fone": "phone", "pic": "picture",
         "pics": "pictures", "cam": "camera", "batt": "battery", "wifi": "Wi-Fi", "wi fi": "Wi-Fi", "bt": "Bluetooth",
         "wont": "won't", "cant": "can't", "dont": "don't", "doesnt": "doesn't", "isnt": "isn't", "im": "I'm",
         "aftr": "after", "coz": "because", "cuz": "because", "sooo": "so", "rly": "really", "super": "very",
         "killing": "draining", "dies": "drains"}
SPLIT_RE = re.compile(r"\s+(?:and also|and|also|plus|as well as|&)\s+|;\s*", re.IGNORECASE)
CONFIG_RE = re.compile(r"^(how (do|can|to)|how do i|where (is|do|can)|can i|i want to|set up|change)\b", re.I)
PROBLEM_RE = re.compile(r"not|n't|wont|cant|stopped|broken|issue|problem|keeps|fail|error|slow|drain|fast", re.I)
INSTRUCTION_RE = re.compile(r"^(open|go to|tap|navigate|restart|reset|turn (on|off)|select)\b|settings\s*>", re.I)


@dataclass
class Enrichment:
    query: str
    canonical_query: str
    topic: str
    intent: str  # troubleshooting | configuration
    symptoms: list[str]
    query_variations: list[str]
    source: str  # llm | rules
    notes: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ rules
@lru_cache(maxsize=1)
def _vocab() -> tuple[str, ...]:
    from app.data_loader import load_kit
    kit = load_kit()
    text = " ".join([d.text + " " + d.title for d in kit.siis] + [e.text for e in kit.catalog])
    counts: dict[str, int] = {}
    for w in re.findall(r"[a-z][a-z\-']{3,}", text.lower()):
        counts[w] = counts.get(w, 0) + 1
    return tuple(w for w, c in counts.items() if c >= 2)


def _known(w: str, vs: set[str]) -> bool:
    """Known word, or an inflection of one (moves, downloading, randomly…): never 'correct' those."""
    stems = {w}
    for suf in ("s", "es", "ed", "d", "ing", "ly", "er"):
        if w.endswith(suf):
            stems |= {w[: -len(suf)], w[: -len(suf)] + "e"}
    return any(s in vs for s in stems if len(s) >= 3)


ENUM_RE = re.compile(r'(?:^|(?<=[.!?"\s]))\s*\d{1,2}[.)]\s+(?=["\'A-Za-z])')


def clean_query(q: str) -> str:
    """Complaints may arrive as a numbered list of quoted parts:
    '1. "My screen is cracked." 2. "The touch doesn\'t work."' -> 'My screen is cracked. The touch doesn\'t work.'"""
    q = ENUM_RE.sub(" ", q or "")
    q = re.sub(r'["“”]', "", q)
    return re.sub(r"\s+", " ", q).strip()


def normalise(q: str) -> str:
    out = []
    vocab = _vocab()
    vs = set(vocab)
    for tok in re.findall(r"[A-Za-z0-9\-']+|[^\sA-Za-z0-9]", q):
        low = tok.lower()
        if low in SLANG:
            out.append(SLANG[low])
        elif low.isalpha() and len(low) >= 5 and not _known(low, vs):
            m = process.extractOne(low, vocab, scorer=fuzz.ratio, score_cutoff=82)
            out.append(m[0] if m else tok)
        else:
            out.append(tok)
    s = " ".join(out)
    s = re.sub(r"\s+([.,!?%])", r"\1", s)
    s = re.sub(r"([!?.])\1+", r"\1", s)
    return s.strip()


def guess_topic(q: str) -> str:
    scores = {t: len(re.findall(p, q, re.I)) for t, p in TOPIC_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] else "Other"


def split_symptoms(q: str) -> list[str]:
    parts = [p.strip(" ,.") for p in SPLIT_RE.split(q) if p and p.strip(" ,.")]
    parts = [p for p in parts if len([w for w in words(p) if len(w) > 2]) >= 2]
    return parts if len(parts) > 1 else [q]


def _typo(text: str) -> str:
    ws = words(text)
    if not ws:
        return text
    i = max(range(len(ws)), key=lambda k: len(ws[k]))
    w = ws[i]
    if len(w) > 4:
        ws[i] = w[:2] + w[3] + w[2] + w[4:]
    return " ".join(ws)


def _keywords(text: str) -> str:
    stop = {"my", "the", "a", "an", "is", "are", "it", "and", "i", "to", "of", "on", "so", "very", "really", "its",
            "this", "that", "me", "in", "for", "with", "when", "after", "keeps", "just", "even"}
    return " ".join(w for w in re.findall(r"[a-z0-9\-']+", text.lower()) if w not in stop)


def rule_variations(query: str, canonical: Optional[str] = None, topic: str = "") -> list[str]:
    c = (canonical or normalise(query)).rstrip(".!? ")
    lc = c[:1].lower() + c[1:]
    kw = _keywords(c)
    cands = [
        f"I would like help because {lc}.",
        f"hey, {lc}, any ideas?",
        kw,
        f"{c} and it is driving me crazy!",
        _typo(lc),
        f"How do I fix it when {lc}?",
        f"What should I do if {lc}?",
        f"Nexa phone problem: {kw}",
        f"{topic or guess_topic(c)} issue - {kw}",
        f"Need help, {lc} on my TechCorp phone",
        f"Why is this happening: {lc}?",
    ]
    seen, out = {query.lower().strip()}, []
    for v in cands:
        k = v.lower().strip()
        if v and k not in seen:
            seen.add(k)
            out.append(v)
    return out[:10]


def rules_enrich(query: str) -> Enrichment:
    canonical = normalise(scrub_urls(query))
    topic = guess_topic(canonical)
    intent = "configuration" if CONFIG_RE.search(canonical) and not PROBLEM_RE.search(canonical) else "troubleshooting"
    return Enrichment(query=query, canonical_query=canonical, topic=topic, intent=intent,
                      symptoms=split_symptoms(canonical), query_variations=rule_variations(query, canonical, topic),
                      source="rules")


# ------------------------------------------------------------------ LLM
SCHEMA = {
    "type": "object",
    "properties": {
        "canonical_query": {"type": "string"},
        "topic": {"type": "string", "enum": TOPICS},
        "intent": {"type": "string", "enum": ["troubleshooting", "configuration"]},
        "symptoms": {"type": "array", "items": {"type": "string"}},
        "query_variations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["canonical_query", "topic", "intent", "symptoms", "query_variations"],
}

PROMPT = """Rewrite this TechCorp Nexa phone or tablet complaint. Do NOT suggest any fixes, steps, settings paths or links.
Complaint: "{q}"
Return JSON:
- canonical_query: one short, clear sentence describing the problem (fix slang and typos).
- topic: one of {topics}.
- intent: "configuration" if the user only asks how to change a setting, else "troubleshooting".
- symptoms: the distinct problems in the complaint, one short phrase each (a single item if there is one problem).
- query_variations: 10 different ways a user could describe the same problem: formal, casual, keyword-only, frustrated, and one with a typo."""


def _clean_list(items, limit: int) -> list[str]:
    out = []
    for x in items or []:
        if not isinstance(x, str):
            continue
        x = scrub_urls(x).strip()
        if x and not INSTRUCTION_RE.search(x) and x.lower() not in {o.lower() for o in out}:
            out.append(x)
    return out[:limit]


def llm_enrich(query: str, llm: LLMClient, usage: Usage) -> Optional[Enrichment]:
    res = llm.complete_json(PROMPT.format(q=query.replace('"', "'"), topics=", ".join(TOPICS)), SCHEMA,
                            max_tokens=400)
    usage.add(res)
    d = res.data if res.ok else None
    if not d:
        return None
    canonical = scrub_urls(str(d.get("canonical_query", ""))).strip()
    if not canonical or INSTRUCTION_RE.search(canonical):
        canonical = normalise(query)
    rules = rules_enrich(query)
    variations = _clean_list(d.get("query_variations"), 10)
    if len(variations) < 8:  # top up deterministically; never below the contract
        variations = (variations + [v for v in rules.query_variations if v not in variations])[:10]
    symptoms = _clean_list(d.get("symptoms"), 4) or [canonical]
    topic = d.get("topic") if d.get("topic") in TOPICS else rules.topic
    intent = d.get("intent") if d.get("intent") in ("troubleshooting", "configuration") else rules.intent
    return Enrichment(query=query, canonical_query=canonical, topic=topic, intent=intent, symptoms=symptoms,
                      query_variations=variations, source="llm")


def enrich(query: str, llm: Optional[LLMClient], usage: Usage, mode: str = "auto") -> Enrichment:
    if mode != "rules" and llm is not None and llm.available():
        e = llm_enrich(query, llm, usage)
        if e:
            return e
    return rules_enrich(query)
