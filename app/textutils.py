"""Deterministic text helpers: URL scrubbing, casing rules, word counts, sentence splitting."""
from __future__ import annotations

import re

# --- URL scrubbing ---------------------------------------------------------------------
MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
URL_RE = re.compile(
    r"(?:https?://|ftp://|www\.)\S+"
    r"|\b[\w-]+(?:\.[\w-]+)*\.(?:com|net|org|in|co|io|ly|info|biz|me|app|dev|uk|us)(?:/\S*)?\b",
    re.IGNORECASE,
)
DEEPLINK_RE = re.compile(r"bixby://\S+", re.IGNORECASE)


def has_url(text: str) -> bool:
    return bool(MD_LINK_RE.search(text) or URL_RE.search(text))


def scrub_urls(text: str) -> str:
    text = MD_LINK_RE.sub(r"\1", text)
    text = URL_RE.sub("", text)
    text = re.sub(r"\b(?:visit|see|go to|check)\s+(?:the\s+)?(?:website|link|page)?\s*([.,;:]|$)", r"\1", text,
                  flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip()


def scrub_step_text(text: str) -> str:
    """Steps may never carry links of any kind (including deeplinks)."""
    return scrub_urls(DEEPLINK_RE.sub("", text))


# --- casing ------------------------------------------------------------------------------
SMALL_WORDS = {"a", "an", "the", "and", "or", "but", "nor", "of", "to", "in", "on", "for", "with", "at", "by",
               "from", "as", "into", "via", "vs", "per", "up", "off", "out"}
# Terms that keep their own casing inside sentence-case titles.
PROPER_TERMS = {"wi-fi", "bluetooth", "gps", "ram", "nfc", "aod", "apn", "sim", "usb", "5g", "lte", "hdr", "dns",
                "galaxy", "samsung", "one", "ui", "dolby", "atmos", "google", "play", "store", "dex", "s", "pen",
                "safe", "mode", "always", "on", "display", "hotspot", "sd", "hz"}


def words(text: str) -> list[str]:
    return [w for w in re.split(r"\s+", text.strip()) if w]


PHRASAL_VERBS = {"turn", "switch", "back", "set", "sign", "log", "free", "clean", "power", "start"}
PARTICLES = {"on", "off", "up", "out"}


def title_case(text: str) -> str:
    ws = words(text)
    out = []
    for i, w in enumerate(ws):
        lw = w.lower()
        if lw in PARTICLES and i > 0 and ws[i - 1].lower() in PHRASAL_VERBS:
            out.append(w[:1].upper() + w[1:])  # "Turn On", not "Turn on"
        elif 0 < i < len(ws) - 1 and lw in SMALL_WORDS:
            out.append(lw)
        elif w.isupper() and len(w) > 1:
            out.append(w)  # acronyms
        elif "-" in w:
            out.append("-".join(p[:1].upper() + p[1:] for p in w.split("-")))
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)


def is_title_case(text: str) -> bool:
    """Every word capitalised, except minor words (a, the, to, on…) which may stay lowercase."""
    ws = words(text)
    if not ws:
        return False
    for i, w in enumerate(ws):
        head = re.sub(r"^[^\w]+", "", w)[:1]
        if not head or head.isupper() or head.isdigit():
            continue
        if i > 0 and w.lower() in SMALL_WORDS:
            continue
        return False
    return True


def sentence_case(text: str) -> str:
    ws = words(text)
    out = []
    for i, w in enumerate(ws):
        if i == 0:
            out.append(w[:1].upper() + w[1:])
        elif w.isupper() or w.lower() in PROPER_TERMS or any(c.isupper() for c in w[1:]):
            out.append(w)
        else:
            out.append(w.lower())
    return " ".join(out)


def is_sentence_case(text: str) -> bool:
    ws = words(text)
    if not ws or not ws[0][:1].isupper():
        return False
    for w in ws[1:]:
        if w[:1].isupper() and not (w.isupper() or w.lower() in PROPER_TERMS or any(c.isupper() for c in w[1:])):
            return False
    return True


# --- description fitting ---------------------------------------------------------------
DESC_PAD = ["for", "you"]
_TRAILING_STOP = SMALL_WORDS | {"your", "this", "that", "its", "it", "is", "are"}


def fit_description(text: str) -> str:
    """Force 'It will …' with 5-7 words."""
    t = scrub_urls(text).strip().rstrip(".")
    t = re.sub(r"^(?:this\s+)?(?:it\s+)?will\s+", "", t, flags=re.IGNORECASE)
    rest = words(t)
    if rest:
        rest[0] = rest[0].lower() if not rest[0].isupper() else rest[0]
    rest = rest[:5]
    while len(rest) > 3 and rest[-1].lower() in _TRAILING_STOP:
        rest = rest[:-1]
    out = ["It", "will"] + rest
    pad = iter(["help", "fix", "the", "issue"] if not rest else DESC_PAD)
    while len(out) < 5:
        out.append(next(pad))
    return " ".join(out)


def is_valid_description(text: str) -> bool:
    return text.startswith("It will") and 5 <= len(words(text)) <= 7 and not has_url(text)


# --- sentences ---------------------------------------------------------------------------
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    return [s.strip() for s in SENT_RE.split(text) if s.strip()]


def as_step(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip().strip(",;:").strip()
    if not t:
        return ""
    t = t[0].upper() + t[1:]
    return t if t.endswith((".", "!", "?")) else t + "."
