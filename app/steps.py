"""Deterministic SIIS sentence -> grounded step units, plus rule-based grouping/naming.

A *unit* is one instruction sentence from the source text, turned into one or more
imperative steps (one physical interaction each) without adding any words that are not
in the source (apart from fixed navigation phrasing such as "Tap on <menu>.").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from app.textutils import as_step, fit_description, split_sentences, title_case

ROOTS = ("Settings", "Camera", "Quick panel", "Game Launcher")
ROOT_RE = re.compile(r"(?<!> )\b(Settings|Camera|Quick panel|Game Launcher) > ")
STOP_RE = re.compile(r"( and | to | if | when | from |,| screen\b)")

IMPERATIVE = {
    "open", "go", "navigate", "tap", "select", "choose", "turn", "toggle", "switch", "set", "drag", "swipe", "press",
    "touch", "hold", "remove", "clean", "wipe", "restart", "reboot", "update", "install", "uninstall", "reinstall",
    "delete", "clear", "close", "force", "check", "make", "try", "use", "visit", "contact", "connect", "reconnect",
    "disconnect", "unplug", "plug", "charge", "keep", "let", "move", "back", "insert", "reinsert", "put", "place",
    "avoid", "enable", "disable", "allow", "add", "forget", "unpair", "pair", "reset", "perform", "run", "enter",
    "test", "lower", "reduce", "increase", "adjust", "stop", "wait", "dry", "follow", "free", "raise", "store",
}
LEAD_ADVERBS = {"optionally", "then", "next", "finally", "now", "also", "first"}
SPLIT_VERBS = sorted(IMPERATIVE - {"hold", "back", "store", "dry", "free"})
_V = "|".join(SPLIT_VERBS)
SPLIT_RE = re.compile(
    rf",?\s+and then\s+|,\s*then\s+|;\s+|,\s+and\s+(?=(?:{_V})\b)|\s+and\s+(?=(?:{_V})\b)"
    rf"|,\s+(?=(?:touch|tap|select|open|turn|choose|enter|press)\b)",
)  # case-sensitive on purpose: a capitalised "Swipe gestures" is a menu item, not a verb
COND_RE = re.compile(r"^(?:If|When|Once|Before|To|While)\b([^,]*),\s*", re.IGNORECASE)
BECAUSE_RE = re.compile(r",?\s+because\b.*$", re.IGNORECASE)
NAV_PRE_RE = re.compile(r"^(?:open|go to|navigate to|launch)(?:\s+the)?\s*$", re.IGNORECASE)
PREP_TAIL_RE = re.compile(r"\s+(?:from|in|under|via|on|using)(?:\s+the)?\s*$", re.IGNORECASE)

CRITICAL_RE = re.compile(r"\b(restart|reboot|factory|safe mode|software update|system software|firmware|"
                         r"download and install|reset)\b", re.IGNORECASE)
NOT_PHONE_RE = re.compile(r"\b(router|modem)\b", re.IGNORECASE)
PHYSICAL_RE = re.compile(r"\b(clean|wipe|remove|case|protector|charger|cable|port|sim card|router|service cent|"
                         r"carrier|cloth|brush|hands|cool|heat|window|outside|pairing mode|lens|earbud|unplug|"
                         r"swab|volume up key)\b", re.IGNORECASE)
CONTINUATION = {"select", "turn", "toggle", "tap", "use", "set", "choose", "drag", "optionally", "reconnect", "enter"}


@dataclass
class Unit:
    idx: int
    sentence: str
    steps: list[str]
    path: Optional[tuple[str, ...]]
    category: str
    condition: str = ""


@dataclass
class DraftAction:
    category: str
    path: Optional[tuple[str, ...]]
    units: list[Unit] = field(default_factory=list)
    name: str = ""
    description: str = ""

    @property
    def steps(self) -> list[str]:
        return [s for u in self.units for s in u.steps]

    @property
    def sources(self) -> list[str]:
        return [u.sentence for u in self.units]


# ------------------------------------------------------------------ menu paths
def find_paths(sentence: str, known: set[str]) -> list[tuple[int, int, tuple[str, ...]]]:
    """Spans of menu paths like 'Settings > Display > Navigation bar' (greedy on known paths)."""
    spans = []
    for m in ROOT_RE.finditer(sentence):
        start = m.start()
        rest = sentence[start:].rstrip(".")
        segs = rest.split(" > ")
        head, last = segs[:-1], segs[-1]
        ws = last.split(" ")
        chosen = None
        for n in range(len(ws), 0, -1):
            cand_last = " ".join(ws[:n]).rstrip(",")
            if " > ".join(head + [cand_last]) in known:
                chosen = cand_last
                break
        if chosen is None:
            chosen = STOP_RE.split(last, maxsplit=1)[0].rstrip(",")
        text = " > ".join(head + [chosen])
        spans.append((start, start + len(text), tuple(head + [chosen])))
    return spans


def nav_steps(path: Iterable[str]) -> list[str]:
    path = list(path)
    first = {"Settings": "Navigate to and open Settings.", "Camera": "Open the Camera app.",
             "Quick panel": "Swipe down from the top of the screen to open the Quick panel.",
             "Game Launcher": "Open the Game Launcher app."}.get(path[0], f"Open {path[0]}.")
    return [first] + [f"Tap on {p}." for p in path[1:]]


# ------------------------------------------------------------------ sentence -> unit
def _first_word(text: str) -> str:
    m = re.match(r"[A-Za-z]+", text.strip())
    return m.group(0).lower() if m else ""


def is_instruction(text: str) -> bool:
    w = _first_word(text)
    if w in LEAD_ADVERBS:
        rest = re.sub(r"^\w+,?\s+", "", text.strip())
        return _first_word(rest) in IMPERATIVE or w == "optionally"
    return w in IMPERATIVE


def _split_clauses(text: str) -> list[str]:
    parts = [p for p in SPLIT_RE.split(text) if p and p.strip()]
    merged: list[str] = []
    for p in parts:
        if merged and len(merged[-1].split()) == 1:  # "Press" + "hold the Side key" -> keep together
            merged[-1] = f"{merged[-1]} and {p}"
        else:
            merged.append(p)
    return merged


def _clause_steps(clause: str, paths: dict[str, tuple[str, ...]]) -> tuple[list[str], Optional[tuple[str, ...]]]:
    m = re.search(r"§(\d+)§", clause)
    if not m:
        return ([as_step(BECAUSE_RE.sub("", clause))], None)
    path = paths[m.group(1)]
    pre, post = clause[: m.start()].strip(), clause[m.end():].strip()
    post = re.sub(r"^(?:screen|menu|page)\b\s*", "", post)
    steps = nav_steps(path)
    post = BECAUSE_RE.sub("", post)
    if NAV_PRE_RE.match(pre) or not pre:
        if post:
            post = re.sub(r"^to\s+", "", post)
            steps.append(as_step(post))
    else:
        action = PREP_TAIL_RE.sub("", pre)
        steps.append(as_step(f"{action} {post}".strip()))
    return [s for s in steps if s], path


def categorise(text: str, path: Optional[tuple[str, ...]]) -> str:
    if CRITICAL_RE.search(text) and not NOT_PHONE_RE.search(text):
        return "critical"
    return "auto" if path else "manual"


def sentence_to_unit(idx: int, sentence: str, known: set[str]) -> Optional[Unit]:
    cond = ""
    body = sentence.strip()
    m = COND_RE.match(body)
    if m and is_instruction(body[m.end():]):
        cond, body = m.group(1).strip(), body[m.end():]
    if not is_instruction(body):
        return None
    spans = find_paths(body, known)
    ph: dict[str, tuple[str, ...]] = {}
    for i, (s, e, p) in enumerate(reversed(spans)):
        key = str(len(spans) - 1 - i)
        ph[key] = p
        body = body[:s] + f"§{key}§" + body[e:]
    body = body.strip().rstrip(".")
    steps: list[str] = []
    path = None
    for clause in _split_clauses(body):
        cs, cp = _clause_steps(clause.strip(), ph)
        steps.extend(cs)
        path = cp or path
    steps = [s for s in steps if len(s) > 2]
    if not steps:
        return None
    flat = " ".join(steps)
    return Unit(idx=idx, sentence=sentence, steps=steps, path=path, category=categorise(flat, path), condition=cond)


def parse_units(text: str, known: set[str]) -> list[Unit]:
    units = []
    for i, s in enumerate(split_sentences(text)):
        u = sentence_to_unit(i, s, known)
        if u:
            units.append(u)
    return units


# ------------------------------------------------------------------ grouping (rules extractor)
def _is_continuation(u: Unit) -> bool:
    return (u.path is None and u.category == "manual" and _first_word(u.steps[0]) in CONTINUATION
            and not PHYSICAL_RE.search(u.sentence))


def group_units(units: list[Unit]) -> list[DraftAction]:
    actions: list[DraftAction] = []
    for u in units:
        cur = actions[-1] if actions else None
        if cur and "safe mode" in u.condition.lower() and "safe mode" in " ".join(cur.steps).lower() \
                and not PHYSICAL_RE.search(u.sentence):
            cur.units.append(u)
        elif u.path:
            if cur and cur.path == u.path and cur.category == u.category:
                cur.units.append(u)
            else:
                actions.append(DraftAction(u.category, u.path, [u]))
        elif cur and cur.category == "auto" and _is_continuation(u):
            cur.units.append(u)
        else:
            actions.append(DraftAction(u.category, None, [u]))
    return actions


# ------------------------------------------------------------------ naming
CRITICAL_NAMES = [
    (r"force restart|volume down key", "Force Restart Your Phone", "It will force the phone to restart"),
    (r"safe mode", "Start Safe Mode", "It will check for problem apps"),
    (r"factory", "Perform a Factory Data Reset", "It will restore factory default settings"),
    (r"reset network|network settings", "Reset Network Settings", "It will restore default network settings"),
    (r"reset all settings", "Reset All Settings", "It will restore all default settings"),
    (r"camera settings|reset settings", "Reset Camera Settings", "It will restore default camera settings"),
    (r"keyboard", "Reset Keyboard Settings", "It will restore default keyboard settings"),
    (r"access point|reset to default", "Reset Access Point Names", "It will restore default mobile data settings"),
    (r"^(?:restart|reboot)|restart the phone|reboot", "Restart Your Phone", "It will refresh the phone system"),
    (r"software update|system software|download and install|update", "Install Software Update",
     "It will install the latest software"),
    (r"restart", "Restart Your Phone", "It will refresh the phone system"),
]
MANUAL_NAMES = [
    (r"^remove|screen protector|case|cover", "Remove the Case", "It will rule out accessory interference"),
    (r"lens", "Clean the Camera Lens", "It will remove smudges from lens"),
    (r"charging port", "Clean the Charging Port", "It will remove dirt from port"),
    (r"speaker|earpiece|grille|mesh", "Clean the Speaker", "It will clear blocked sound openings"),
    (r"clean the screen|microfiber cloth|soft, dry cloth", "Clean the Screen", "It will remove dirt from screen"),
    (r"service cent", "Visit a Service Centre", "It will get the hardware checked"),
    (r"carrier", "Contact Your Carrier", "It will confirm your network service"),
    (r"charger|cable", "Try a Different Charger", "It will rule out faulty charging accessories"),
    (r"sim card", "Reinsert the SIM Card", "It will reseat the SIM card"),
    (r"router", "Restart Your Router", "It will refresh your router connection"),
    (r"cool|heat", "Let the Phone Cool Down", "It will bring the temperature down"),
    (r"update", "Update Your Apps", "It will install the latest app fixes"),
    (r"uninstall", "Uninstall Recent Apps", "It will remove apps causing problems"),
    (r"pairing mode", "Enable Pairing Mode", "It will make the device discoverable"),
    (r"volume up", "Raise the Call Volume", "It will make calls louder"),
    (r"focus", "Tap to Focus", "It will sharpen focus on subject"),
]
_STOP = {"the", "a", "an", "your", "any", "all", "on", "with", "to", "for", "of", "and", "from", "that", "it", "if"}


def _generic_name(step: str) -> str:
    ws = re.sub(r"[.,]", "", step).split()
    out = [ws[0]] if ws else ["Fix", "Issue"]
    for w in ws[1:]:
        if len(out) >= 2 and w.lower() in _STOP:
            break
        out.append(w)
        if len(out) >= 4:
            break
    return title_case(" ".join(out))


def name_action(a: DraftAction) -> tuple[str, str]:
    text = " ".join(a.steps).lower()
    if a.category == "critical":
        for pat, name, desc in CRITICAL_NAMES:
            if re.search(pat, text):
                return name, desc
    if a.category == "manual":
        for pat, name, desc in MANUAL_NAMES:
            if re.search(pat, text):
                return name, desc
        return _generic_name(a.steps[0]), fit_description("It will help resolve the problem")
    leaf = a.path[-1] if a.path else "Settings"
    lt = leaf.lower() if not any(c.isupper() for c in leaf[1:]) else leaf
    body = " ".join(a.steps).lower()
    onoff = re.search(r"\bturn (?:it |them )?(on|off)\b", body)
    if "optimize now" in body:
        return "Optimize the Device", "It will close apps and free memory"
    obj = re.search(r"\bTurn (on|off) ([A-Z][^.,]*)", " ".join(a.steps))
    if obj:  # "Turn on Put unused apps to sleep" -> name the toggle, not the parent screen
        target = " ".join(re.split(r" and | if | when ", obj.group(2))[0].split()[:5])
        return (title_case(f"Turn {obj.group(1)} {target}"),
                fit_description(f"It will turn {obj.group(1)} {target.lower()}"))
    if onoff:
        return title_case(f"Turn {onoff.group(1)} {leaf}"), fit_description(f"It will turn {onoff.group(1)} {lt}")
    if re.search(r"\bcheck\b", body):
        return title_case(f"Check {leaf}"), fit_description(f"It will show {lt} details")
    noun = leaf if leaf.lower().endswith("settings") else f"{leaf} Settings"
    return title_case(f"Configure {noun}"), fit_description(f"It will let you adjust {lt}")


def rules_extract(text: str, known: set[str]) -> list[DraftAction]:
    actions = group_units(parse_units(text, known))
    for a in actions:
        a.name, a.description = name_action(a)
    return actions
