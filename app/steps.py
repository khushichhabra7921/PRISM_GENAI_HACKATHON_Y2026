"""Deterministic SIIS sentence -> grounded step units, plus rule-based grouping/naming.

A *unit* is one instruction sentence from the source text, turned into one or more
imperative steps (one physical interaction each) without adding any words that are not
in the source (apart from fixed navigation phrasing such as "Tap on <menu>.").

Two source styles are understood:
  * menu paths written as "Settings > Display > Navigation bar" (synthetic regression kit), and
  * the official TechCorp style: comma chains ("go to Settings, tap Display, and then tap Navigation bar"),
    one step per line ("Navigate to Settings." / "Tap Apps."), "search for and select X", markdown sections
    whose headings name the procedure, labels ("On devices with a Side button: ...") and polite or modal
    phrasing ("please remove it", "you may need to check ..."), which is trimmed, never paraphrased.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from rapidfuzz import fuzz

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
    # official TechCorp articles
    "examine", "inspect", "ensure", "verify", "confirm", "shine", "eject", "locate", "review", "schedule", "provide",
    "scan", "search", "rotate", "attempt", "power", "launch", "change", "create", "exit", "customize", "give",
    "return", "send", "reach", "access", "mimic", "boot", "bring", "share", "rearrange", "disable", "expand",
}
LEAD_ADVERBS = {"optionally", "then", "next", "finally", "now", "also", "first", "alternatively", "simply"}
SPLIT_VERBS = sorted(IMPERATIVE - {"hold", "back", "store", "dry", "free", "power", "access", "give", "return",
                                   "reach", "bring", "share", "expand"})
_V = "|".join(SPLIT_VERBS)
SPLIT_RE = re.compile(
    rf",?\s+and then\s+|,\s*then\s+|;\s+|,\s+and\s+(?=(?:{_V})\b)|(?<!\bfor)\s+and\s+(?=(?:{_V})\b)"
    rf"|,\s+(?=(?:touch|tap|select|open|turn|choose|enter|press)\b)",
)  # case-sensitive on purpose: a capitalised "Swipe gestures" is a menu item, not a verb
COND_RE = re.compile(r"^(?:If|When|Once|Before|To|While|After|Unless)\b([^,]*),\s*", re.IGNORECASE)
BECAUSE_RE = re.compile(r",?\s+because\b.*$", re.IGNORECASE)
NAV_PRE_RE = re.compile(r"^(?:open|go to|navigate to|launch)(?:\s+the)?\s*$", re.IGNORECASE)
PREP_TAIL_RE = re.compile(r"\s+(?:from|in|under|via|on|using)(?:\s+the)?\s*$", re.IGNORECASE)

# --- official-style phrasing ---------------------------------------------------------------
MARKER_RE = re.compile(r"^(?:first|then|next|now|afterwards?|finally|additionally|in (?:this|that|such) cases?)\s*,\s*",
                       re.IGNORECASE)
POLITE_RE = re.compile(r"^(?:please|simply|just)\s+", re.IGNORECASE)
MODAL_RE = re.compile(
    r"^(?:you\s+(?:can|could|may|might|should|must|will)\s+(?:also\s+|still\s+|simply\s+|even\s+|always\s+|then\s+)?"
    r"(?:need\s+to\s+|want\s+to\s+|have\s+to\s+)?(?:also\s+)?"
    r"|it\s+is\s+(?:also\s+)?(?:recommended|advisable|important|best|a\s+good\s+idea)\s+(?:that\s+you\s+|to\s+)"
    r"|it'?s\s+(?:also\s+)?(?:a\s+good\s+(?:practice|idea)|recommended|important|best)\s+to\s+)",
    re.IGNORECASE)
LABEL_RE = re.compile(r"^([A-Z][^:.?!]{2,60}):\s+(?=\S)")
LEAD_WORDS = {"on", "for", "from", "with", "using", "in", "after", "once", "even", "to", "when", "if", "while",
              "before", "under", "within", "at", "by", "without", "during", "instead", "otherwise", "here", "there",
              "again", "meanwhile", "sometimes", "depending", "next", "then", "now", "also", "first", "lastly"}
CLAUSE1_RE = re.compile(r"^([A-Z][^,.:;?!]{1,60}),\s+(?=[a-z§])")
CLAUSE2_RE = re.compile(r"^([A-Z][^,.:;?!]{1,60},\s*[^,.:;?!]{1,60}),\s+(?=[a-z§])")
ADVERB_RE = re.compile(r"^[A-Za-z]+ly\s+")

# official comma-chain navigation: "go to Settings, tap Display, and then tap Navigation bar"
NAV_ROOT_RE = re.compile(r"\b(?i:navigate to and open|navigate to|go to|open|from)\s+(?:the\s+)?Settings\b(?:\s+app\b)?")
NAV_TAP_RE = re.compile(r"\s*,?\s*(?:and\s+)?(?:then\s+)?(?i:(tap|select|touch))\s+(?:on\s+)?(?=[A-Z])")
META_RE = re.compile(r"\b(?:follow|try|complete|go through)\s+(?:these|the following|the steps below|all (?:the|of the) "
                     r"(?:above|previous)|the above)\b(?:\s+\w+)?\s*[.:]?$", re.IGNORECASE)
TERM_END_RE = re.compile(r",|;|\.(?=\s|$)|\s+\(|\s+(?:to|if|when|again|then|and then|until|for|from|so|or)\b|$")
ACTION_WORDS = {"clear", "delete", "reset", "restart", "start", "allow", "remove", "add", "disconnect", "uninstall",
                "force", "install", "download", "turn", "enable", "disable", "confirm", "ok", "done", "yes", "save",
                "apply", "cancel", "back", "update", "power", "swipe", "receive", "send"}
PLACEHOLDER_RE = re.compile(r"§(\w+)§")
NAV_STEP_RE = re.compile(r"^(?:Navigate to and open \S|Open the .+ app\.$|Swipe down from the top of the screen to open the Quick panel\.$)")

CRITICAL_RE = re.compile(r"\b(restart|reboot|factory|safe mode|software update|system software|firmware|"
                         r"download and install|reset)\b", re.IGNORECASE)
NOT_PHONE_RE = re.compile(r"\b(?:(?i:router|modem)|TV|PC)\b")
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
    section: int = 0
    block: int = 0
    heading: str = ""
    headed: bool = False  # parsed from a sectioned (official-style) article


@dataclass
class DraftAction:
    category: str
    path: Optional[tuple[str, ...]]
    units: list[Unit] = field(default_factory=list)
    name: str = ""
    description: str = ""
    relevance: Optional[float] = None

    @property
    def steps(self) -> list[str]:
        return [s for u in self.units for s in u.steps]

    @property
    def sources(self) -> list[str]:
        return [u.sentence for u in self.units]

    @property
    def heading(self) -> str:
        return self.units[0].heading if self.units else ""

    def step_groups(self) -> list[tuple[list[str], Optional[tuple[str, ...]]]]:
        """One step group per navigation from a root screen: a second 'Navigate to and open Settings.' inside
        the same action (e.g. clear cache, then clear data) starts a new group with its own deeplink."""
        groups: list[list] = []
        for u in self.units:
            starts_nav = bool(u.path) and bool(u.steps) and bool(NAV_STEP_RE.match(u.steps[0]))
            if not groups or (starts_nav and groups[-1][1] is not None):
                groups.append([[], None])
            g = groups[-1]
            for st in u.steps:  # a variant repeated for another device model ("To open one in pop-up view, ...")
                if len(st.split()) >= 6 and any(fuzz.ratio(st.lower(), p.lower()) >= 90 for p in g[0]):
                    continue
                g[0].append(st)
            g[1] = g[1] or u.path
        return [(steps, path) for steps, path in groups if steps]


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


def _menu_term(rest: str, path: list[str], known: set[str]) -> Optional[str]:
    """The menu item named at the start of `rest` ('Display, and then tap ...' -> 'Display'), or None when the
    tapped thing is a control/button ('Clear cache', 'the switch next to ...') rather than a screen."""
    m = TERM_END_RE.search(rest)
    cand = rest[: m.start()].strip() if m else rest.strip()
    if not cand or not cand[0].isupper():
        return None
    if " and " in cand and " > ".join(path + [cand]) not in known:
        cand = cand.split(" and ")[0].strip()
    if " > ".join(path + [cand]) in known:
        return cand
    if cand.split()[0].lower() in ACTION_WORDS or len(cand.split()) > 5:
        return None
    return cand


def rewrite_nav(sentence: str, known: set[str]) -> tuple[str, dict[str, tuple[str, ...]]]:
    """Replace official-style navigation chains with §nK§ placeholders. 'Settings > A > B' notation is left
    for find_paths. Returns the rewritten text and {key: path}."""
    paths: dict[str, tuple[str, ...]] = {}
    text, start = sentence, 0
    while True:
        m = NAV_ROOT_RE.search(text, start)
        if not m:
            break
        if text[m.end():].startswith(" > "):
            start = m.end()
            continue
        path, pos = ["Settings"], m.end()
        while True:
            t = NAV_TAP_RE.match(text, pos)
            term = _menu_term(text[t.end():], path, known) if t else None
            if term and t.group(1).lower() == "select" and " > ".join(path + [term]) not in known:
                term = None  # "Select Buttons to ..." picks an option; only a known screen is selected into
            if not term:
                break
            path.append(term)
            pos = t.end() + len(term)
        key = f"n{len(paths)}"
        paths[key] = tuple(path)
        text = text[: m.start()] + f"§{key}§" + text[pos:]
        start = m.start() + len(key) + 2
    return text, paths


def _ends_with_nav(sentence: str, known: set[str]) -> bool:
    text, paths = rewrite_nav(sentence, known)
    return bool(paths) and bool(re.search(r"§n\d+§[\s.]*$", text))


def merge_nav_lines(sentences: list[str], known: set[str]) -> list[tuple[str, list[str]]]:
    """'Navigate to Settings.' + 'Tap Apps.' (one step per line) -> one navigation sentence.
    Returns (text to parse, original sentences)."""
    out: list[tuple[str, list[str]]] = []
    for s in sentences:
        if out and re.match(r"^Tap\s+(?:on\s+)?[A-Z]", s) and _ends_with_nav(out[-1][0], known):
            prev, origin = out[-1]
            out[-1] = (prev.rstrip(" .") + ", " + s[0].lower() + s[1:], origin + [s])
        else:
            out.append((s, [s]))
    return out


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
    text = text.strip()
    if text.startswith("§"):
        return True
    w = _first_word(text)
    if w in LEAD_ADVERBS:
        rest = re.sub(r"^\w+,?\s+", "", text)
        return _first_word(rest) in IMPERATIVE or rest.startswith("§") or w == "optionally"
    if w.endswith("ly") and w not in IMPERATIVE:  # "Carefully inspect ..."
        return _first_word(ADVERB_RE.sub("", text)) in IMPERATIVE
    if text.lower().startswith(("let's", "let us")):
        return False
    return w in IMPERATIVE


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def strip_preamble(text: str) -> str:
    """Drop sequencing markers, 'please', and modal preambles ('you may need to check' -> 'check').
    Only words are removed, and only when an imperative verb follows."""
    t = text.strip()
    for _ in range(4):
        before = t
        t = MARKER_RE.sub("", t)
        m = POLITE_RE.match(t)
        if m and is_instruction(t[m.end():]):
            t = t[m.end():]
        m = MODAL_RE.match(t)
        if m and m.end() < len(t) and is_instruction(t[m.end():]):
            t = t[m.end():]
        t = _cap(t)
        if t == before:
            break
    return t


def _split_clauses(text: str) -> list[str]:
    parts = [p for p in SPLIT_RE.split(text) if p and p.strip()]
    merged: list[str] = []
    for p in parts:
        last = merged[-1].split() if merged else []
        if last and not merged[-1].startswith("§") and (len(last) == 1 or last[-1].lower() in ("to", "for", "on")):
            merged[-1] = f"{merged[-1]} and {p}"  # "Press" + "hold the Side key", "Swipe to" + "tap Reset"
        else:
            merged.append(p)
    return merged


def _clause_steps(clause: str, paths: dict[str, tuple[str, ...]]) -> tuple[list[str], Optional[tuple[str, ...]]]:
    m = PLACEHOLDER_RE.search(clause)
    if not m:
        return ([as_step(BECAUSE_RE.sub("", clause))], None)
    path = paths[m.group(1)]
    pre, post = clause[: m.start()].strip(), clause[m.end():].strip()
    pre = re.sub(r"^(?:[A-Za-z]+ly|then|next|now|also|first),?$", "", pre, flags=re.IGNORECASE)  # "Alternatively,"
    post = re.sub(r"^(?:screen|menu|page)\b\s*", "", post)
    steps = nav_steps(path)
    post = BECAUSE_RE.sub("", post).lstrip(" ,;:")
    if NAV_PRE_RE.match(pre) or not pre:
        if post:
            post = re.sub(r"^(?:and\s+)?to\s+", "", post)
            steps.append(as_step(post))
    else:
        action = PREP_TAIL_RE.sub("", pre)
        steps.append(as_step(f"{action} {post}".strip()))
    return [s for s in steps if s], path


def categorise(text: str, path: Optional[tuple[str, ...]]) -> str:
    if CRITICAL_RE.search(text) and not NOT_PHONE_RE.search(text):
        return "critical"
    return "auto" if path else "manual"


def _lead_clause(body: str, headed: bool) -> Optional[tuple[str, str, str]]:
    """(lead kept in the first step, condition, rest) when a leading condition or phrase precedes an instruction."""
    m = COND_RE.match(body)
    if m:
        rest = strip_preamble(body[m.end():])
        if is_instruction(rest):
            return (body[: m.end()].rstrip().rstrip(",") + ", " if headed else ""), m.group(1).strip(), rest
    if body.startswith("§") or is_instruction(body):
        return None
    w = _first_word(body)
    if w not in LEAD_WORDS and not w.endswith("ly"):
        return None  # "Certain apps, like Netflix, allow you to ..." is a statement, not an instruction
    for rx in (CLAUSE1_RE, CLAUSE2_RE):  # shortest phrase first
        m = rx.match(body)
        if m:
            rest = strip_preamble(body[m.end():])
            if is_instruction(rest):
                return m.group(1) + ", ", "", rest
    return None


def _split_lead(body: str, headed: bool) -> tuple[str, str, str]:
    """-> (lead kept in the first step, condition, body). Labels ('On devices with a Side button:') and
    leading phrases ('Using two fingers,') are kept; If/When conditions are kept only in sectioned
    (official) articles, where they carry device variants ('If your device has a removable battery, ...')."""
    lead = ""
    m = LABEL_RE.match(body)
    if m:
        rest = strip_preamble(body[m.end():])
        if is_instruction(rest) or _lead_clause(rest, headed):
            lead, body = m.group(1) + ": ", rest
    lc = _lead_clause(body, headed)
    if lc:
        return lead + lc[0], lc[1], lc[2]
    return lead, "", body


def sentence_to_unit(idx: int, sentence: str, known: set[str], headed: bool = False,
                     source: Optional[str] = None) -> Optional[Unit]:
    body = strip_preamble(sentence)
    body, ph = rewrite_nav(body, known)
    lead, cond, body = _split_lead(body, headed)
    if not is_instruction(body) or (headed and META_RE.search(body) and len(body.split()) <= 6):
        return None  # "follow these steps" announces steps, it is not one
    spans = find_paths(body, known)
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
    if lead and not NAV_STEP_RE.match(steps[0]):
        first = steps[0]
        keep_case = lead.endswith(": ") or first.split()[0].isupper() or first.split()[0] in ("I",)
        steps[0] = lead + (first if keep_case else first[:1].lower() + first[1:])
    flat = " ".join(steps)
    return Unit(idx=idx, sentence=source or sentence, steps=steps, path=path, category=categorise(flat, path),
                condition=cond)


def parse_units(text: str, known: set[str]) -> list[Unit]:
    units = []
    for i, s in enumerate(split_sentences(text)):
        u = sentence_to_unit(i, s, known)
        if u:
            units.append(u)
    return units


HEADING_CRITICAL_RE = re.compile(r"\b(restart|reboot|factory|safe mode|software update|firmware|reset)\w*", re.I)


def heading_category(heading: str) -> Optional[str]:
    """'Restarting Your Device', 'Safe Mode', 'Factory Data Reset' -> every step below is critical."""
    if heading and HEADING_CRITICAL_RE.search(heading) and not NOT_PHONE_RE.search(heading) \
            and not re.search(r"\bpin\b", heading, re.I):
        return "critical"
    return None


def parse_blocks(blocks, known: set[str]) -> list[Unit]:
    """Units for a sectioned article; each unit remembers its section, block and heading.
    A heading that names a restart / reset / safe-mode procedure makes every step under it critical."""
    units: list[Unit] = []
    idx = 0
    for bi, b in enumerate(blocks):
        crit = heading_category(b.heading)
        sentences = [s for p in b.paragraphs for s in split_sentences(p)]
        for text, origin in merge_nav_lines(sentences, known):
            u = sentence_to_unit(idx, text, known, headed=True, source=" ".join(origin))
            idx += 1
            if not u:
                continue
            u.section, u.block, u.heading, u.headed = b.section, bi, b.heading, True
            if crit and u.category != "critical":
                u.category = crit
            units.append(u)
    return units


# ------------------------------------------------------------------ grouping (rules extractor)
def _is_continuation(u: Unit) -> bool:
    return (u.path is None and u.category == "manual" and _first_word(u.steps[0]) in CONTINUATION
            and not PHYSICAL_RE.search(u.sentence))


def _headed_merge(cur: DraftAction, u: Unit) -> bool:
    """Inside one section of a sectioned article: same-category steps stay one action, except two different
    Settings screens; a manual tap/select right after a Settings screen continues it."""
    if cur.units[-1].section != u.section:
        return False
    if not u.heading and cur.units[-1].block != u.block:
        return False  # an untitled section is a list of separate tips
    if u.category == cur.category:
        if u.category == "auto":
            return u.path is None or cur.path is None or u.path == cur.path
        return True
    return cur.category == "auto" and _is_continuation(u)


def _content_words(steps: list[str]) -> set[str]:
    text = " ".join(s for s in steps if not NAV_STEP_RE.match(s) and not s.startswith("Tap on "))
    return {w.rstrip("s") for w in re.findall(r"[a-z][a-z\-]{3,}", text.lower())} - {
        "this", "that", "your", "with", "from", "then", "when", "want", "wish", "keep", "next", "switch", "tap"}


def _introduces(intro: Unit, a: DraftAction) -> bool:
    """'If you wish to keep your screen protector on, try enabling the Touch sensitivity option.' introduces the
    Settings steps that follow it in the same paragraph block ('... tap the switch next to Touch sensitivity')."""
    first = a.units[0]
    return (intro.headed and intro.section == first.section and intro.block == first.block
            and len(_content_words(intro.steps) & _content_words(a.steps)) >= 2)


def _attach_intros(actions: list[DraftAction]) -> list[DraftAction]:
    out: list[DraftAction] = []
    for a in actions:
        prev = out[-1] if out else None
        if prev and a.category == "auto" and prev.category == "manual" and _introduces(prev.units[-1], a):
            a.units.insert(0, prev.units.pop())  # the sentence announcing these Settings steps moves with them
            if not prev.units:
                out.pop()
        out.append(a)
    return out


def group_units(units: list[Unit]) -> list[DraftAction]:
    actions: list[DraftAction] = []
    for u in units:
        cur = actions[-1] if actions else None
        if u.headed:
            if cur and cur.units[-1].headed and _headed_merge(cur, u):
                cur.units.append(u)
                cur.path = cur.path or u.path
            else:
                actions.append(DraftAction(u.category, u.path, [u]))
            continue
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
    return _attach_intros(actions)


# ------------------------------------------------------------------ naming
CRITICAL_NAMES = [
    (r"force restart|force a restart|volume down (?:key|button)", "Force Restart Your Phone", "It will force the phone to restart"),
    (r"safe mode", "Start Safe Mode", "It will check for problem apps"),
    (r"factory", "Perform a Factory Data Reset", "It will restore factory default settings"),
    (r"reset network|network settings", "Reset Network Settings", "It will restore default network settings"),
    (r"reset all settings", "Reset All Settings", "It will restore all default settings"),
    (r"camera settings|reset settings", "Reset Camera Settings", "It will restore default camera settings"),
    (r"keyboard", "Reset Keyboard Settings", "It will restore default keyboard settings"),
    (r"access point|reset to default", "Reset Access Point Names", "It will restore default mobile data settings"),
    (r"^(?:restart|reboot)|restart the phone|reboot", "Restart Your Phone", "It will refresh the phone system"),
    (r"software update|system software|download and install|update", "Install Software Update",
     "It will install the latest software"),
    (r"restart", "Restart Your Phone", "It will refresh the phone system"),
]
MANUAL_NAMES = [
    (r"^remove|case|cover", "Remove the Case", "It will rule out accessory interference"),
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
# sectioned (official) articles: matched on the first step only, before MANUAL_NAMES
HEADED_MANUAL_NAMES = [
    (r"screen protector|protective film", "Remove the Screen Protector", "It will rule out touch interference"),
    (r"^schedule\b.*repair|mail-in repair", "Schedule a Repair Service", "It will book a device repair"),
    (r"service cent", "Visit a Service Center", "It will get the hardware checked"),
    (r"customer support|support center", "Contact Customer Support", "It will arrange expert help and service"),
    (r"service provider", "Contact Your Service Provider", "It will check your account server"),
    (r"(?:different|another|undamaged)\b.*charger", "Try a Different Charger",
     "It will rule out faulty charging accessories"),
    (r"lint-free|microfiber cloth|wipe the front", "Clean the Device", "It will remove dust and moisture"),
    (r"(?:^|, )(?:check|inspect|examine)\b.*damage", "Check for Physical Damage", "It will rule out hardware damage"),
    (r"charger", "Charge the Device", "It will restore battery power"),
]
HEADING_VERBS = IMPERATIVE | {"mirror", "project"}
AMBIGUOUS_VERBS = {"touch", "swipe", "power", "switch", "set", "back"}
DETERMINERS = {"the", "a", "an", "your", "my", "this", "these", "that", "it", "all", "any", "for", "to", "on", "up",
               "off", "out", "and", "with", "from", "into", "between"}
_ARTICLES = {"the", "a", "an", "your", "any", "all", "that", "it", "and", "or", "its", "this"}
_PREPS = {"on", "with", "to", "for", "of", "from", "if", "by", "as", "so", "until", "in", "at", "within", "into",
          "using", "when", "while", "before", "after", "because", "such", "like"}
_STOP = _ARTICLES | _PREPS
_PARTICLES = {"up", "on", "off", "out", "in", "down"}


def _core(step: str) -> str:
    """The instruction without its label or leading condition ('If X, remove it.' -> 'remove it.')."""
    step = re.sub(r"^[A-Z][^:.?!]{2,60}:\s+", "", step)
    m = COND_RE.match(step)
    if m and is_instruction(step[m.end():]):
        return step[m.end():]
    for rx in (CLAUSE1_RE, CLAUSE2_RE):
        m = rx.match(step)
        if m and is_instruction(step[m.end():]):
            return step[m.end():]
    return step


def _verb_from_gerund(w: str) -> Optional[str]:
    w = w.lower()
    if not w.endswith("ing"):
        return None
    stem = w[:-3]
    for cand in (stem + "e", stem, stem[:-1] if len(stem) > 2 and stem[-1] == stem[-2] else ""):
        if cand in IMPERATIVE:
            return cand
    return None


def _generic_name(step: str) -> str:
    core = _core(step)
    m = re.match(r"(?i)(?:make|get|keep|help)\b.*?\bby (\w+ing)\b (.+)$", core)
    if m and _verb_from_gerund(m.group(1)):  # "make it a little bigger by changing the aspect ratio"
        core = f"{_verb_from_gerund(m.group(1))} {m.group(2)}"
    ws = re.sub(r"[.,;:()\"]", "", core).split()
    out, content = [], 0
    for w in ws:
        lw = w.lower()
        if out and ((lw in _PREPS and content >= 2 and not (lw in _PARTICLES and len(out) == 1))
                    or (lw in _ARTICLES and content >= 3)):
            break
        out.append(w)
        if lw not in _STOP and lw not in _PARTICLES:
            content += 1
        if len(out) >= 5:
            break
    while len(out) > 2 and out[-1].lower() in _STOP:
        out.pop()
    return title_case(" ".join(out)) if out else "Follow These Steps"


def heading_name(heading: str) -> Optional[str]:
    """'Restarting Your Device' -> 'Restart Your Device'; noun-phrase headings ('Charger Issues',
    'Touch Sensitivity Setting') -> None."""
    ws = heading.split()
    if len(ws) < 2:
        return None
    first = ws[0].lower()
    if first.endswith("ing") and (first[:-3] in HEADING_VERBS or first[:-3] + "e" in HEADING_VERBS):
        stem = first[:-3] if first[:-3] in HEADING_VERBS else first[:-3] + "e"
        ws = [stem.capitalize()] + ws[1:]
        first = stem
    if first not in HEADING_VERBS:
        return None
    if first in AMBIGUOUS_VERBS and ws[1].lower() not in DETERMINERS and not ws[1].lower().endswith("'s"):
        return None  # "Touch Sensitivity Setting", "Swipe gestures for ..." are nouns
    return title_case(" ".join(ws))


def _heading_description(name: str) -> str:
    """'Mirror Your TV with Smart View' -> 'It will mirror your TV' (cut at a joint, never mid-name)."""
    ws = [w if w.isupper() else w.lower() for w in name.split()]
    if len(ws) > 5:
        cut = max((i for i, w in enumerate(ws[:6]) if i >= 2 and w in _PREPS | {"and", "or"}), default=None)
        ws = ws[:cut] if cut else ws
    return fit_description("It will " + " ".join(ws))


def _match(table, text: str) -> Optional[tuple[str, str]]:
    for pat, name, desc in table:
        if re.search(pat, text):
            return name, desc
    return None


def _heading_pattern(a: DraftAction) -> Optional[tuple[str, str]]:
    """'Safe Mode' -> Start Safe Mode, 'Charger Issues' -> Charge the Device ..."""
    if a.category == "auto" or not a.heading:
        return None
    return _match(CRITICAL_NAMES if a.category == "critical" else HEADED_MANUAL_NAMES, a.heading.lower())


def _pattern_name(a: DraftAction) -> Optional[tuple[str, str]]:
    if a.category == "auto":
        return None
    table = CRITICAL_NAMES if a.category == "critical" else MANUAL_NAMES
    if not a.units[0].headed:  # plain articles: match on every step (unchanged behaviour)
        return _match(table, " ".join(a.steps).lower())
    first = a.steps[0].lower()
    if a.category == "manual":
        return (_match(HEADED_MANUAL_NAMES, first) or _match(HEADED_MANUAL_NAMES, _core(a.steps[0]).lower())
                or _match(MANUAL_NAMES, _core(a.steps[0]).lower()))
    return _match(table, _core(a.steps[0]).lower()) or _match(table, " ".join(a.steps).lower())


def name_action(a: DraftAction, use_heading: bool = False) -> tuple[str, str]:
    hn, hp = (heading_name(a.heading), _heading_pattern(a)) if use_heading and a.heading else (None, None)
    if hn:
        return hn, hp[1] if hp else _heading_description(hn)
    if hp and a.category == "critical":  # 'Safe Mode' names the procedure better than its first key press
        return hp
    pat = _pattern_name(a) or hp
    if pat:
        return pat
    if a.category in ("manual", "critical") or not a.path:
        name = _generic_name(a.steps[0])
        if a.units[0].headed:  # official articles: describe the named instruction ("It will change the aspect ratio")
            return name, _heading_description(name)
        return name, fit_description("It will help resolve the problem")
    leaf = a.path[-1] if a.path else "Settings"
    lt = leaf.lower() if not any(c.isupper() for c in leaf[1:]) else leaf
    body = " ".join(a.steps).lower()
    joined = " ".join(a.steps)
    onoff = next((m for m in (re.match(r"(?:optionally )?turn (?:it |them )?(on|off)\b", s.lower()) for s in a.steps)
                  if m), None)
    if "optimize now" in body:
        return "Optimize the Device", "It will close apps and free memory"
    obj = re.search(r"\bTurn (on|off) ([A-Z][^.,]*)", joined)
    if obj:  # "Turn on Put unused apps to sleep" -> name the toggle, not the parent screen
        target = " ".join(re.split(r" and | if | when ", obj.group(2))[0].split()[:5])
        return (title_case(f"Turn {obj.group(1)} {target}"),
                fit_description(f"It will turn {obj.group(1)} {target.lower()}"))
    sw = re.search(r"\bswitch(?:es)? next to ([A-Z][^.,]*?)(?: to | or |[.,]|$)", joined)
    if sw:  # "tap the switch next to Touch sensitivity to disable it"
        target = " ".join(sw.group(1).split()[:4])
        verb = "Off" if re.search(r"\b(disable|turn (?:it )?off)\b", body) else "On"
        return title_case(f"Turn {verb} {target}"), fit_description(f"It will turn {verb.lower()} {target.lower()}")
    search = re.search(r"\b[Ss]earch for and select ([A-Z][^.,]*?)(?: and |[.,]|$)", joined)
    if search and leaf == "Settings":
        target = search.group(1)
        return title_case(f"Adjust {target}"), fit_description(f"It will let you adjust {target.lower()}")
    if onoff:
        return title_case(f"Turn {onoff.group(1)} {leaf}"), fit_description(f"It will turn {onoff.group(1)} {lt}")
    if re.search(r"\bcheck\b", body):
        return title_case(f"Check {leaf}"), fit_description(f"It will show {lt} details")
    noun = leaf if leaf.lower().endswith("settings") else f"{leaf} Settings"
    return title_case(f"Configure {noun}"), fit_description(f"It will let you adjust {lt}")


def name_actions(actions: list[DraftAction]) -> None:
    """The first action of each headed section takes the section heading as its name when the heading is an
    instruction ('Clear the Email App's Cache and Data') or names a known procedure ('Safe Mode'); the rest
    are named by rules. A repeated name falls back to the action's first step."""
    seen_sections: set[int] = set()
    used: set[str] = set()
    for a in actions:
        first_in_section = bool(a.heading) and a.units[0].section not in seen_sections
        if a.heading:
            seen_sections.add(a.units[0].section)
        a.name, a.description = name_action(a, use_heading=first_in_section)
        if a.name.lower() in used:
            alt = _generic_name(a.steps[0])
            if alt.lower() not in used:
                a.name = alt
        used.add(a.name.lower())


def rules_extract(text: str, known: set[str]) -> list[DraftAction]:
    actions = group_units(parse_units(text, known))
    for a in actions:
        a.name, a.description = name_action(a)
    return actions
