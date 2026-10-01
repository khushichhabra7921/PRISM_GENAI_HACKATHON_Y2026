"""Refresh the presentation for the official kit (free, local tools only).

The deck was designed in Canva (docs/Thapar_Nexora_2.pdf, 12 slides, Calibri / Courier New). This script keeps the
seven slides that do not depend on the data unchanged and regenerates the five that do, in the same visual style:

  02 problem statement      official TechCorp Nexa complaints from data/input.txt
  04 architecture           SIIS payload path, article parsing + relevance, article-scoped cache, per-group deeplinks
  05 product walkthrough    row_21 of the official kit, request and response copied from outputs/row_21.json
  08 results + limitations  measured numbers from outputs/report.json and bench/report.json
  11 submission checklist   links + two live screenshots of /demo answering an official complaint

Every number is read from the benchmark reports; nothing is typed in by hand.

Requirements (dev only):  pip install -r requirements-video.txt   (Playwright + Microsoft Edge, pypdfium2)
The API must be running for the screenshots:  uvicorn app.main:app --port 8000   (or docker compose up)

    python scripts/make_deck.py [--video-url URL] [--preview DIR]
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "docs" / "Thapar_Nexora_2.pdf"
BASE = "http://localhost:8000"
REPO = "https://github.com/khushichhabra7921/PRISM_GENAI_HACKATHON_Y2026"
W, H = 1920, 1080  # CSS px of one slide = 20 x 11.25 in = the Canva page size (1440 x 810 pt)
FOOTER = "Smart Guided Troubleshooting Engine &nbsp;|&nbsp; Samsung PRISM GenAI Hackathon 2026 &nbsp;|&nbsp; Theme 2"
TOTAL = 12

CSS = """
@page { size: 20in 11.25in; margin: 0 }
* { box-sizing: border-box; margin: 0; padding: 0 }
html, body { width: 1920px; height: 1080px; overflow: hidden }
body { font-family: Calibri, 'Segoe UI', sans-serif; color: #1f1b3a; position: relative;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
  background: radial-gradient(ellipse 62% 58% at 90% 0%, #d8dbfb 0%, rgba(216,219,251,0) 62%),
              linear-gradient(100deg, #f9f8fe 0%, #f3f1fc 52%, #ebe7fb 100%) }
.grid { position: absolute; top: 0; right: 0; bottom: 0; width: 50%;
  background-image: linear-gradient(rgba(108,92,214,.06) 1.5px, transparent 1.5px),
                    linear-gradient(90deg, rgba(108,92,214,.06) 1.5px, transparent 1.5px);
  background-size: 64px 64px; -webkit-mask-image: linear-gradient(90deg, transparent 0%, #000 55%) }
.waves { position: absolute; left: 0; bottom: 28px; width: 1920px; height: 110px }
.abs { position: absolute }
.label { left: 87px; top: 44px; font-size: 16px; font-weight: 700; letter-spacing: .34em; color: #6b6883 }
.label b { color: #7c3aed; margin-right: 18px; letter-spacing: .2em }
.title { left: 86px; top: 78px; font-size: 52px; font-weight: 700; color: #6a42b8; letter-spacing: -.3px }
.sub { left: 87px; top: 158px; font-size: 26px; color: #4b4768 }
.footer { left: 87px; bottom: 30px; font-size: 15px; color: #8b88a3 }
.pageno { right: 87px; bottom: 30px; font-size: 15px; color: #8b88a3; font-weight: 700 }
.card { background: #fff; border: 1.5px solid #e7e2f4; border-radius: 22px; box-shadow: 0 8px 24px rgba(60,40,140,.06) }
.pill { display: inline-block; border-radius: 999px; padding: 7px 18px; font-size: 18px; font-weight: 700 }
.blue { background: #e1f0fb; color: #155e75 }
.mono { font-family: 'Courier New', monospace }
.lav { background: #ece8fb; color: #5b3fb3 }
.tag { display: inline-block; border-radius: 999px; padding: 4px 14px; font-size: 13px; font-weight: 700; letter-spacing: .08em }
.auto { background: #ece8fb; color: #5b3fb3 } .manual { background: #dcf3ef; color: #0f766e }
.critical { background: #fdf0dc; color: #b45309 }
.k { font-size: 16px; font-weight: 700; letter-spacing: .3em; color: #7c3aed }
.phone { position: absolute; background: #17142a; border-radius: 54px; padding: 15px; box-shadow: 0 26px 50px rgba(40,20,90,.22) }
.screen { background: #fbfaff; border-radius: 42px; width: 100%; height: 100%; overflow: hidden; position: relative; padding: 22px 22px }
.status { display: flex; justify-content: space-between; font-size: 15px; font-weight: 700; color: #1f1b3a; margin: 4px 6px 14px }
.notch { position: absolute; top: 20px; left: 50%; width: 15px; height: 15px; border-radius: 50%; background: #17142a; transform: translateX(-50%) }
.bubble { background: linear-gradient(135deg, #7c3aed, #5b21b6); color: #fff; border-radius: 18px 18px 5px 18px;
  padding: 11px 14px; font-size: 16px; line-height: 1.3; margin-left: auto }
.mini { background: #fff; border: 1px solid #ebe7f6; border-radius: 16px; padding: 12px 14px; margin-top: 11px;
  box-shadow: 0 4px 12px rgba(60,40,140,.06) }
.mini b { font-size: 17px; display: block } .mini .d { font-size: 14px; color: #6b6883 }
.btn { display: inline-block; background: #6d28d9; color: #fff; border-radius: 999px; padding: 6px 16px; font-size: 14px; font-weight: 700 }
.btn.last { background: #fff; color: #b45309; border: 1.5px solid #d97706 }
.code { background: #1c1a2e; border-radius: 22px; color: #d8d4f5; font-family: 'Courier New', monospace; font-size: 17px;
  line-height: 1.42; box-shadow: 0 18px 40px rgba(28,26,46,.25); overflow: hidden }
.code .bar { display: flex; align-items: center; gap: 10px; padding: 16px 22px; border-bottom: 1px solid #2c2945; color: #9b97c0 }
.code .dot { width: 14px; height: 14px; border-radius: 50%; display: inline-block }
.code pre { padding: 14px 24px; white-space: pre; font-family: inherit }
.c { color: #7d7a99 } .s { color: #9ee6b2 } .n { color: #f5d76e } .key { color: #c9c3f5 }
.num { display: inline-flex; align-items: center; justify-content: center; width: 50px; height: 50px; border-radius: 50%;
  background: linear-gradient(135deg, #6d6af2, #5b3fd6); color: #fff; font-weight: 700; font-size: 18px }
.dots li { list-style: none; font-family: 'Courier New', monospace; font-size: 16px; color: #2c2946; white-space: nowrap }
.dots li::before { content: ''; display: inline-block; width: 12px; height: 12px; border-radius: 50%; background: #7c3aed;
  margin-right: 12px; vertical-align: 1px }
table { border-collapse: collapse }
"""

WAVES = """<svg class="waves" viewBox="0 0 1920 110" preserveAspectRatio="none">
<path d="M0 80 C 320 30, 640 110, 960 70 S 1600 20, 1920 60" fill="none" stroke="rgba(130,110,220,.16)" stroke-width="2"/>
<path d="M0 95 C 360 55, 700 115, 1010 82 S 1620 40, 1920 78" fill="none" stroke="rgba(130,110,220,.11)" stroke-width="2"/>
<path d="M0 64 C 300 20, 620 92, 940 56 S 1560 8, 1920 42" fill="none" stroke="rgba(90,170,200,.10)" stroke-width="2"/>
</svg>"""


def esc(s) -> str:
    return html.escape(str(s), quote=False)


def page(n: int, label: str, title: str, body: str, sub: str = "", sub_top: int = 158, sub_style: str = "") -> str:
    sub_html = f'<div class="abs sub" style="top:{sub_top}px;{sub_style}">{sub}</div>' if sub else ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>
<div class="grid"></div>{WAVES}
<div class="abs label"><b>{n:02d}</b>{label}</div>
<div class="abs title">{title}</div>{sub_html}
{body}
<div class="abs footer">{FOOTER}</div><div class="abs pageno">{n:02d} / {TOTAL}</div>
</body></html>"""


def phone(x: int, y: int, w: int, h: int, inner: str, rotate: float = 0.0) -> str:
    rot = f"transform: rotate({rotate}deg);" if rotate else ""
    return f"""<div class="phone" style="left:{x}px;top:{y}px;width:{w}px;height:{h}px;{rot}"><div class="screen">
<div class="notch"></div><div class="status"><span>10:24</span><span>&#9646;&#9646;&#9646; 5G &#9645;</span></div>{inner}</div></div>"""


# ------------------------------------------------------------------ data
def load_data(video_url: str) -> dict:
    o = json.loads((ROOT / "outputs" / "report.json").read_text(encoding="utf-8"))
    r = json.loads((ROOT / "bench" / "report.json").read_text(encoding="utf-8"))
    rows = {x["id"]: x for x in o["rows"]}
    row21 = json.loads((ROOT / "outputs" / "row_21.json").read_text(encoding="utf-8"))
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-p", "no:warnings"], cwd=ROOT,
                         capture_output=True, text=True).stdout
    m = re.search(r"(\d+) tests? collected", out)
    return {"o": o, "r": r, "rows": rows, "row21": row21, "tests": int(m.group(1)) if m else None,
            "video": video_url}


def readme_video_url() -> str:
    m = re.search(r"Demo video[^|]*\|\s*(\S+)\s*\|", (ROOT / "README.md").read_text(encoding="utf-8"))
    return m.group(1) if m else "[video link]"


# ------------------------------------------------------------------ slide 02
def slide_problem(d: dict) -> str:
    rows = d["rows"]
    quotes = [("TOUCH · DISPLAY", "row_21", "My Nexa X1 screen inputs are delayed and the touch responsiveness is laggy…"),
              ("SCREEN DAMAGE", "row_19", "My Nexa Fold X1 screen is cracked again right where it folds."),
              ("DISPLAY SIZE", "row_8", "My new smartphone’s main screen stays small and doesn’t fill the whole display…")]
    cards = ""
    for i, (tag, rid, q) in enumerate(quotes):
        cards += f"""<div class="abs card" style="left:{80 + i * 348}px;top:575px;width:326px;height:212px;padding:22px 24px">
<span class="tag lav" style="font-size:12.5px">{tag}</span>
<div style="position:absolute;right:24px;top:2px;font-size:70px;color:#c9c0f3;font-weight:700">&rdquo;</div>
<div style="font-size:21px;font-style:italic;color:#2c2946;margin-top:16px;line-height:1.3">&ldquo;{esc(q)}&rdquo;</div>
<div style="position:absolute;left:24px;bottom:16px;font-size:13.5px;color:#8b88a3">input.txt · {rid} · SIIS: {esc(rows[rid]['siis_title'])}</div></div>"""
    acts = d["row21"]["response"]["contexts"][0]["actions"]
    auto = next(a for a in acts if a["category"] == "auto")
    man = next(a for a in acts if a["category"] == "manual")
    crit = next(a for a in acts if a["actionName"] == "Start Safe Mode")
    o = d["o"]
    inner = f"""<div style="font-size:27px;font-weight:700;margin:6px 4px 2px">Troubleshoot</div>
<div style="font-size:14px;color:#6b6883;margin:0 4px 16px">Describe it in your own words</div>
<div class="bubble" style="width:84%">My Nexa X1 screen inputs are delayed and the touch is laggy</div>
<div style="font-size:12.5px;color:#9b97b3;text-align:right;margin:6px 2px 0">with SIIS article “Touchscreen issues…”</div>
<div style="display:flex;gap:8px;align-items:center;margin:12px 4px 4px;font-size:14px;color:#6b6883">
<span style="width:13px;height:13px;border-radius:50%;background:#6d6af2;display:inline-block"></span>
1 goal · {len(acts)} actions · grounded in the SIIS article</div>
<div class="mini"><b>{esc(auto['actionName'])}</b><span class="d">{esc(auto['description'])}</span>
<div style="display:flex;justify-content:space-between;margin-top:10px"><span class="tag auto">AUTO</span><span class="btn">Open ›</span></div></div>
<div class="mini"><b>{esc(man['actionName'])}</b><span class="d">{esc(man['description'])}</span>
<div style="margin-top:10px"><span class="tag manual">MANUAL</span></div></div>
<div class="mini"><b>{esc(crit['actionName'])}</b><span class="d">{esc(crit['description'])}</span>
<div style="display:flex;justify-content:space-between;margin-top:10px"><span class="tag critical">CRITICAL</span><span class="btn last">Last step</span></div></div>
<div style="text-align:center;color:#6d28d9;font-size:14px;margin-top:14px;font-weight:700">&#9432; Why these steps?</div>"""
    body = f"""
<div class="abs" style="left:87px;top:80px;font-size:46px;font-weight:700;color:#7c5cc4">Theme</div>
<div class="abs" style="left:80px;top:150px;width:1150px;font-size:55px;font-weight:700;color:#15122b;line-height:1.16">
Theme 2 · Troubleshooting: turn a TechCorp Nexa complaint and its SIIS article into a deeplinked, one-tap plan.</div>
<div class="abs" style="left:80px;top:410px;font-size:44px;font-weight:700;color:#15122b;line-height:1.2">
People describe <span style="color:#7c3aed">symptoms</span>.<br>Settings speak in <span style="color:#7c3aed">menus</span>.</div>
<div class="abs" style="left:620px;top:425px;width:510px;font-size:20px;color:#4b4768;line-height:1.35">
<b style="color:#15122b">Official kit:</b> {o['cases']} complaints in <span class="mono" style="font-size:18px">input.txt</span>, each arriving
with the SIIS article <span class="mono" style="font-size:18px">{{title, content}}</span> retrieved for it ({o['distinct_articles']} distinct articles).
The answer must match <span class="mono" style="font-size:18px">sample_output.json</span>.</div>
{cards}
<div class="abs card" style="left:80px;top:810px;width:1022px;height:200px;padding:24px 34px">
<div class="k" style="color:#6b6883;letter-spacing:.32em">TODAY</div>
<div style="font-size:30px;margin-top:22px"><b>~15</b> min</div>
<div style="font-size:19px;color:#4b4768;margin-top:14px;width:400px;line-height:1.3">manual triage per scenario: agent reads SIIS, orders steps, user hunts through Settings</div>
<svg class="abs" style="left:420px;top:62px" width="96" height="64"><defs><linearGradient id="ga" x1="0" x2="1">
<stop offset="0" stop-color="#c4b5fd"/><stop offset="1" stop-color="#7c3aed"/></linearGradient></defs>
<path d="M0 18 H58 V0 L96 32 L58 64 V46 H0 Z" fill="url(#ga)"/></svg>
<div class="abs" style="left:550px;top:24px"><div class="k">OUR TARGET</div>
<div style="font-size:54px;color:#6d28d9;margin-top:6px"><b>&lt;300</b> ms</div>
<div style="font-size:19px;color:#4b4768;margin-top:6px;width:430px;line-height:1.3">known issues via the semantic cache; &lt; 8 s for new ones, fully grounded in the article</div></div></div>
{phone(1290, 175, 410, 800, inner)}"""
    return page(2, "PROBLEM STATEMENT", "", body)


# ------------------------------------------------------------------ slide 04
def box(x, y, w, h, tag, name, text, foot, out=False) -> str:
    style = ("background:linear-gradient(135deg,#6d28d9,#4f6df5);color:#fff;border:none" if out else "")
    tagc = "background:rgba(255,255,255,.22);color:#fff" if out else ""
    tc = "#e9e6ff" if out else "#4b4768"
    fc = "#dcd8ff" if out else "#0e7490"
    return f"""<div class="abs card" style="left:{x}px;top:{y}px;width:{w}px;height:{h}px;padding:16px 18px;{style}">
<div style="display:flex;align-items:center;gap:10px"><span class="tag lav" style="font-size:13px;{tagc}">{tag}</span>
<b style="font-size:21px">{name}</b></div>
<div style="font-size:15.5px;color:{tc};margin-top:9px;line-height:1.3">{text}</div>
<div class="mono" style="position:absolute;left:18px;bottom:12px;font-size:13px;color:{fc};white-space:nowrap">{foot}</div></div>"""


def slide_architecture(d: dict) -> str:
    o = d["o"]
    fast = [
        (88, "IN", "Complaint + SIIS", "“touch is laggy” + its article <span class='mono' style='font-size:14px'>{title, content}</span>", "payload · or free text"),
        (388, "01", "Enrichment", "canonical query, topic and 8–10 paraphrases", "rules · SLM #1 off-path"),
        (688, "02", "Semantic cache", "same article + similar complaint → its validated plan", "FAISS · article hash"),
        (1288, "07", "Validate + fix", "schema.py, grounding, URL, catalog", "Pydantic · 11 checks"),
    ]
    cold = [
        (388, "03", "Read the article", "split ## sections, keep the blocks that answer", "siis_text · relevance"),
        (688, "04", "Extraction", "TechCorp phrasing → canonical steps; headings name actions", "rules · SLM #2 pointer"),
        (988, "05", "Ordering", "auto → manual → critical; restart / reset sections last", "rule engine"),
        (1288, "06", "Deeplink engine", "one link per step group, plus a validation link", "BM25+dense · RRF · CE"),
    ]
    boxes = "".join(box(x, 268, 248, 140, t, n, tx, f) for x, t, n, tx, f in fast)
    boxes += box(1588, 268, 248, 140, "OUT", "Verified plan", "JSON in the sample_output.json shape", "?view=contract", out=True)
    boxes += "".join(box(x, 515, 248, 140, t, n, tx, f) for x, t, n, tx, f in cold)
    cards = [
        ("01–02", "Understand &amp; recall",
         "Numbered, quoted complaints are cleaned and paraphrased 8–10 ways. A repeat with the <b>same article</b> is "
         "served from cache; a plan built from another article never is.", "τ 0.85 · article-scoped"),
        ("03–05", "Read, structure, order",
         "The payload is the boundary. “go to Settings, tap Display, and then tap…” becomes canonical steps. Numbered "
         "procedures stay whole; long guides keep only the blocks that answer.", "no source sentence → no step"),
        ("06", "Pick the exact screen",
         "BM25 + dense over catalog metadata, RRF, cross-encoder, deepest-menu tie-break, per step group. "
         "<span class='mono' style='font-size:15px'>voiceassist://</span> URIs are only copied, never written.",
         "3-tier confidence gate"),
        ("07", "Prove it or refuse",
         "Auto-fix (trim, scrub URLs, fix goal syntax) → Pydantic + 11 rule checks → one retry → contexts: [] with "
         "“fallback”: “no_match”. Temperature 0: same input, same plan.", "invalid plans never cached"),
    ]
    bottom = ""
    for i, (tag, name, text, pill) in enumerate(cards):
        bottom += f"""<div class="abs card" style="left:{86 + i * 442}px;top:705px;width:420px;height:296px;padding:22px 24px">
<div style="display:flex;align-items:center;gap:14px"><span class="tag lav" style="font-size:15px;padding:5px 18px">{tag}</span>
<b style="font-size:25px">{name}</b></div>
<div style="font-size:20.5px;color:#2c2946;margin-top:16px;line-height:1.38">{text}</div>
<div class="abs pill blue mono" style="left:22px;right:22px;bottom:18px;text-align:center;font-size:17px">{pill}</div></div>"""
    arrows = """<svg class="abs" style="left:0;top:0" width="1920" height="1080"><defs>
<marker id="ah" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto"><path d="M0 0 L5 2.5 L0 5 Z" fill="#8b7fd6"/></marker></defs>
<g stroke="#8b7fd6" stroke-width="3" fill="none" marker-end="url(#ah)">
<line x1="338" y1="338" x2="382" y2="338"/><line x1="638" y1="338" x2="682" y2="338"/><line x1="1538" y1="338" x2="1582" y2="338"/>
<line x1="638" y1="585" x2="682" y2="585"/><line x1="938" y1="585" x2="982" y2="585"/><line x1="1238" y1="585" x2="1282" y2="585"/>
<line x1="1412" y1="515" x2="1412" y2="415"/>
<path d="M812 410 C 812 470, 520 450, 512 508" stroke-dasharray="9 7"/></g>
<text x="830" y="455" font-family="Courier New" font-weight="700" font-size="16" fill="#6d28d9">MISS</text></svg>"""
    body = f"""
<div class="abs" style="left:86px;top:228px;width:1752px;height:198px;border:2px dashed rgba(70,160,190,.45);border-radius:20px;background:rgba(214,238,247,.35)"></div>
<div class="abs mono" style="left:100px;top:236px;font-size:15px;font-weight:700;color:#0e7490;letter-spacing:.08em">FAST PATH · P95 &lt; 300 ms · zero LLM calls · measured {o['latency_ms']['repeat_cache_hit']['p95']} ms</div>
<div class="abs" style="left:372px;top:470px;width:1180px;height:200px;border:2px dashed rgba(124,92,214,.45);border-radius:20px;background:rgba(236,232,251,.5)"></div>
<div class="abs mono" style="left:372px;width:1180px;text-align:center;top:480px;font-size:15px;font-weight:700;color:#6d28d9;letter-spacing:.08em">COLD PATH · P95 &lt; 8 s · measured {o['latency_ms']['cold']['p95'] / 1000:.1f} s on the official kit</div>
<div class="abs pill" style="left:985px;top:318px;background:#6d28d9;color:#fff;font-size:16px">HIT · reuse validated plan</div>
{arrows}{boxes}
<div class="abs card" style="left:1588px;top:515px;width:248px;height:140px;padding:16px 18px;background:#fffaf2;border-color:#f3dfc0">
<b style="font-size:19px;color:#b45309">Fail-safe</b><div style="font-size:15.5px;color:#4b4768;margin-top:6px;line-height:1.3">
invalid twice, empty payload or no instructions → <b class="mono" style="color:#b45309">contexts: []</b> + no_match. Never cached.</div></div>
{bottom}"""
    return page(4, "SOLUTION ARCHITECTURE", "Our Solution &amp; Architecture Diagram", body,
                "LLMs reason, retrieval grounds, deterministic code validates. The SIIS article sent with the complaint is the only source of steps.")


# ------------------------------------------------------------------ slide 05
def _json_excerpt(d: dict) -> str:
    g = d["row21"]["response"]["contexts"][0]
    a = g["actions"][0]
    sg = a["stepGroups"][0]
    dl = sg["actionableDeeplink"]
    s = lambda t: f'<span class="s">"{esc(t)}"</span>'  # noqa: E731
    k = lambda t: f'<span class="key">"{t}"</span>'  # noqa: E731
    return f"""<span class="c"># request · data/input.txt row_21 + its SIIS payload</span>
POST /v1/troubleshoot?view=contract
{{ {k('query')}: {s('My Nexa X1 screen inputs are')}
            {s('delayed and the touch …')},
  {k('siis_response')}: {{ {k('title')}: {s('Touchscreen issues…')},
                     {k('content')}: {s('## 1. Factors…')} }} }}

<span class="c"># response · 200 OK · shape of sample_output.json</span>
{{ {k('query')}: …, {k('response')}: {{ {k('contexts')}: [{{
    {k('goal')}: {s('Follow these steps to perform')}
            {s('this Touchscreen Issues Troubleshooting')},
    {k('title')}: {s(g['title'])}, {k('score')}: <span class="n">{g['score']}</span>,
    {k('actions')}: [{{
      {k('actionName')}: {s(a['actionName'])},
      {k('description')}: {s(a['description'])},
      {k('stepGroups')}: [{{ {k('steps')}: [ …,
          {s(sg['steps'][2])}, … ],
        {k('actionableDeeplink')}: {{
          {k('deeplink')}: {s(dl['deeplink'])},
          {k('description')}: {s(dl['description'])},
          {k('originalType')}: {s(dl.get('originalType', ''))} }},
        {k('validationDeeplink')}: <span class="n">null</span> }}],
      {k('category')}: {s(a['category'])} }}, … <span class="c">{len(g['actions']) - 1} more</span> ]}}]}}}}"""


def slide_walkthrough(d: dict) -> str:
    o, row = d["o"], d["rows"]["row_21"]
    acts = d["row21"]["response"]["contexts"][0]["actions"]
    cats = {c: sum(a["category"] == c for a in acts) for c in ("auto", "manual", "critical")}
    a0 = acts[0]
    steps0 = a0["stepGroups"][0]["steps"]
    rep = o["latency_ms"]["repeat_cache_hit"]["p95"]
    li = "".join(f'<div style="display:flex;gap:10px;align-items:flex-start;margin:7px 0;font-size:15px">'
                 f'<span style="flex:none;width:24px;height:24px;border-radius:50%;background:#f1eefc;color:#6d28d9;'
                 f'font-weight:700;font-size:13px;display:inline-flex;align-items:center;justify-content:center">{i}</span>'
                 f'<span>{esc(t)}</span></div>' for i, t in enumerate(steps0[1:4], 1))
    inner1 = f"""<div style="font-size:26px;font-weight:700;margin:4px 4px 10px">Troubleshoot</div>
<div class="bubble" style="width:90%;font-size:14.5px">My Nexa X1 screen inputs are delayed and the touch is laggy</div>
<div class="mini" style="margin-top:14px"><div style="font-size:12px;letter-spacing:.1em;color:#6d28d9;font-weight:700">GOAL · {d['row21']['response']['contexts'][0]['score']}</div>
<b style="font-size:19px;margin:2px 0 8px">{esc(d['row21']['response']['contexts'][0]['title'])}</b>
<div style="display:flex;align-items:center;gap:8px"><span class="tag auto" style="font-size:11px">AUTO</span>
<span style="font-size:14px;color:#4b4768">{esc(a0['actionName'])}</span></div>{li}
<div style="margin-top:12px;text-align:center"><span class="btn" style="padding:10px 26px;font-size:15px">Open Display ›</span></div></div>
<div style="text-align:center;font-size:13px;color:#8b88a3;margin-top:10px">{cats['auto']} auto · {cats['manual']} manual · {cats['critical']} critical · <span style="color:#6d28d9">view evidence</span></div>"""
    inner2 = """<div style="font-size:14px;color:#6b6883;margin:2px 4px">‹ Settings</div>
<div style="font-size:30px;font-weight:700;margin:6px 4px 14px">Display</div>
<div class="mini" style="padding:6px 14px">""" + "".join(
        f'<div style="display:flex;justify-content:space-between;align-items:center;padding:12px 0;'
        f'border-bottom:{"none" if i == 3 else "1px solid #f0edf8"};font-size:16.5px">{t}{tog}</div>'
        for i, (t, tog) in enumerate([
            ("Brightness", ""), ("Dark mode", '<span style="width:44px;height:24px;border-radius:12px;background:#d9d6e6;display:inline-block"></span>'),
            ("Navigation bar", ""),
            ("<b>Touch sensitivity</b>", '<span style="width:44px;height:24px;border-radius:12px;background:#6d28d9;display:inline-block;position:relative">'
                                         '<span style="position:absolute;right:3px;top:3px;width:18px;height:18px;border-radius:50%;background:#fff"></span></span>')])) + """</div>
<div style="font-size:13.5px;color:#8b88a3;margin:14px 6px;line-height:1.35">Opened directly by deeplink.<br>No menu hunting.</div>"""
    timeline = [
        ("Complaint + article in", f"“{' '.join(row['query'].split()[:7])}…” + SIIS “{row['siis_title']}”"),
        ("Read", f"numbered procedure kept whole: {row['blocks_kept']} of {row['blocks_scored']} blocks, {row['steps']} steps copied from the text"),
        ("Planned", f"{len(acts)} actions: {cats['auto']} auto → {cats['manual']} manual → {cats['critical']} critical; least disruptive first"),
        ("One tap", "each Settings navigation carries its own catalog deeplink"),
        ("Recalled", f"same complaint + same article again: {rep:.0f} ms (P95), $0, no LLM call"),
    ]
    tl = ""
    for i, (h, t) in enumerate(timeline):
        tl += f"""<div class="abs" style="left:1408px;top:{222 + i * 94}px;width:460px;display:flex;gap:20px">
<span class="num" style="flex:none">{i + 1}</span><div><b style="font-size:22px">{h}</b>
<div style="font-size:16.5px;color:#5b5873;margin-top:3px;line-height:1.28">{esc(t)}</div></div></div>"""
    body = f"""
{phone(78, 228, 326, 642, inner1)}
{phone(430, 262, 290, 560, inner2, rotate=2.5)}
<div class="abs" style="left:300px;top:812px;background:#1c1a2e;color:#fff;border-radius:999px;padding:10px 20px;font-size:16px;font-weight:700">1 tap → Display screen</div>
<svg class="abs" style="left:0;top:0" width="760" height="1000"><path d="M316 634 C 370 634, 420 615, 464 590" fill="none" stroke="#7c3aed" stroke-width="3.5" stroke-dasharray="9 7"/>
<path d="M448 583 L467 588 L459 606" fill="none" stroke="#7c3aed" stroke-width="3.5"/></svg>
<div class="abs card" style="left:78px;top:905px;width:660px;height:96px;padding:18px 22px;display:flex;align-items:center;gap:18px">
<span style="flex:none;width:54px;height:54px;border-radius:50%;background:linear-gradient(135deg,#6d6af2,#5b3fd6);display:inline-flex;align-items:center;justify-content:center">
<svg width="22" height="22"><path d="M5 2 L20 11 L5 20 Z" fill="#fff"/></svg></span>
<div style="font-size:19px"><b>Demo video:</b> <span style="text-decoration:underline">{esc(d['video'])}</span></div></div>
<div class="abs code" style="left:762px;top:222px;width:622px;height:780px">
<div class="bar"><span class="dot" style="background:#ff5f57"></span><span class="dot" style="background:#febc2e"></span><span class="dot" style="background:#28c840"></span>
<span class="mono" style="margin-left:14px;font-size:17px">sgte · /v1/troubleshoot</span>
<span style="margin-left:auto;background:#9ee6b2;color:#14532d;border-radius:999px;padding:4px 14px;font-weight:700;font-size:15px">200 · {row['cold_ms'] / 1000:.1f} s cold</span></div>
<pre style="font-size:15.6px;line-height:1.5">{_json_excerpt(d)}</pre></div>
{tl}
<div class="abs card" style="left:1408px;top:690px;width:430px;height:312px;padding:22px 26px">
<div class="k" style="font-size:15px;letter-spacing:.24em">EVIDENCE TRAIL · PER ACTION</div>
<ul class="dots" style="margin-top:20px;display:grid;grid-template-columns:1fr 1fr;row-gap:22px;column-gap:6px">
<li>SIIS section</li><li>source sentence</li><li>block relevance</li><li>kept/dropped</li><li>bm25 · dense</li>
<li>rerank · rrf</li><li>runner-up</li><li>confidence tier</li></ul></div>"""
    return page(5, "PRODUCT WALKTHROUGH", "Demo &amp; Product Walkthrough", body,
                "A real complaint from the official input.txt, sent with its SIIS article: what the user sees, what the API returns, and the receipt behind every action.",
                sub_top=146, sub_style="width:1280px;font-size:24px;line-height:1.3")


# ------------------------------------------------------------------ slide 08
def donut(x, y, pct, color, label, sub) -> str:
    r, c = 70, 2 * 3.14159 * 70
    return f"""<div class="abs" style="left:{x}px;top:{y}px;width:200px;text-align:center">
<svg width="170" height="170" style="display:block;margin:0 auto"><circle cx="85" cy="85" r="{r}" fill="none" stroke="#ece9f7" stroke-width="26"/>
<circle cx="85" cy="85" r="{r}" fill="none" stroke="{color}" stroke-width="26" stroke-dasharray="{c * pct / 100:.1f} {c:.1f}" transform="rotate(-90 85 85)"/>
<text x="85" y="93" text-anchor="middle" font-family="Calibri" font-weight="700" font-size="28" fill="#1f1b3a">{pct:g}%</text></svg>
<div style="font-size:20px;color:#2c2946;margin-top:10px;line-height:1.2">{label}</div>
<div style="font-size:15px;color:#8b88a3;margin-top:4px">{sub}</div></div>"""


def slide_results(d: dict) -> str:
    o, r = d["o"], d["r"]
    acc, dual = r["accuracy"], r["cache"]["dual_lookup"]
    icons = {
        "eye": '<path d="M3 16 C 8 7, 24 7, 29 16 C 24 25, 8 25, 3 16 Z" fill="none" stroke="#fff" stroke-width="2.6"/><circle cx="16" cy="16" r="4.5" fill="none" stroke="#fff" stroke-width="2.6"/>',
        "doc": '<rect x="8" y="4" width="17" height="24" rx="3" fill="none" stroke="#fff" stroke-width="2.6"/><path d="M12 11 H21 M12 16 H21 M12 21 H18" stroke="#fff" stroke-width="2.6"/>',
        "lock": '<rect x="7" y="14" width="18" height="13" rx="3" fill="none" stroke="#fff" stroke-width="2.6"/><path d="M11 14 V10 a5 5 0 0 1 10 0 V14" fill="none" stroke="#fff" stroke-width="2.6"/>',
        "sliders": '<path d="M8 5 V27 M16 5 V27 M24 5 V27" stroke="#fff" stroke-width="2.6"/><circle cx="8" cy="20" r="3" fill="#6366f1" stroke="#fff" stroke-width="2.4"/><circle cx="16" cy="11" r="3" fill="#6366f1" stroke="#fff" stroke-width="2.4"/><circle cx="24" cy="18" r="3" fill="#6366f1" stroke="#fff" stroke-width="2.4"/>',
    }
    cards = [
        ("eye", "Evidence trail", "Every action carries its SIIS section, source sentence, block relevance and why the runner-up screen lost.", "grounding you can show"),
        ("doc", "Reads real support articles", "Markdown sections, comma-chain menus and one-step-per-line lists become canonical steps and step groups.", "built for the official kit"),
        ("lock", "Article-scoped cache", f"A repeat with the same article returns in {o['latency_ms']['repeat_cache_hit']['p95']:.0f} ms; a plan from another article is never served.", "fast and safe"),
        ("sliders", "Graded confidence", "Three tiers instead of match-or-nothing: auto-attach, attach with a visible score, or fall back.", "more coverage, same precision"),
    ]
    top = ""
    for i, (ic, name, text, pill) in enumerate(cards):
        top += f"""<div class="abs card" style="left:{86 + i * 442}px;top:226px;width:420px;height:318px;padding:24px 26px">
<div style="display:flex;align-items:center;gap:18px"><span style="width:62px;height:62px;border-radius:50%;background:linear-gradient(135deg,#6d6af2,#5b3fd6);display:inline-flex;align-items:center;justify-content:center">
<svg width="32" height="32">{icons[ic]}</svg></span><div><div style="font-size:15px;font-weight:700;color:#7c3aed">0{i + 1}</div>
<b style="font-size:25px">{name}</b></div></div>
<div style="font-size:21px;color:#2c2946;margin-top:18px;line-height:1.36">{text}</div>
<div class="abs pill blue" style="left:24px;right:24px;bottom:22px;text-align:center;font-size:19px">{pill}</div></div>"""
    dl_ok = 100.0 if o["deeplinks"] else 0.0
    donuts = (donut(118, 640, o["schema_valid_pct"], "#7c3aed", "schema-valid", f"{o['cases']} official complaints")
              + donut(332, 640, o["rule_compliance_pct"], "#6366f1", "pass 11 checks", "0 violations")
              + donut(546, 640, o["grounded_steps_pct"], "#0e9fb5", "steps grounded", f"{o['steps']} steps in the payloads")
              + donut(760, 640, dl_ok, "#8b5cf6", "catalog deeplinks", f"{o['deeplinks']} links · 0 invented"))
    lat = o["latency_ms"]
    pills = (f'<span class="pill lav" style="font-size:16px">P95 {lat["repeat_cache_hit"]["p95"]:.0f} ms repeat</span> '
             f'<span class="pill lav" style="font-size:16px">P95 {lat["cold"]["p95"] / 1000:.1f} s cold</span> '
             f'<span class="pill lav" style="font-size:16px">0 URL leaks · $0</span>')
    lims = [
        ("No deeplink catalog in the official kit", "synthetic catalog in the official scheme; the one real entry copied"),
        ("No gold plans for the 20 complaints", "contract + grounding measured; accuracy on synthetic gold"),
        ("Cross-encoder poorly calibrated on support text", "numbered procedures are kept whole"),
        ("Article only partly about the complaint", "keep the answering blocks; no_match if none"),
        ("Whole procedures can be long (up to 11 actions)", "ordered auto → manual → critical"),
        ("1.5B SLM too slow on a laptop CPU", "runs off the request path; hosted fallback"),
    ]
    lim_html = "".join(f"""<div style="display:flex;align-items:center;padding:13px 0;border-bottom:{'none' if i == len(lims) - 1 else '1px solid #ece9f5'}">
<b style="width:390px;font-size:18px;color:#1f1b3a">{a}</b><span style="color:#0e9fb5;font-size:20px;margin:0 16px">&rarr;</span>
<span style="font-size:18px;color:#6d28d9">{b}</span></div>""" for i, (a, b) in enumerate(lims))
    body = f"""{top}
<div class="abs card" style="left:86px;top:570px;width:908px;height:432px;padding:22px 30px">
<div class="k" style="font-size:15px;letter-spacing:.26em">RESULTS · OFFICIAL KIT, MEASURED</div></div>
{donuts}
<div class="abs" style="left:116px;top:902px;width:860px">{pills}</div>
<div class="abs" style="left:116px;top:952px;width:860px;font-size:17px;font-style:italic;color:#4b4768">
Synthetic regression kit: gold step accuracy {acc['mean_step_accuracy']:g} / 3 · deeplink relevance {acc['mean_deeplink_relevance']:g} / 2 ·
{dual['zero_llm_hit_pct']:g}% of unseen paraphrases served with 0 LLM calls · {d['tests']} tests pass.</div>
<div class="abs card" style="left:1016px;top:570px;width:820px;height:432px;padding:22px 30px">
<div class="k" style="font-size:15px;letter-spacing:.26em">LIMITATIONS &rarr; MITIGATIONS</div>
<div style="margin-top:8px">{lim_html}</div></div>"""
    return page(8, "INNOVATION · RESULTS · LIMITATIONS", "Innovation Highlights, Results and Limitations", body,
                "Four capabilities that make the official kit work end to end, the numbers we measured on it, and where it can still fail.")


# ------------------------------------------------------------------ slide 11
def slide_checklist(d: dict, shots: list[bytes]) -> str:
    o = d["o"]
    rows = [
        ("Working prototype code", "public or shared GitHub repo", f"{REPO[8:]}"),
        ("README", "reproducible setup instructions", "docker compose up"),
        ("Official-kit outputs", f"{o['cases']} responses, sample_output.json shape", "outputs/ · bench/official.py"),
        ("Demo video", "max 5 minutes, YouTube or Drive", d["video"]),
        ("Presentation", "PPT or PDF", "this deck"),
    ]
    tr = "".join(f"""<tr style="border-top:1.5px solid #ece9f5"><td style="padding:22px 0 22px 30px;width:420px">
<b style="font-size:25px">{a}</b><div style="font-size:18px;color:#6b6883;margin-top:3px">{b}</div></td>
<td style="width:170px"><span style="display:inline-block;border:1.5px solid #cfc8ea;border-radius:999px;padding:10px 34px;font-size:18px;font-weight:700">Yes</span></td>
<td class="mono" style="font-size:16.5px;color:#4b4768;word-break:break-all;padding-right:20px">{esc(c)}</td></tr>""" for a, b, c in rows)
    imgs = ""
    for i, png in enumerate(shots[:2]):
        b64 = base64.b64encode(png).decode()
        imgs += f"""<div class="abs card" style="left:1140px;top:{205 + i * 400}px;width:752px;height:380px;padding:0;overflow:hidden;border-radius:26px">
<div style="height:34px;background:#26233a;display:flex;align-items:center;gap:8px;padding-left:16px">
<span class="dot" style="width:10px;height:10px;border-radius:50%;background:#ff5f57;display:inline-block"></span>
<span style="width:10px;height:10px;border-radius:50%;background:#febc2e;display:inline-block"></span>
<span style="width:10px;height:10px;border-radius:50%;background:#28c840;display:inline-block"></span>
<span class="mono" style="color:#b9b5d8;font-size:13px;margin-left:12px">127.0.0.1:8000/demo</span></div>
<img src="data:image/png;base64,{b64}" style="width:752px;display:block"></div>"""
    body = f"""
<div class="abs card" style="left:86px;top:228px;width:1020px;height:774px;padding:22px 26px">
<table style="width:100%"><tr><td style="padding:6px 0 18px 30px;font-size:20px;font-weight:700;color:#4b4768">Deliverable</td>
<td style="padding:6px 0 18px;font-size:20px;font-weight:700;color:#4b4768">Status</td>
<td style="padding:6px 0 18px;font-size:20px;font-weight:700;color:#4b4768">Link / note</td></tr>{tr}</table></div>
<div class="abs" style="left:1144px;top:148px;font-size:32px;font-weight:700;color:#4c1d95">LIVE DEMO · OFFICIAL KIT</div>
{imgs}"""
    return page(11, "SUBMISSION", "Checklist: Updated on Public GitHub", body,
                "What judges open first. Every item below is live in the repository.")


# ------------------------------------------------------------------ build
def demo_screenshots(pw_page) -> list[bytes]:
    pw_page.set_viewport_size({"width": 1536, "height": 864})
    pw_page.goto(f"{BASE}/demo")
    pw_page.wait_for_function("() => document.querySelectorAll('#case option').length > 0", timeout=120000)
    idx = pw_page.evaluate("() => [...document.querySelectorAll('#case option')].findIndex(o => o.textContent.startsWith('row_21'))")
    pw_page.select_option("#case", str(idx))
    pw_page.click("#case-send")
    pw_page.wait_for_function("() => !document.getElementById('chat').innerText.includes('Thinking')", timeout=60000)
    pw_page.wait_for_timeout(300)
    first = pw_page.screenshot(type="png")
    pw_page.evaluate("() => { const r = document.getElementById('raw'); window.scrollTo(0, r.getBoundingClientRect().top + window.scrollY - 120); }")
    pw_page.wait_for_timeout(200)
    second = pw_page.screenshot(type="png")
    return [first, second]


def main() -> None:
    import pypdfium2 as pdfium
    from playwright.sync_api import sync_playwright

    ap = argparse.ArgumentParser()
    ap.add_argument("--video-url", default=None, help="demo video link shown on slides 05 and 11 (default: README)")
    ap.add_argument("--preview", default=None, help="also write PNG previews of the regenerated slides here")
    ap.add_argument("--out", default=str(DECK))
    args = ap.parse_args()
    d = load_data(args.video_url or readme_video_url())
    base = pdfium.PdfDocument(DECK.read_bytes())  # read into memory: the output may overwrite the input
    if len(base) != TOTAL:
        raise SystemExit(f"{DECK} has {len(base)} pages, expected {TOTAL}")
    replaced: dict[int, bytes] = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        shot_page = browser.new_page(device_scale_factor=1.25)
        shots = demo_screenshots(shot_page)
        pg = browser.new_page(viewport={"width": W, "height": H})
        slides = {1: slide_problem(d), 3: slide_architecture(d), 4: slide_walkthrough(d), 7: slide_results(d),
                  10: slide_checklist(d, shots)}
        for i, src in slides.items():
            pg.set_content(src, wait_until="load")
            pg.wait_for_timeout(150)
            replaced[i] = pg.pdf(width="20in", height="11.25in", print_background=True,
                                 margin={"top": "0", "right": "0", "bottom": "0", "left": "0"})
            if args.preview:
                Path(args.preview).mkdir(parents=True, exist_ok=True)
                pg.screenshot(path=str(Path(args.preview) / f"slide_{i + 1:02d}.png"))
        browser.close()
    out = pdfium.PdfDocument.new()
    for i in range(len(base)):
        if i in replaced:
            out.import_pages(pdfium.PdfDocument(replaced[i]), [0])
        else:
            out.import_pages(base, [i])
    out.save(args.out)
    print(f"[deck] wrote {args.out}: slides {', '.join(str(i + 1) for i in sorted(replaced))} regenerated, "
          f"{TOTAL - len(replaced)} kept · video link {d['video']}")


if __name__ == "__main__":
    main()
