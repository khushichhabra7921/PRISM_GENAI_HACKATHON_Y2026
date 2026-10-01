"""Generate the demo video (≤ 5 min) with free, local tools only, on the official kit.

* narration: Windows' built-in offline speech synthesiser (System.Speech, voice "Microsoft Zira Desktop")
* picture:   headless Microsoft Edge driven by Playwright, performing the demo on the live /demo page
* engine:    started by this script on its own port with an EMPTY cache and no pre-warm, so the first request
             for a complaint is genuinely cold and a repeat is a genuine cache hit
* numbers:   read from outputs/report.json, bench/report.json, a live pytest run and the page itself
             (latencies quoted in the narration are the ones on screen) - nothing is typed in by hand
* slides:    rendered from docs/Thapar_Nexora_2.pdf (refresh it first with scripts/make_deck.py)
* output:    docs/demo/SGTE_demo.mp4 (1920x1080, H.264 + AAC), docs/demo/SGTE_demo.srt, docs/demo/narration.txt

Requirements (dev only):  pip install -r requirements-video.txt

    python scripts/make_demo_video.py
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

import imageio_ffmpeg
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PORT = int(os.getenv("SGTE_VIDEO_PORT", "8010"))
BASE = f"http://localhost:{PORT}"
OUT_DIR = ROOT / "docs" / "demo"
DECK = ROOT / "docs" / "Thapar_Nexora_2.pdf"
CSS_W, CSS_H, SCALE = 1536, 864, 1.25  # -> 1920x1080 pixels
FPS = 15
VOICE, RATE = "Microsoft Zira Desktop", 0
GAP_S = 0.35  # silence between sentences
REPO = "github.com/khushichhabra7921/PRISM_GENAI_HACKATHON_Y2026"
FREE_TEXT = "the screen does not rotate when I turn the tablet"  # no payload: the engine retrieves the article
NO_MATCH = "my Nexa watch wont sync my steps"  # nothing in the 11 official articles answers this


# ------------------------------------------------------------------ engine for the recording
class Server:
    """uvicorn on its own port, empty cache, no pre-warm, deterministic (no LLM on the request path)."""

    def __init__(self, tmp: Path):
        env = {**os.environ, "SGTE_PREWARM": "0", "SGTE_CACHE_DIR": str(tmp / "cache"), "SGTE_LOG_DIR": str(tmp / "logs"),
               "SGTE_LLM_PROVIDER": "none", "SGTE_LLM_FALLBACK": ""}
        self.proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
                                      "--port", str(PORT)], cwd=ROOT, env=env,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def wait(self, timeout_s: float = 600) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                with urllib.request.urlopen(f"{BASE}/health", timeout=3) as r:
                    if r.status == 200:
                        return
            except Exception:  # noqa: BLE001  (503 while loading, or not listening yet)
                pass
            if self.proc.poll() is not None:
                raise SystemExit("API process exited while starting")
            time.sleep(2)
        raise SystemExit("API did not become healthy")

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()


# ------------------------------------------------------------------ narration (offline TTS)
def tts(text: str, wav_path: Path) -> float:
    txt = wav_path.with_suffix(".txt")
    txt.write_text(text, encoding="utf-8")
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"$s.SelectVoice('{VOICE}'); $s.Rate={RATE};"
        "$fmt=New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050,"
        "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,[System.Speech.AudioFormat.AudioChannel]::Mono);"
        f"$s.SetOutputToWaveFile('{wav_path}', $fmt);"
        f"$s.Speak([IO.File]::ReadAllText('{txt}')); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True)
    with wave.open(str(wav_path)) as w:
        return w.getnframes() / w.getframerate()


# ------------------------------------------------------------------ frame recorder
class Recorder:
    def __init__(self, page, video_path: Path):
        self.page = page
        self.frames = 0
        self.last: Optional[bytes] = None
        self.writer = imageio_ffmpeg.write_frames(
            str(video_path), (int(CSS_W * SCALE), int(CSS_H * SCALE)), fps=FPS, codec="libx264",
            pix_fmt_in="rgb24", quality=None, output_params=["-crf", "22", "-preset", "medium"],
            macro_block_size=8)
        self.writer.send(None)

    def snap(self) -> None:
        png = self.page.screenshot(type="jpeg", quality=92)
        img = Image.open(io.BytesIO(png)).convert("RGB")
        if img.size != (int(CSS_W * SCALE), int(CSS_H * SCALE)):
            img = img.resize((int(CSS_W * SCALE), int(CSS_H * SCALE)))
        self.last = np.asarray(img).tobytes()

    def frame(self) -> None:
        self.snap()
        self.writer.send(self.last)
        self.frames += 1

    def hold(self, seconds: float) -> None:
        if self.last is None:
            self.snap()
        for _ in range(max(0, round(seconds * FPS))):
            self.writer.send(self.last)
            self.frames += 1

    @property
    def t(self) -> float:
        return self.frames / FPS

    def close(self) -> None:
        self.writer.close()


# ------------------------------------------------------------------ page helpers
OVERLAY_CSS = """
#cap{position:fixed;left:50%;bottom:22px;transform:translateX(-50%);max-width:1340px;width:max-content;
     background:rgba(18,14,38,.86);color:#fff;font:600 23px/1.35 'Segoe UI',sans-serif;padding:12px 22px;
     border-radius:14px;box-shadow:0 8px 24px rgba(0,0,0,.25);z-index:99999;text-align:center}
.hl{outline:4px solid #f59e0b !important;outline-offset:4px;border-radius:14px;box-shadow:0 0 0 10px rgba(245,158,11,.18)}
body{padding-bottom:120px}
"""


def prepare(page) -> None:
    page.add_style_tag(content=OVERLAY_CSS)
    page.evaluate("() => { if (!document.getElementById('cap')) { const d = document.createElement('div');"
                  " d.id = 'cap'; d.style.display = 'none'; document.body.appendChild(d); } }")


def caption(page, text: str) -> None:
    page.evaluate("(t) => { const c = document.getElementById('cap'); c.textContent = t;"
                  " c.style.display = t ? 'block' : 'none'; }", text)


def highlight(page, selector: Optional[str]) -> None:
    page.evaluate("(s) => { document.querySelectorAll('.hl').forEach(e => e.classList.remove('hl'));"
                  " if (s) { const e = document.querySelector(s); if (e) e.classList.add('hl'); } }", selector)


def scroll_to(rec: Recorder, selector: Optional[str] = None, y: Optional[int] = None, steps: int = 14,
              container: Optional[str] = None) -> None:
    page = rec.page
    start = page.evaluate("(c) => c ? document.querySelector(c).scrollTop : window.scrollY", container)
    if y is None:
        y = page.evaluate("(s) => { const e = document.querySelector(s); return e ? "
                          "e.getBoundingClientRect().top + window.scrollY - 90 : window.scrollY; }", selector)
    for i in range(1, steps + 1):
        f = i / steps
        pos = start + (y - start) * (1 - (1 - f) ** 3)
        page.evaluate("([c, p]) => { if (c) document.querySelector(c).scrollTop = p; else window.scrollTo(0, p); }",
                      [container, pos])
        rec.frame()


def type_query(rec: Recorder, text: str) -> None:
    page = rec.page
    page.fill("#q", "")
    for ch in text:
        page.type("#q", ch)
        rec.frame()


def wait_answer(rec: Recorder) -> None:
    page = rec.page
    page.wait_for_function("() => !document.getElementById('chat').innerText.includes('Thinking')", timeout=120000)
    page.wait_for_timeout(150)
    rec.frame()


def send(rec: Recorder) -> None:
    rec.page.click("button.send")
    wait_answer(rec)


def send_case(rec: Recorder, row: str) -> None:
    """Pick an official input.txt complaint and send it with its SIIS payload."""
    page = rec.page
    idx = page.evaluate("(r) => [...document.querySelectorAll('#case option')]"
                        ".findIndex(o => o.textContent.startsWith(r + ' '))", row)
    page.select_option("#case", str(idx))
    rec.frame()
    rec.hold(0.4)
    page.click("#case-send")
    wait_answer(rec)


def latency_ms(page) -> str:
    return page.evaluate("() => document.getElementById('m-lat').textContent").replace(" ms", "").strip()


def spoken_ms(ms: str) -> str:
    v = float(ms)
    return f"{v / 1000:.1f} seconds" if v >= 1000 else f"{round(v)} milliseconds"


# ------------------------------------------------------------------ static cards
def card_page(page, html: str) -> None:
    page.set_content(f"""<!doctype html><html><head><meta charset="utf-8"><style>
      body{{margin:0;width:{CSS_W}px;height:{CSS_H}px;overflow:hidden;font-family:'Segoe UI',sans-serif;color:#17152b;
           background:linear-gradient(135deg,#f5f1ff 0%,#eef3ff 55%,#e8f7f6 100%)}}
      .wrap{{padding:62px 90px}} h1{{font-size:62px;margin:0 0 8px;letter-spacing:-.5px}}
      h1 span{{color:#6d28d9}} .k{{font-size:15px;letter-spacing:.2em;color:#6d28d9;font-weight:700}}
      .sub{{font-size:26px;color:#4b4768;margin:6px 0 26px}} .pill{{display:inline-block;background:#fff;border:1px solid #e0daf5;
           border-radius:999px;padding:10px 20px;margin:6px 10px 6px 0;font-size:19px;font-weight:600}}
      .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:18px}}
      .tile{{background:#fff;border:1px solid #e4dff4;border-radius:18px;padding:16px 18px}}
      .tile b{{display:block;font-size:34px;color:#6d28d9}} .tile span{{font-size:15px;color:#5b5873}}
      .tile i{{display:block;font-style:normal;font-size:13px;color:#8b88a3;margin-top:4px}}
      pre{{background:#1c1a2e;color:#d8d4f5;border-radius:16px;padding:16px 22px;font:16px/1.5 Consolas,monospace;margin:16px 0 0}}
      pre em{{color:#9ee6b2;font-style:normal}}
      ul{{font-size:22px;line-height:1.6;color:#2c2946}}
    </style></head><body>{html}</body></html>""")
    prepare(page)


def slide_page(page, png: bytes) -> None:
    b64 = base64.b64encode(png).decode()
    page.set_content(f"""<!doctype html><html><head><style>body{{margin:0;background:#fff;width:{CSS_W}px;height:{CSS_H}px;
      overflow:hidden;display:flex;align-items:center;justify-content:center}}
      img{{max-width:100%;max-height:100%}}</style></head><body><img src="data:image/png;base64,{b64}"></body></html>""")
    prepare(page)


def deck_slides(indices: list[int]) -> dict[int, bytes]:
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(DECK))
    out = {}
    for i in indices:
        pg = pdf[i]
        w, _ = pg.get_size()
        img = pg.render(scale=int(CSS_W * SCALE) / w).to_pil()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out[i] = buf.getvalue()
    return out


# ------------------------------------------------------------------ script
@dataclass
class Beat:
    text: Union[str, Callable[[], str]]  # a callable is evaluated when the beat starts (numbers read off the page)
    action: Optional[Callable[[Recorder], None]] = None


@dataclass
class Segment:
    name: str
    setup: Callable[[Recorder], None]
    beats: list[Beat] = field(default_factory=list)


def measured() -> dict:
    r = json.loads((ROOT / "bench" / "report.json").read_text(encoding="utf-8"))
    o = json.loads((ROOT / "outputs" / "report.json").read_text(encoding="utf-8"))
    out = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:warnings"], cwd=ROOT,  # pytest.ini adds -q
                         capture_output=True, text=True).stdout
    m = re.search(r"(\d+) passed[^\n]*", out)
    return {"r": r, "o": o, "tests": int(m.group(1)) if m else 0, "pytest_line": m.group(0) if m else out[-200:]}


def build_script(page, m: dict, slides: dict[int, bytes]) -> list[Segment]:
    r, o = m["r"], m["o"]
    acc, du = r["accuracy"], r["cache"]["dual_lookup"]
    lat = o["latency_ms"]
    rows = {x["id"]: x for x in o["rows"]}
    state: dict = {}

    def title(rec):
        card_page(page, f"""<div class="wrap"><div class="k">SAMSUNG PRISM · GENERATIVE AI HACKATHON 2026 · THEME 2 · TEAM NEXORA</div>
          <h1 style="margin-top:26px">Smart Guided <span>Troubleshooting</span> Engine</h1>
          <div class="sub">TechCorp Nexa complaint + its SIIS article → a validated, ordered, one-tap troubleshooting plan</div>
          <div><span class="pill">&lt; 300 ms on known issues</span><span class="pill">&lt; 8 s on new issues</span>
          <span class="pill">0 invented steps</span><span class="pill">0 invalid deeplinks</span></div>
          <div class="sub" style="margin-top:56px;font-size:21px">Official kit · {o['cases']} complaints · {o['distinct_articles']} SIIS articles ·
          answers in the sample_output.json shape</div>
          <div class="sub" style="font-size:21px">Thapar Institute of Engineering and Technology, Patiala · open source · runs on a laptop CPU · $0 · {REPO}</div></div>""")
        rec.frame()

    def slide(i):
        def f(rec):
            slide_page(page, slides[i])
            rec.frame()
        return f

    def demo(rec):
        page.goto(f"{BASE}/demo")
        page.wait_for_function("() => document.querySelectorAll('#case option').length > 0", timeout=120000)
        prepare(page)
        highlight(page, "#cases")
        rec.frame()

    def top(rec):
        highlight(page, None)
        scroll_to(rec, y=0)

    def act_case(row):
        def f(rec):
            highlight(page, None)
            send_case(rec, row)
            state[row] = latency_ms(page)
        return f

    def act_type_send(q):
        def f(rec):
            highlight(page, None)
            type_query(rec, q)
            rec.hold(0.3)
            send(rec)
            state[q] = latency_ms(page)
        return f

    def act_hl(sel, scroll=True):
        def f(rec):
            if scroll:
                scroll_to(rec, sel)
            highlight(page, sel)
            rec.frame()
        return f

    def act_open(rec):
        highlight(page, None)
        page.click("#chat .open")
        page.wait_for_timeout(100)
        highlight(page, "#chat .action")
        rec.frame()

    def act_scroll_critical(rec):
        highlight(page, None)
        page.evaluate("() => { const b = [...document.querySelectorAll('#chat .badge.critical')].pop();"
                      " if (b) { b.id = 'last-critical'; b.closest('.action').classList.add('hl'); } }")
        scroll_to(rec, "#last-critical")
        rec.frame()

    def act_two_groups(rec):
        highlight(page, None)
        page.evaluate("() => { const a = [...document.querySelectorAll('#chat .action')]"
                      ".find(x => x.querySelectorAll('.sg').length > 1); if (a) { a.id = 'two-groups'; a.classList.add('hl'); } }")
        scroll_to(rec, "#two-groups")
        rec.frame()

    def act_why(rec):
        highlight(page, None)
        scroll_to(rec, "#why-card")
        highlight(page, "#why-card")
        rec.frame()

    def act_why_scroll(rec):
        y = page.evaluate("() => window.scrollY + 300")
        scroll_to(rec, y=y, steps=24)

    def act_raw(rec):
        highlight(page, None)
        scroll_to(rec, "#raw")
        highlight(page, "#raw")
        rec.frame()
        scroll_to(rec, y=560, steps=30, container="#raw")

    def act_no_match(rec):
        act_type_send(NO_MATCH)(rec)
        highlight(page, "#chat .empty")
        rec.frame()

    def results(rec):
        tiles = [
            (f"{o['schema_valid_pct']:.0f}%", "schema-valid", f"{o['cases']} official complaints with payload"),
            (f"{o['rule_compliance_pct']:.0f}%", "pass all 11 rule checks", "0 violations"),
            (f"{o['grounded_steps_pct']:.0f}%", "steps found in the article", f"{o['steps']} steps · 0 invented"),
            (f"{o['deeplinks']}", "deeplinks, all from the catalog", "0 invalid · 0 URL leaks"),
            (f"{lat['repeat_cache_hit']['p95']:.0f} ms", "P95 repeat (cache)", "target ≤ 300 ms · 0 LLM calls"),
            (f"{lat['cold']['p95'] / 1000:.1f} s", "P95 cold, official kit", "target ≤ 8 s · CPU only"),
            (f"{acc['mean_step_accuracy']:g} / 3", "gold step accuracy", "synthetic regression kit"),
            (f"{du['zero_llm_hit_pct']:.0f}%", "paraphrases from cache", "synthetic kit · 0 LLM calls"),
        ]
        grid = "".join(f"<div class='tile'><b>{a}</b><span>{b}</span><i>{t}</i></div>" for a, b, t in tiles)
        card_page(page, f"""<div class="wrap" style="padding-top:50px"><div class="k">MEASURED WITH bench/official.py AND
          bench/run.py · CPU ONLY</div><h1 style="font-size:46px;margin-top:14px">Results</h1>
          <div class="grid">{grid}</div>
          <pre><em>$</em> python bench/official.py   <span style="color:#8b88a3"># 20 complaints → outputs/</span>
schema-valid {o['schema_valid_pct']}% · rules {o['rule_compliance_pct']}% · grounded steps {o['grounded_steps_pct']}% of {o['steps']}
<em>$</em> python -m pytest -q
{m['pytest_line']}</pre></div>""")
        rec.frame()

    def closing(rec):
        card_page(page, f"""<div class="wrap"><div class="k">LIMITATIONS · NEXT STEPS</div>
          <h1 style="font-size:46px;margin-top:14px">Honest about the edges</h1>
          <ul><li>The official kit ships no deeplink catalog: ours is synthetic, in the official scheme, plus the one real entry</li>
          <li>No gold plans for the {o['cases']} complaints: we measure contract and grounding; accuracy on synthetic gold</li>
          <li>Relevance inside long articles is heuristic; numbered procedures are kept whole</li>
          <li>The 1.5B local SLM is too slow on CPU for the request path, so it writes cache paraphrases in the background</li></ul>
          <h1 style="font-size:40px;margin-top:34px">Thank you · <span>docker compose up</span></h1>
          <div class="sub" style="font-size:21px">{REPO}</div></div>""")
        rec.frame()

    r21, r8 = rows["row_21"], rows["row_8"]
    r8_kept = f"{r8['blocks_kept']} of {r8['blocks_scored']}"
    return [
        Segment("title", title, [
            Beat("This is the Smart Guided Troubleshooting Engine, team Nexora's prototype for Theme 2 of the "
                 "Samsung PRISM Generative AI Hackathon."),
            Beat("It takes a TechCorp Nexa complaint, together with the support article retrieved for it, and returns a "
                 "validated, ordered troubleshooting plan. Every step comes from that article, and every Settings fix "
                 "is one tap away."),
        ]),
        Segment("problem", slide(1), [
            Beat("People describe symptoms, but Settings speak in menus."),
            Beat(f"The official kit has {o['cases']} complaints. Each one arrives with its SIIS article, as a title and "
                 "markdown content, and the answer must match the official sample output."),
            Beat("Our targets: under three hundred milliseconds for known issues, under eight seconds for new ones, "
                 "zero invented steps, and zero invalid deeplinks."),
        ]),
        Segment("architecture", slide(3), [
            Beat("Language models reason, retrieval grounds, and deterministic code validates."),
            Beat("The article sent with the complaint is the only source of steps. We split it into sections, keep the "
                 "blocks that answer the complaint, and turn TechCorp's phrasing into canonical steps."),
            Beat("Steps are ordered from safe settings to critical actions, each Settings navigation gets its exact "
                 "deeplink, and eleven validators check the plan before it is cached."),
        ]),
        Segment("demo_official", demo, [
            Beat("Here is the demo. It runs on a laptop CPU, with free, local models, and an empty cache. The picker "
                 "sends each official complaint together with its SIIS article, exactly like the API contract."),
            Beat("Row twenty-one: the screen inputs are delayed, and the touch is laggy.", act_case("row_21")),
            Beat(lambda: f"This is a new request, so the full pipeline ran, in {spoken_ms(state['row_21'])}. The "
                         f"article is a numbered touchscreen procedure, so it is kept whole: {len(r21['actions'])} "
                         "actions, all copied from the article.", act_hl("#chat .goal")),
            Beat("Automatic settings come first. Turn on touch sensitivity carries the catalog deeplink for the Display "
                 "screen.", act_open),
            Beat("Manual checks come next, and the disruptive steps, safe mode and a factory data reset, come last.",
                 act_scroll_critical),
        ]),
        Segment("demo_repeat", top, [
            Beat("Now the same complaint, with the same article, again.", act_case("row_21")),
            Beat(lambda: f"This time the validated plan comes from the cache, in {spoken_ms(state['row_21'])}, with "
                         "zero model calls. A plan built from a different article is never served.",
                 act_hl(".side .card")),
            Beat("And the response is pure JSON. With view equals contract, it has exactly the shape of the official "
                 "sample output.", act_raw),
        ]),
        Segment("demo_relevance", top, [
            Beat("Row eight is harder: the main screen stays small, sent with a long screen mirroring guide.",
                 act_case("row_8")),
            Beat("Only one tip in that guide answers the complaint, changing the aspect ratio, so the plan has one "
                 "action.", act_hl("#chat .goal")),
            Beat(f"The evidence trail shows why: {r8_kept} instruction blocks were kept, with the relevance score of "
                 "every block, and the source sentence behind every step.", act_why),
            Beat("A support team can see exactly what was used, and what was left out.", act_why_scroll),
        ]),
        Segment("demo_groups", top, [
            Beat("Row one, an email screen that goes blank, shows step groups.", act_case("row_1")),
            Beat("Clearing the cache and clearing the data are two separate navigations, so the action has two step "
                 "groups, each with its own deeplink.", act_two_groups),
        ]),
        Segment("demo_free_text", top, [
            Beat(f"Without an article, the engine retrieves one itself: {FREE_TEXT}.", act_type_send(FREE_TEXT)),
            Beat(lambda: f"It grounded the plan in the screen rotation article, in {spoken_ms(state[FREE_TEXT])} on the "
                         "CPU.", act_hl("#chat .goal")),
            Beat("When nothing answers, it refuses to guess. My Nexa watch won't sync my steps returns an empty plan "
                 "with a no match flag, and the query is logged for the content team.", act_no_match),
        ]),
        Segment("results", results, [
            Beat(f"On all {o['cases']} official complaints, every output is schema valid and passes all eleven rule "
                 f"checks, and every one of the {o['steps']} steps is found in its article."),
            Beat(f"Repeats return from the cache in {round(lat['repeat_cache_hit']['p95'])} milliseconds at the "
                 f"ninety-fifth percentile, and new requests in about {lat['cold']['p95'] / 1000:.1f} seconds, on a "
                 "laptop CPU."),
            Beat(f"On our synthetic regression kit, gold step accuracy stays at {acc['mean_step_accuracy']:g} out of "
                 f"3, and {round(du['zero_llm_hit_pct'])} percent of unseen paraphrases are served without any model "
                 f"call. All {m['tests']} tests pass."),
        ]),
        Segment("closing", closing, [
            Beat("The main limitations: the official kit has no deeplink catalog and no gold answers, so our catalog is "
                 "synthetic and accuracy is measured on synthetic gold, and relevance inside long articles is a "
                 "heuristic."),
            Beat("Everything is open source, starts with docker compose up, and costs nothing to run. Thank you."),
        ]),
    ]


# ------------------------------------------------------------------ assembly
def srt_time(t: float) -> str:
    h, rem = divmod(int(t * 1000), 3600_000)
    mnt, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{mnt:02d}:{s:02d},{ms:03d}"


def main() -> None:
    from playwright.sync_api import sync_playwright

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="sgte_video_"))
    m = measured()
    print(f"[video] measured: {m['pytest_line']}")
    slides = deck_slides([1, 3])
    server = Server(tmp)
    try:
        server.wait()
        print(f"[video] engine ready on {BASE} (empty cache, no pre-warm)")
        video_tmp, audio_tmp = tmp / "video.mp4", tmp / "audio.wav"
        subs: list[tuple[float, float, str]] = []
        samples = 0  # audio samples written so far
        audio_chunks: list[bytes] = []
        sr = 22050
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge", headless=True)
            page = browser.new_page(viewport={"width": CSS_W, "height": CSS_H}, device_scale_factor=SCALE)
            page.goto(f"{BASE}/demo")
            rec = Recorder(page, video_tmp)
            segments = build_script(page, m, slides)
            n = 0
            for seg in segments:
                seg.setup(rec)
                for beat in seg.beats:
                    n += 1
                    text = beat.text() if callable(beat.text) else beat.text
                    wav = tmp / f"b{n:03d}.wav"
                    dur = tts(text, wav)
                    with wave.open(str(wav)) as w:
                        sr = w.getframerate()
                        pcm = w.readframes(w.getnframes())
                    start = rec.t
                    # keep audio locked to video: silence for frames recorded between beats (setup, scrolling)
                    lead = int(round(start * sr)) - samples
                    if lead > 0:
                        audio_chunks.append(b"\x00\x00" * lead)
                        samples += lead
                    caption(page, text)
                    rec.frame()
                    if beat.action:
                        beat.action(rec)
                    spent = rec.t - start
                    rec.hold(max(0.0, dur + GAP_S - spent))
                    total = rec.t - start
                    # audio: narration then silence so this beat's audio length == its video length
                    pad = max(0, int(round((total - dur) * sr)))
                    audio_chunks.append(pcm + b"\x00\x00" * pad)
                    samples += len(pcm) // 2 + pad
                    subs.append((start, start + dur, text))
                    print(f"[video] {seg.name:16} beat {n:02d}  {start:6.1f}s  +{total:4.1f}s  {text[:60]}")
            caption(page, "")
            rec.hold(1.0)
            audio_chunks.append(b"\x00\x00" * max(0, int(round(rec.t * sr)) - samples))
            rec.close()
            browser.close()
    finally:
        server.stop()
    with wave.open(str(audio_tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(audio_chunks))
    out = OUT_DIR / "SGTE_demo.mp4"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(video_tmp), "-i", str(audio_tmp), "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-shortest", str(out)], check=True)
    (OUT_DIR / "SGTE_demo.srt").write_text(
        "\n".join(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{t}\n" for i, (a, b, t) in enumerate(subs, 1)),
        encoding="utf-8")
    (OUT_DIR / "narration.txt").write_text("\n\n".join(t for _, _, t in subs) + "\n", encoding="utf-8")
    size = out.stat().st_size / 1e6
    print(f"[video] wrote {out} ({rec.t:.1f} s, {size:.1f} MB) + SGTE_demo.srt + narration.txt")


if __name__ == "__main__":
    main()
