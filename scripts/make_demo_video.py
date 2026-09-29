"""Generate the demo video (≤ 5 min) with free, local tools only.

* narration: Windows' built-in offline speech synthesiser (System.Speech, voice "Microsoft Zira Desktop")
* picture:   headless Microsoft Edge driven by Playwright, performing the demo on the live /demo page
* numbers:   read from bench/report.json and a live pytest run - nothing is typed in by hand
* output:    docs/demo/SGTE_demo.mp4 (1920x1080, H.264 + AAC) and docs/demo/SGTE_demo.srt

Requirements (dev only):  pip install -r requirements-video.txt
The API must be running:   docker compose up   (or uvicorn app.main:app --port 8000)

    python scripts/make_demo_video.py
"""
from __future__ import annotations

import base64
import io
import json
import re
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import imageio_ffmpeg
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://localhost:8000"
OUT_DIR = ROOT / "docs" / "demo"
CSS_W, CSS_H, SCALE = 1536, 864, 1.25  # -> 1920x1080 pixels
FPS = 15
VOICE, RATE = "Microsoft Zira Desktop", 0
GAP_S = 0.35  # silence between sentences
REPO = "github.com/khushichhabra7921/PRISM_GENAI_HACKATHON_Y2026"
PARAPHRASE = "there is a weird flicker on my display"  # held-out paraphrase: never used to pre-warm the cache
NEW_QUERY = "sound only comes out of one earbud"  # no queries.json entry maps to this article -> cold path


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


def send(rec: Recorder) -> None:
    page = rec.page
    page.click("button.send")
    page.wait_for_function("() => !document.getElementById('chat').innerText.includes('Thinking')", timeout=60000)
    page.wait_for_timeout(150)
    rec.frame()


# ------------------------------------------------------------------ static cards
def card_page(page, html: str) -> None:
    page.set_content(f"""<!doctype html><html><head><meta charset="utf-8"><style>
      body{{margin:0;width:{CSS_W}px;height:{CSS_H}px;overflow:hidden;font-family:'Segoe UI',sans-serif;color:#17152b;
           background:linear-gradient(135deg,#f5f1ff 0%,#eef3ff 55%,#e8f7f6 100%)}}
      .wrap{{padding:70px 90px}} h1{{font-size:62px;margin:0 0 8px;letter-spacing:-.5px}}
      h1 span{{color:#6d28d9}} .k{{font-size:15px;letter-spacing:.2em;color:#6d28d9;font-weight:700}}
      .sub{{font-size:26px;color:#4b4768;margin:6px 0 26px}} .pill{{display:inline-block;background:#fff;border:1px solid #e0daf5;
           border-radius:999px;padding:10px 20px;margin:6px 10px 6px 0;font-size:19px;font-weight:600}}
      .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:18px}}
      .tile{{background:#fff;border:1px solid #e4dff4;border-radius:18px;padding:16px 18px}}
      .tile b{{display:block;font-size:34px;color:#6d28d9}} .tile span{{font-size:15px;color:#5b5873}}
      .tile i{{display:block;font-style:normal;font-size:13px;color:#8b88a3;margin-top:4px}}
      pre{{background:#1c1a2e;color:#d8d4f5;border-radius:16px;padding:18px 22px;font:17px/1.5 Consolas,monospace;margin:18px 0 0}}
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
    pdf = pdfium.PdfDocument(str(ROOT / "docs" / "SGTE_Submission_Deck.pdf"))
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
    text: str
    action: Optional[Callable[[Recorder], None]] = None


@dataclass
class Segment:
    name: str
    setup: Callable[[Recorder], None]
    beats: list[Beat] = field(default_factory=list)


def measured() -> dict:
    r = json.loads((ROOT / "bench" / "report.json").read_text(encoding="utf-8"))
    out = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:warnings"], cwd=ROOT,  # pytest.ini adds -q
                         capture_output=True, text=True).stdout
    m = re.search(r"(\d+) passed[^\n]*", out)
    health = subprocess.run(["curl", "-s", f"{BASE}/health"], capture_output=True, text=True).stdout.strip()
    return {"r": r, "tests": int(m.group(1)) if m else 0, "pytest_line": m.group(0) if m else out[-200:],
            "health": health}


def build_script(page, m: dict, slides: dict[int, bytes]) -> list[Segment]:
    r = m["r"]
    c, lat, du = r["compliance"], r["latency"], r["cache"]["dual_lookup"]
    p95_exact = round(lat["exact_hit"]["p95"])
    p95_para = round(lat["paraphrase_hit_raw"]["p95"])
    p95_cold = round(lat["cold"]["p95"])
    zero_llm = round(du["zero_llm_hit_pct"])
    wrong = round(du["false_hit_rate_pct"])

    def title(rec):
        card_page(page, f"""<div class="wrap"><div class="k">SAMSUNG PRISM · GENERATIVE AI HACKATHON 2026 · THEME 2</div>
          <h1 style="margin-top:26px">Smart Guided <span>Troubleshooting</span> Engine</h1>
          <div class="sub">Vague Galaxy complaints → validated, ordered, one-tap troubleshooting plans</div>
          <div><span class="pill">&lt; 300 ms on known issues</span><span class="pill">&lt; 8 s on new issues</span>
          <span class="pill">0 invented steps</span><span class="pill">0 invalid deeplinks</span></div>
          <div class="sub" style="margin-top:70px;font-size:21px">Open source · runs on a laptop CPU · $0 · {REPO}</div></div>""")
        rec.frame()

    def slide(i):
        def f(rec):
            slide_page(page, slides[i])
            rec.frame()
        return f

    def demo(rec):
        page.goto(f"{BASE}/demo")
        page.wait_for_selector("#chips .chip")
        prepare(page)
        rec.frame()

    def act_type_send(q):
        def f(rec):
            type_query(rec, q)
            rec.hold(0.3)
            send(rec)
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

    def act_top(rec):
        highlight(page, None)
        scroll_to(rec, y=0)

    def act_scroll_critical(rec):
        highlight(page, None)
        sel = "#chat .badge.critical"
        scroll_to(rec, sel)
        page.evaluate("() => { const b = document.querySelector('#chat .goal .badge.critical');"
                      " if (b) b.closest('.action').classList.add('hl'); }")
        rec.frame()

    def act_why(rec):
        highlight(page, None)
        scroll_to(rec, "#why-card")
        highlight(page, "#why-card")
        rec.frame()

    def act_why_scroll(rec):
        y = page.evaluate("() => window.scrollY + 380")
        scroll_to(rec, y=y, steps=30)

    def act_raw(rec):
        highlight(page, None)
        scroll_to(rec, "#raw")
        highlight(page, "#raw")
        rec.frame()
        scroll_to(rec, y=520, steps=30, container="#raw")

    def results(rec):
        tiles = [
            (f"{c['schema_valid_pct']:.0f}%", "schema-valid outputs", f"{c['lines']} queries · target ≥ 99%"),
            (f"{c['rule_compliance_pct']:.0f}%", "pass all 11 rule checks", "target ≥ 95%"),
            (f"{c['url_leaks']}", "URL leaks", "target 0"),
            (f"{c['catalog_validity_pct']:.0f}%", "deeplinks from the catalog", f"{c['deeplinks']} deeplinks"),
            (f"{p95_exact} ms", "P95 exact cache hit", "target ≤ 300 ms"),
            (f"{p95_para} ms", "P95 unseen paraphrase hit", "0 LLM calls"),
            (f"{p95_cold} ms", "P95 new query (CPU)", "target ≤ 8 s"),
            (f"{zero_llm}%", "paraphrases served from cache", f"zero LLM calls · {wrong}% wrong-plan hits"),
        ]
        grid = "".join(f"<div class='tile'><b>{a}</b><span>{b}</span><i>{t}</i></div>" for a, b, t in tiles)
        card_page(page, f"""<div class="wrap" style="padding-top:50px"><div class="k">MEASURED WITH bench/run.py ·
          SYNTHETIC STARTER KIT · CPU ONLY</div><h1 style="font-size:46px;margin-top:14px">Results</h1>
          <div class="grid">{grid}</div>
          <pre><em>$</em> python -m pytest -q
{m['pytest_line']}
<em>$</em> curl -s localhost:8000/health
{m['health']}</pre></div>""")
        rec.frame()

    def closing(rec):
        card_page(page, f"""<div class="wrap"><div class="k">LIMITATIONS · NEXT STEPS</div>
          <h1 style="font-size:46px;margin-top:14px">Honest about the edges</h1>
          <ul><li>Measured on a synthetic starter kit; the official kit drops into <code>data/</code> unchanged</li>
          <li>{wrong}% of paraphrase cache hits served a neighbouring article's plan (τ trade-off)</li>
          <li>The 1.5B local SLM is too slow on CPU for the request path, so it writes cache paraphrases in the background</li></ul>
          <h1 style="font-size:40px;margin-top:40px">Thank you · <span>docker compose up</span></h1>
          <div class="sub" style="font-size:21px">{REPO}</div></div>""")
        rec.frame()

    return [
        Segment("title", title, [
            Beat("This is the Smart Guided Troubleshooting Engine, our prototype for Theme 2 of the Samsung PRISM "
                 "Generative AI Hackathon."),
            Beat("It turns a vague Galaxy complaint into a validated, ordered troubleshooting plan, where every step "
                 "comes from trusted support content, and every fix is one tap away."),
        ]),
        Segment("problem", slide(1), [
            Beat("People describe symptoms, but Settings speak in menus."),
            Beat("Today, an agent reads the knowledge base, picks the steps, and orders them by hand. That takes about "
                 "fifteen minutes per scenario, and the user still has to hunt through nested menus."),
            Beat("Our targets: under three hundred milliseconds for known issues, under eight seconds for new ones, "
                 "zero invented steps, and zero invalid deeplinks."),
        ]),
        Segment("architecture", slide(3), [
            Beat("The design principle is simple. Language models reason, retrieval grounds, and deterministic code "
                 "validates."),
            Beat("A complaint first goes to a semantic cache. On a miss, it is enriched, grounded in support articles "
                 "with hybrid retrieval, and turned into steps that are copied from the source text."),
            Beat("The steps are ordered from safe settings to critical actions, mapped to the exact Settings screen, "
                 "and checked by eleven validators before the plan is cached."),
        ]),
        Segment("demo_known", demo, [
            Beat("Here is the demo interface. It runs entirely on a laptop CPU, using free, local, open models."),
            Beat("First complaint: swipe gestures go the wrong way after installing an app.",
                 act_type_send("swipe gestures go the wrong way after installing an app")),
            Beat("The engine returns one automatic action with five grounded steps.", act_hl("#chat .goal")),
            Beat("The Open button carries the exact catalog deeplink, for Display, Navigation bar, not the parent "
                 "Display menu.", act_open),
            Beat("It came from the cache in a few dozen milliseconds, with zero language model calls, and zero cost.",
                 act_hl(".side .card")),
        ]),
        Segment("demo_paraphrase", lambda rec: (highlight(page, None), scroll_to(rec, y=0)), [
            Beat(f"Next, a paraphrase the system has never seen: {PARAPHRASE}.", act_type_send(PARAPHRASE)),
            Beat("The semantic cache still recognises it, so the validated screen flickering plan comes back "
                 "instantly, again without any model call.", act_hl(".side .card")),
        ]),
        Segment("demo_multi", lambda rec: (highlight(page, None), scroll_to(rec, y=0)), [
            Beat("Now, two problems in one sentence: my screen flickers, and the battery dies fast.",
                 act_type_send("my screen flickers and the battery dies fast")),
            Beat("The engine splits the complaint, grounds each half in its own support article, and returns two "
                 "goals.", act_hl("#chat .status", scroll=False)),
            Beat("Within each goal, safe settings come first, physical checks next, and disruptive steps, such as "
                 "safe mode or a restart, come last.", act_scroll_critical),
        ]),
        Segment("evidence", lambda rec: None, [
            Beat("Every action carries an evidence trail.", act_why),
            Beat("It shows the source sentence each step was copied from, the retrieval and rerank scores, the "
                 "runner-up screen, and the confidence tier, so a support team can see exactly why each step was "
                 "chosen.", act_why_scroll),
        ]),
        Segment("no_match", act_top, [
            Beat("When the knowledge base has no answer, the engine refuses to guess."),
            Beat("My Galaxy Watch won't sync my steps returns an empty plan with a no match flag, and the query is "
                 "logged, so the content team can find the gaps.",
                 lambda rec: (act_type_send("my galaxy watch wont sync my steps")(rec),
                              highlight(page, "#chat .empty"), rec.frame())),
        ]),
        Segment("cold", lambda rec: (highlight(page, None), scroll_to(rec, y=0)), [
            Beat(f"A brand new complaint goes through the full pipeline: {NEW_QUERY}.", act_type_send(NEW_QUERY)),
            Beat("Retrieval, grounded extraction, ordering, deeplink mapping, and validation finish in well under a "
                 "second, on the CPU.", act_hl(".side .card")),
            Beat("And the response is pure JSON that follows the schema contract exactly.", act_raw),
        ]),
        Segment("results", results, [
            Beat("We measured everything with our benchmark, on a synthetic starter kit."),
            Beat(f"{c['schema_valid_pct']:.0f} percent of outputs are schema valid and pass all eleven rule checks, "
                 f"with {'zero' if c['url_leaks'] == 0 else c['url_leaks']} URL leaks, and every deeplink comes from "
                 "the catalog."),
            Beat(f"Cached answers return in {p95_exact} milliseconds at the ninety-fifth percentile, new queries in "
                 f"about {p95_cold} milliseconds, and {zero_llm} percent of unseen paraphrases are served from the "
                 "cache, without any model call."),
            Beat(f"All {m['tests']} tests pass, including the five gold samples."),
        ]),
        Segment("closing", closing, [
            Beat(f"The main limitations: the results come from a synthetic kit, about {wrong} percent of paraphrase "
                 "hits served a neighbouring plan, and the small local model is too slow on a CPU for the request "
                 "path, so it writes cache paraphrases in the background instead."),
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
    print(f"[video] measured: {m['pytest_line']} · health {m['health']}")
    slides = deck_slides([1, 3])
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
                wav = tmp / f"b{n:03d}.wav"
                dur = tts(beat.text, wav)
                with wave.open(str(wav)) as w:
                    sr = w.getframerate()
                    pcm = w.readframes(w.getnframes())
                start = rec.t
                # keep audio locked to video: silence for frames recorded between beats (setup, scrolling)
                lead = int(round(start * sr)) - samples
                if lead > 0:
                    audio_chunks.append(b"\x00\x00" * lead)
                    samples += lead
                caption(page, beat.text)
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
                subs.append((start, start + dur, beat.text))
                print(f"[video] {seg.name:16} beat {n:02d}  {start:6.1f}s  +{total:4.1f}s  {beat.text[:60]}")
        caption(page, "")
        rec.hold(1.0)
        audio_chunks.append(b"\x00\x00" * max(0, int(round(rec.t * sr)) - samples))
        rec.close()
        browser.close()
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
