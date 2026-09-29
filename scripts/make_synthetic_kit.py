"""Generate the SYNTHETIC starter kit in data/ (seeded, reproducible).

    python scripts/make_synthetic_kit.py

Replace the files in data/ with the official kit later; only app/data_loader.py
knows the raw formats.
"""
from __future__ import annotations

import json
import random
import re
import string
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kit_catalog import CATALOG, INTENTIONALLY_MISSING  # noqa: E402
from kit_queries import HELDOUT, HELDOUT_NEGATIVES, QUERIES, VARIATIONS  # noqa: E402
from kit_siis import SIIS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 2026
META = {"synthetic": True, "generator": "scripts/make_synthetic_kit.py", "seed": SEED,
        "note": "Synthetic data written from general One UI knowledge; not official Samsung content."}

PROPER = ["Always On Display", "Dolby Atmos", "RAM Plus", "Quick Share", "Game Booster", "Google Play",
          "Samsung Keyboard", "Wi-Fi", "Bluetooth", "NFC", "Access Point Names", "Private DNS", "Camera",
          "Digital Wellbeing", "Modes and Routines", "Auto HDR", "Mobile Hotspot", "Quick panel", "Settings"]


def _leaf_text(leaf: str) -> str:
    return leaf if any(p in leaf for p in PROPER) else leaf.lower()


def _describe(path: list[str], ctype: str) -> tuple[str, str]:
    leaf, parent = path[-1], (path[-2] if len(path) > 1 else "")
    lt = _leaf_text(leaf)
    if len(path) == 1:
        return f"Open the main {leaf} screen", f"Open {leaf}"
    where = "" if parent == "Settings" else f" under {parent}"
    if ctype == "toggle":
        return f"Turn {lt} on or off" + (f" in {parent}" if where else ""), f"Toggle {lt}"
    if ctype == "slider":
        return f"Adjust {lt}{where}", f"Set the {lt} level"
    noun = lt if lt.endswith("settings") else f"{lt} settings"
    return f"Open {noun}{where}", f"Go to {lt}" + (f" in {parent}" if where else "")


def build_deeplinks() -> list[dict]:
    rng = random.Random(SEED)
    alphabet = string.ascii_lowercase + string.digits
    seen: set[str] = set()
    out = []
    for path_s, ctype, cna, vkey, msg in CATALOG:
        path = [p.strip() for p in path_s.split(">")]
        token = ""
        while not token or token in seen:
            token = "".join(rng.choice(alphabet) for _ in range(6))
        seen.add(token)
        desc, default_msg = _describe(path, ctype)
        validation = None
        if vkey:
            validation = {"key": vkey, "resultType": "integer" if ctype == "slider" else "boolean"}
        out.append({"deeplink": f"bixby://masked/act/{token}", "description": desc,
                    "message": msg or default_msg, "cna_description": cna, "controlType": ctype,
                    "path": path, "validation": validation})
    return out


ROOT_RE = re.compile(r"(?<!> )\b(Settings|Camera|Quick panel|Game Launcher) > ")
STOP_RE = re.compile(r"( and | to | if | when |,|\.(?:\s|$)| screen\b)")


def extract_paths(text: str, known: set[str]) -> list[str]:
    """Greedy: longest menu path starting at each root that is known; else cut at first stop word."""
    paths = []
    for m in ROOT_RE.finditer(text):
        rest = text[m.start():]
        end = re.search(r"\.(?:\s|$)", rest)
        rest = rest[: end.start()] if end else rest
        segs = rest.split(" > ")
        head, last = segs[:-1], segs[-1]
        words = last.split(" ")
        best = None
        for n in range(len(words), 0, -1):
            cand = " > ".join(head + [" ".join(words[:n]).rstrip(",")])
            if cand in known:
                best = cand
                break
        if best is None:
            best = " > ".join(head + [STOP_RE.split(last, maxsplit=1)[0]])
        paths.append(best)
    return paths


def check_paths(deeplinks: list[dict]) -> list[str]:
    known = {" > ".join(d["path"]) for d in deeplinks} | set(INTENTIONALLY_MISSING)
    problems = []
    for sid, _, _, text in SIIS:
        for p in extract_paths(text, known):
            if p not in known:
                problems.append(f"{sid}: '{p}' not in catalog")
    return problems


def build_queries() -> list[dict]:
    out = []
    for i, (q, domain, target, register) in enumerate(QUERIES, 1):
        rec = {"id": f"q{i:03d}", "query": q, "domain": domain, "register": register}
        if isinstance(target, list):
            rec.update(siis_id=target[0], siis_ids=target, type="multi")
        else:
            rec.update(siis_id=target, siis_ids=[target] if target else [], type="single" if target else "no_match")
        out.append(rec)
    return out


# ------------------------------------------------------------------ gold samples
def _nav(path: list[str]) -> list[str]:
    first = {"Settings": "Navigate to and open Settings.", "Camera": "Open the Camera app.",
             "Quick panel": "Swipe down from the top of the screen to open the Quick panel.",
             "Game Launcher": "Open the Game Launcher app."}[path[0]]
    return [first] + [f"Tap on {p}." for p in path[1:]]


def _p(s: str) -> list[str]:
    return [x.strip() for x in s.split(">")]


GOLD = {
    "sample_01": dict(
        query="The mobile phone swipe navigation moves up or down instead of left or right after downloading an app",
        siis="siis_d01", topic="Swipe Navigation", title="Swipe navigation settings", score=0.93,
        actions=[("Configure Navigation Bar Settings", "It will let you choose navigation type", "auto",
                  _nav(_p("Settings > Display > Navigation bar")) + [
                      "Select your preferred navigation type between Buttons and Swipe gestures.",
                      "Optionally toggle on Gesture hint to display guidance lines at the bottom of the screen."],
                  "Settings > Display > Navigation bar")]),
    "sample_02": dict(
        query="My battery is draining really fast even when I'm not using the phone",
        siis="siis_b01", topic="Battery Drain", title="Battery drain fix", score=0.9,
        actions=[
            ("Check Battery Usage", "It will show which apps drain battery", "auto",
             _nav(_p("Settings > Battery > Battery usage")) + ["Check which apps use the most battery."],
             "Settings > Battery > Battery usage"),
            ("Put Unused Apps to Sleep", "It will stop unused apps running", "auto",
             _nav(_p("Settings > Battery > Background usage limits")) + ["Turn on Put unused apps to sleep."],
             "Settings > Battery > Background usage limits > Put unused apps to sleep"),
            ("Turn On Power Saving", "It will extend your battery life", "auto",
             _nav(_p("Settings > Battery > Power saving")) + ["Turn it on to extend battery life."],
             "Settings > Battery > Power saving"),
            ("Lower the Refresh Rate", "It will reduce display power use", "auto",
             _nav(_p("Settings > Display > Motion smoothness")) + ["Select Standard to lower power use."],
             "Settings > Display > Motion smoothness"),
            ("Restart Your Phone", "It will clear temporary system glitches", "critical",
             ["Restart the phone."], None)]),
    "sample_03": dict(
        query="My camera photos come out blurry",
        siis="siis_c01", topic="Blurry Photos", title="Blurry photos fix", score=0.88,
        actions=[
            ("Turn On Scene Optimizer", "It will improve colour and exposure", "auto",
             _nav(_p("Camera > Camera settings > Scene optimizer")) + ["Turn it on."],
             "Camera > Camera settings > Scene optimizer"),
            ("Clean the Camera Lens", "It will remove smudges from lens", "manual",
             ["Clean the camera lens with a soft microfiber cloth."], None),
            ("Remove the Case", "It will uncover the camera lens", "manual",
             ["Remove any case or lens protector that covers the camera."], None),
            ("Tap to Focus", "It will sharpen focus on subject", "manual",
             ["Tap the subject on the preview screen to focus before taking the picture."], None),
            ("Reset Camera Settings", "It will restore default camera settings", "critical",
             _nav(_p("Camera > Camera settings > Reset settings")) + ["Tap Reset if photos are still blurry."],
             "Camera > Camera settings > Reset settings")]),
    "sample_04": dict(
        query="my phone got slow after the update",
        siis="siis_p02", topic="Slow Performance", title="Slow phone fix", score=0.87,
        actions=[
            ("Optimize the Device", "It will close apps and free memory", "auto",
             _nav(_p("Settings > Device care")) + ["Tap Optimize now."], "Settings > Device care"),
            ("Update All Apps", "It will install the latest app fixes", "manual",
             ["Update all apps from the Galaxy Store and Play Store."], None),
            ("Install Software Update", "It will install the newest system software", "critical",
             _nav(_p("Settings > Software update > Download and install"))
             + ["Check for a newer update that fixes known issues."],
             "Settings > Software update > Download and install"),
            ("Restart Your Phone", "It will apply updates and refresh memory", "critical",
             ["Restart the phone after installing updates."], None),
            ("Perform a Factory Data Reset", "It will restore the original phone state", "critical",
             ["Back up your data."] + _nav(_p("Settings > General management > Reset > Factory data reset"))
             + ["Perform a factory data reset."],
             "Settings > General management > Reset > Factory data reset")]),
    "sample_05": dict(query="My Galaxy Watch won't sync my step count with the phone", siis=None),
}


def build_samples(deeplinks: list[dict], siis_by_id: dict) -> dict[str, dict]:
    by_path = {" > ".join(d["path"]): d for d in deeplinks}
    samples = {}
    for name, g in GOLD.items():
        contexts, fallback = [], None
        if g["siis"] is None:
            fallback = "no_match"
        else:
            actions = []
            for an, desc, cat, steps, dl_path in g["actions"]:
                group = {"steps": steps, "validationDeeplink": None, "actionableDeeplink": None}
                if dl_path:
                    d = by_path[dl_path]
                    group["actionableDeeplink"] = {"deeplink": d["deeplink"], "description": d["description"],
                                                   "message": d["message"]}
                actions.append({"actionName": an, "description": desc, "category": cat, "stepGroups": [group]})
            contexts = [{"goal": f"Follow these steps to perform this {g['topic']} Troubleshooting",
                         "title": g["title"], "score": g["score"], "actions": actions}]
        samples[name] = {
            "metadata": META,
            "input": {"query": g["query"], "siis_response": siis_by_id[g["siis"]]["text"] if g["siis"] else None,
                      "siis_id": g["siis"]},
            "output": {"query": g["query"], "query_variations": VARIATIONS[name],
                       "response": {"contexts": contexts},
                       "meta": {"latency_ms": None, "cache_hit": False, "model": "gold", "cost_usd": 0.0,
                                "fallback": fallback}},
        }
    return samples


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    deeplinks = build_deeplinks()
    problems = check_paths(deeplinks)
    if problems:
        print("\n".join(problems))
        raise SystemExit("SIIS text references menu paths missing from the catalog")
    siis = [{"id": i, "topic": t, "title": ti, "text": tx} for i, t, ti, tx in SIIS]
    siis_by_id = {s["id"]: s for s in siis}
    queries = build_queries()
    by_text = {q["query"]: q for q in queries}
    for q in queries:
        for sid in q["siis_ids"]:
            assert sid in siis_by_id, (q["id"], sid)
    heldout = [{"query_id": by_text[q]["id"], "query": q, "siis_id": by_text[q]["siis_id"], "paraphrases": ps}
               for q, ps in HELDOUT.items()]

    _dump(DATA / "deeplinks.json", {"metadata": META, "deeplinks": deeplinks})
    _dump(DATA / "siis_responses.json", {"metadata": META, "responses": siis})
    _dump(DATA / "queries.json", {"metadata": META, "queries": queries})
    _dump(DATA / "paraphrases_heldout.json", {"metadata": {**META, "use": "cache benchmarking only; never pre-warm"},
                                              "items": heldout, "negatives": HELDOUT_NEGATIVES})
    for name, s in build_samples(deeplinks, siis_by_id).items():
        _dump(DATA / "samples" / f"{name}.json", s)
    summarise(deeplinks, siis, queries, heldout)


def summarise(deeplinks, siis, queries, heldout) -> None:
    print(f"deeplinks.json: {len(deeplinks)} entries; control types {dict(Counter(d['controlType'] for d in deeplinks))}; "
          f"top-level areas {dict(Counter(d['path'][1] if len(d['path']) > 1 else d['path'][0] for d in deeplinks).most_common(8))}")
    print(f"siis_responses.json: {len(siis)} entries; topics {dict(Counter(s['topic'] for s in siis))}")
    print(f"queries.json: {len(queries)} queries; domains {dict(Counter(q['domain'] for q in queries))}; "
          f"types {dict(Counter(q['type'] for q in queries))}; registers {dict(Counter(q['register'] for q in queries))}")
    print(f"paraphrases_heldout.json: {len(heldout)} queries, {sum(len(h['paraphrases']) for h in heldout)} paraphrases, "
          f"{len(HELDOUT_NEGATIVES)} negatives")
    print(f"samples/: {len(GOLD)} gold pairs")


if __name__ == "__main__":
    main()
