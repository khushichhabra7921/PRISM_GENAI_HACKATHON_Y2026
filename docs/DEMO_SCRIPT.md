# Demo video script (≤ 5 minutes)

The submitted video, [demo/SGTE_demo.mp4](demo/SGTE_demo.mp4), was generated automatically by `scripts/make_demo_video.py` (narration text: [demo/narration.txt](demo/narration.txt)). The run of show below is for recording a live version by hand.

Everything here is free: the prototype runs locally, and the video can be recorded with tools built into
Windows 11 and uploaded to YouTube (as *Unlisted*) or Google Drive (*Anyone with the link can view*).

## Before recording

1. Start the stack and wait until it is healthy:
   ```bash
   docker compose up --build
   ```
   ```bash
   curl -s localhost:8000/health
   ```
   (Without Docker: `.venv/Scripts/python -m uvicorn app.main:app --port 8000`.)
2. Open `http://localhost:8000/demo` in a browser, full screen (F11), zoom about 110%.
3. Open a terminal next to it (for the `curl`, `pytest` and `metrics.md` moments).
4. Record with **Snipping Tool → Record** (Win + Shift + R) or **Xbox Game Bar** (Win + Alt + R).
   Trim in **Clipchamp** (free, preinstalled on Windows 11) if needed.

## Run of show

| Time | Show | Say (key points) |
|---|---|---|
| 0:00–0:30 | Deck slide 2 (problem) | Vague complaints → ~15 min manual triage. Goal: validated, ordered, one-tap plan; < 300 ms for known issues, < 8 s for new ones. |
| 0:30–1:15 | Deck slide 4 (architecture) | LLMs reason, retrieval grounds, deterministic code validates. Cache tiers, SIIS RAG, grounded extraction, hybrid deeplink engine, 11 validators. |
| 1:15–1:50 | Demo page: chip *"swipe gestures go the wrong way after installing an app"* | One AUTO action, five grounded steps, **Open ›** reveals the exact catalog deeplink (Display › Navigation bar). Metadata card: cache hit, ~20–50 ms, 0 LLM calls, $0.00. |
| 1:50–2:20 | Chip *"there is a weird flicker on my display"* (an unseen paraphrase) | Served from the semantic cache without any LLM call: the "lookup" line shows the key or article-anchored tier. |
| 2:20–2:50 | Chip *"my screen flickers and the battery dies fast"* | Multi-symptom split: two goals (screen flickering, battery drain), each grounded in its own article; auto → manual → critical ordering. |
| 2:50–3:20 | Click **Why these steps?** | Evidence trail: source sentence per action, BM25 / dense / rerank scores, runner-up and margin, confidence tier. |
| 3:20–3:45 | Chip *"my galaxy watch wont sync my steps"* | No SIIS coverage → `contexts: []` + `fallback: no_match`; nothing invented; logged for the content team (`bench/cluster_nomatch.py`). |
| 3:45–4:10 | Type a new complaint, e.g. *"sound only comes out of one earbud"*, then show the raw JSON panel | New query through the full pipeline in well under a second on CPU; response is pure JSON in the exact `schema.py` contract. |
| 4:10–4:40 | Terminal: `.venv/Scripts/python -m pytest -q`, then scroll `metrics.md` | 34 tests pass (all 5 gold samples); 100% schema / rule compliance, 0 URL leaks, 100% catalog-valid deeplinks, P95 23 ms cached / 429 ms cold, 84% paraphrase hits with zero LLM calls. |
| 4:40–5:00 | Deck slide 8 (limitations) | Synthetic kit, CPU-only local SLM kept off the request path, τ trade-off; next steps. |

## After recording

1. Upload to YouTube (visibility **Unlisted**) or Google Drive (share: *Anyone with the link*).
2. Put the link in `README.md` (the "Demo video" row of the Submission table) and in the deck (slide 5).
3. Commit, then move the release tag to that final commit (see the README's *Release* section).
