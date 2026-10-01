"""FastAPI service: POST /v1/troubleshoot, GET /health (truthful: 503 until everything is loaded).

Request (official kit): {"query": "...", "siis_response": {"title": "...", "content": "..."}}. siis_response may
also be a plain string, or omitted (the engine then retrieves from the kit's SIIS articles).
Response: {"query", "query_variations", "response": {"contexts": [...]}, "meta"}; with ?view=contract only
{"query", "response"}, exactly the shape of data/sample_output.json.
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional, Union

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

from app.validate import to_json

log = logging.getLogger("sgte")
STATE: dict = {"engine": None, "error": None, "started": time.time(), "ready_s": None}


def _load() -> None:
    try:
        from app.pipeline import Engine
        from scripts.prewarm import prewarm
        eng = Engine()
        if len(eng.cache) == 0:  # no persisted cache (e.g. fresh volume): pre-warm before reporting healthy
            prewarm(eng)
        STATE["engine"] = eng
        if eng.enrich_mode() == "background":  # upgrade template paraphrases with the local SLM while serving
            threading.Thread(target=lambda: (eng.upgrade_all(), eng.drain()), daemon=True).start()
        STATE["ready_s"] = round(time.time() - STATE["started"], 1)
        log.warning("engine ready in %ss (cache plans=%d, llm=%s)", STATE["ready_s"], len(eng.cache),
                    eng.llm.model_name if eng.llm else "none")
    except Exception as e:  # keep /health truthful
        STATE["error"] = f"{type(e).__name__}: {e}"
        log.exception("engine failed to load")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_load, daemon=True).start()
    yield


app = FastAPI(title="Smart Guided Troubleshooting Engine", version="1.0.0", lifespan=lifespan)


class SiisPayload(BaseModel):
    """The official SIIS payload. Extra fields are ignored."""
    title: str = ""
    content: str = Field("", max_length=200_000)


class TroubleshootRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    siis_response: Optional[Union[SiisPayload, str]] = None

    def payload(self):
        if isinstance(self.siis_response, SiisPayload):
            return {"title": self.siis_response.title, "content": self.siis_response.content}
        return self.siis_response


def _json(obj: dict, status: int = 200) -> Response:
    return Response(content=to_json(obj), media_type="application/json", status_code=status)


@app.get("/health")
def health() -> Response:
    eng = STATE["engine"]
    if eng is None:
        return _json({"status": "error" if STATE["error"] else "loading", "detail": STATE["error"]}, 503)
    return _json({"status": "ok"})


DEMO_PAGE = Path(__file__).resolve().parent / "static" / "demo.html"


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/demo")


@app.get("/demo", include_in_schema=False)
def demo() -> HTMLResponse:
    """Phone-style demo UI over the same /v1/troubleshoot API (used for the demo video)."""
    return HTMLResponse(DEMO_PAGE.read_text(encoding="utf-8"))


@app.get("/v1/info")
def info() -> Response:
    eng = STATE["engine"]
    if eng is None:
        return _json({"status": "loading"}, 503)
    return _json({"catalog_entries": len(eng.kit.catalog), "siis_articles": len(eng.kit.siis),
                  "official_cases": len(eng.kit.cases), "catalog_synthetic": eng.kit.catalog_synthetic,
                  "cache_plans": len(eng.cache), "cache_keys": eng.cache.index.ntotal, "tau": eng.cache.tau,
                  "llm": eng.llm.model_name if eng.llm else "none", "enrich_mode": eng.enrich_mode(),
                  "extract_mode": eng.extract_mode(), "pending_background_upgrades": eng.pending_upgrades,
                  "synthetic_kit": eng.kit.synthetic,
                  "ready_after_s": STATE["ready_s"]})


@app.get("/v1/examples")
def examples() -> Response:
    """The official input.txt complaints with their SIIS payloads (used by the demo page)."""
    eng = STATE["engine"]
    if eng is None:
        return _json({"status": "loading"}, 503)
    return _json({"examples": [{"id": c.id, "query": c.query, "siis_response": c.siis_response}
                               for c in eng.kit.cases]})


@app.post("/v1/troubleshoot")
def troubleshoot(req: TroubleshootRequest, view: Literal["full", "contract"] = Query("full")) -> Response:
    eng = STATE["engine"]
    if eng is None:
        return _json({"error": "engine is still loading", "status": "loading"}, 503)
    record = eng.troubleshoot(req.query, req.payload())
    if not record["meta"]["cache_hit"] and record["meta"]["fallback"] is None:
        threading.Thread(target=eng.cache.save, daemon=True).start()
    if view == "contract":
        return _json({"query": record["query"], "response": record["response"]})
    return _json(record)
