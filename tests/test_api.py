"""API contract: truthful /health and pure-JSON responses (engine stubbed; no models loaded)."""
import json

from fastapi.testclient import TestClient

from app import main


class FakeCache:
    def save(self):
        pass


class FakeEngine:
    cache = FakeCache()

    def troubleshoot(self, query, siis_response=None):
        return {"query": query, "query_variations": [], "response": {"contexts": []},
                "meta": {"latency_ms": 1.0, "cache_hit": True, "model": "x", "cost_usd": 0.0, "fallback": None}}


def test_health_is_503_until_engine_loaded():
    client = TestClient(main.app)  # no context manager -> lifespan (model loading) not started
    main.STATE["engine"] = None
    r = client.get("/health")
    assert r.status_code == 503 and r.json()["status"] == "loading"
    r = client.post("/v1/troubleshoot", json={"query": "battery drains"})
    assert r.status_code == 503


def test_health_ok_and_pure_json_when_ready():
    client = TestClient(main.app)
    main.STATE["engine"] = FakeEngine()
    try:
        assert client.get("/health").json() == {"status": "ok"}
        r = client.post("/v1/troubleshoot", json={"query": "battery drains", "siis_response": "text"})
        assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
        assert not r.text.lstrip().startswith("```") and r.text.lstrip().startswith("{")
        body = json.loads(r.text)
        assert set(body) >= {"query", "query_variations", "response", "meta"}
    finally:
        main.STATE["engine"] = None


def test_rejects_empty_query():
    client = TestClient(main.app)
    main.STATE["engine"] = FakeEngine()
    try:
        assert client.post("/v1/troubleshoot", json={"query": ""}).status_code == 422
    finally:
        main.STATE["engine"] = None


def test_demo_page_served():
    client = TestClient(main.app)
    r = client.get("/demo")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "/v1/troubleshoot" in r.text


def test_official_payload_object_and_contract_view():
    """The official request carries siis_response as {"title", "content"}; ?view=contract returns exactly
    {"query", "response"} like data/sample_output.json."""
    seen = {}

    class Recorder(FakeEngine):
        def troubleshoot(self, query, siis_response=None):
            seen["payload"] = siis_response
            return super().troubleshoot(query, siis_response)

    client = TestClient(main.app)
    main.STATE["engine"] = Recorder()
    try:
        body = {"query": "screen is black", "siis_response": {"title": "Blank display", "content": "## Step 1: X\nTap Y."}}
        r = client.post("/v1/troubleshoot?view=contract", json=body)
        assert r.status_code == 200 and list(r.json()) == ["query", "response"]
        assert seen["payload"] == body["siis_response"]
        r = client.post("/v1/troubleshoot", json={"query": "q", "siis_response": "plain text still works"})
        assert r.status_code == 200 and seen["payload"] == "plain text still works"
        assert client.post("/v1/troubleshoot?view=bogus", json={"query": "q"}).status_code == 422
    finally:
        main.STATE["engine"] = None
