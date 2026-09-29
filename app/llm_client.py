"""One LLM swap layer: local Ollama by default, hosted Anthropic/OpenAI via env vars.

temperature 0, fixed seed where supported, JSON-constrained output, timeouts, one retry,
token-based cost per call. If nothing is reachable, callers fall back to deterministic code.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from app.config import SETTINGS, Settings

SYSTEM = ("You are a careful assistant inside a Samsung Galaxy troubleshooting engine. "
          "Reply with one JSON object only: no markdown, no code fences, no commentary, no URLs.")


@dataclass
class LLMResult:
    ok: bool
    data: Optional[dict] = None
    model: str = ""
    provider: str = ""
    latency_ms: float = 0.0
    in_tokens: int = 0
    out_tokens: int = 0
    cost_usd: float = 0.0
    error: str = ""


@dataclass
class Usage:
    """Accumulates calls for one request -> meta.model / meta.cost_usd."""
    calls: list[LLMResult] = field(default_factory=list)

    def add(self, r: LLMResult) -> None:
        self.calls.append(r)

    @property
    def cost_usd(self) -> float:
        return round(sum(c.cost_usd for c in self.calls), 6)

    @property
    def models(self) -> list[str]:
        return sorted({f"{c.provider}/{c.model}" for c in self.calls if c.ok})

    def summary(self) -> dict:
        return {"llm_calls": len(self.calls), "llm_ok": sum(c.ok for c in self.calls),
                "llm_latency_ms": round(sum(c.latency_ms for c in self.calls), 1),
                "tokens_in": sum(c.in_tokens for c in self.calls), "tokens_out": sum(c.out_tokens for c in self.calls)}


def _raise_for_status(r: httpx.Response) -> None:
    if r.status_code >= 400:
        try:
            err = r.json().get("error", {})
            code = err.get("code") or err.get("type") or ""
            kind = err.get("type") or ""
        except Exception:
            code, kind = "", ""
        raise RuntimeError(f"HTTP {r.status_code} {kind} {code}".strip())


def parse_json(text: str) -> Optional[dict]:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
    return None


class _Provider:
    name = ""

    def __init__(self, s: Settings):
        self.s = s
        self._up: Optional[tuple[float, bool]] = None

    @property
    def model(self) -> str:
        raise NotImplementedError

    def available(self) -> bool:
        raise NotImplementedError

    def call(self, prompt: str, schema: dict, system: str, max_tokens: int) -> LLMResult:
        raise NotImplementedError


class OllamaProvider(_Provider):
    name = "ollama"

    @property
    def model(self) -> str:
        return self.s.ollama_model

    def available(self) -> bool:
        now = time.time()
        if self._up and now - self._up[0] < 30:
            return self._up[1]
        try:
            r = httpx.get(f"{self.s.ollama_url}/api/tags", timeout=1.5)
            names = {m.get("name", "") for m in r.json().get("models", [])}
            ok = r.status_code == 200 and any(n.split(":")[0] == self.model.split(":")[0] and
                                              (n == self.model or n.startswith(self.model)) for n in names)
        except Exception:
            ok = False
        self._up = (now, ok)
        return ok

    def call(self, prompt, schema, system, max_tokens, num_ctx=4096):
        body = {"model": self.model, "stream": False, "format": schema, "keep_alive": "60m",
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                "options": {"temperature": 0, "seed": self.s.llm_seed, "num_predict": max_tokens, "num_ctx": num_ctx}}
        t = time.perf_counter()
        r = httpx.post(f"{self.s.ollama_url}/api/chat", json=body, timeout=self.s.llm_timeout_s)
        _raise_for_status(r)
        j = r.json()
        return LLMResult(ok=True, data=parse_json(j["message"]["content"]), model=self.model, provider=self.name,
                         latency_ms=(time.perf_counter() - t) * 1000, in_tokens=j.get("prompt_eval_count", 0),
                         out_tokens=j.get("eval_count", 0), cost_usd=0.0)


class AnthropicProvider(_Provider):
    name = "anthropic"

    @property
    def model(self) -> str:
        return self.s.anthropic_model

    def available(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY"))

    def call(self, prompt, schema, system, max_tokens, num_ctx=None):
        full = f"{prompt}\n\nJSON schema to follow:\n{json.dumps(schema)}"
        body = {"model": self.model, "max_tokens": max_tokens, "temperature": 0, "system": system,
                "messages": [{"role": "user", "content": full}]}
        headers = {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        t = time.perf_counter()
        r = httpx.post("https://api.anthropic.com/v1/messages", json=body, headers=headers,
                       timeout=self.s.llm_timeout_s)
        _raise_for_status(r)
        j = r.json()
        text = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
        u = j.get("usage", {})
        pin, pout = self.s.price_anthropic
        return LLMResult(ok=True, data=parse_json(text), model=self.model, provider=self.name,
                         latency_ms=(time.perf_counter() - t) * 1000, in_tokens=u.get("input_tokens", 0),
                         out_tokens=u.get("output_tokens", 0),
                         cost_usd=(u.get("input_tokens", 0) * pin + u.get("output_tokens", 0) * pout) / 1e6)


class OpenAIProvider(_Provider):
    name = "openai"

    @property
    def model(self) -> str:
        return self.s.openai_model

    def available(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

    def call(self, prompt, schema, system, max_tokens, num_ctx=None):
        body = {"model": self.model, "temperature": 0, "seed": self.s.llm_seed, "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": f"{prompt}\n\nJSON schema:\n{json.dumps(schema)}"}]}
        headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
        t = time.perf_counter()
        r = httpx.post("https://api.openai.com/v1/chat/completions", json=body, headers=headers,
                       timeout=self.s.llm_timeout_s)
        _raise_for_status(r)
        j = r.json()
        u = j.get("usage", {})
        pin, pout = self.s.price_openai
        return LLMResult(ok=True, data=parse_json(j["choices"][0]["message"]["content"]), model=self.model,
                         provider=self.name, latency_ms=(time.perf_counter() - t) * 1000,
                         in_tokens=u.get("prompt_tokens", 0), out_tokens=u.get("completion_tokens", 0),
                         cost_usd=(u.get("prompt_tokens", 0) * pin + u.get("completion_tokens", 0) * pout) / 1e6)


PROVIDERS = {"ollama": OllamaProvider, "anthropic": AnthropicProvider, "openai": OpenAIProvider}


class LLMClient:
    def __init__(self, s: Settings = SETTINGS, provider: Optional[str] = None):
        names = [provider or s.llm_provider] + ([s.llm_fallback] if s.llm_fallback and not provider else [])
        self.chain = [PROVIDERS[n](s) for n in names if n in PROVIDERS]

    def available(self) -> bool:
        return any(p.available() for p in self.chain)

    @property
    def model_name(self) -> str:
        for p in self.chain:
            if p.available():
                return f"{p.name}/{p.model}"
        return "none"

    def complete_json(self, prompt: str, schema: dict[str, Any], system: str = SYSTEM,
                      max_tokens: int = 512, num_ctx: int = 4096) -> LLMResult:
        last = LLMResult(ok=False, error="no LLM provider reachable")
        for p in self.chain:
            if not p.available():
                continue
            for _attempt in range(2):  # one retry
                try:
                    res = p.call(prompt, schema, system, max_tokens, num_ctx)
                    if res.data is not None:
                        return res
                    last = LLMResult(ok=False, model=p.model, provider=p.name, latency_ms=res.latency_ms,
                                     in_tokens=res.in_tokens, out_tokens=res.out_tokens, cost_usd=res.cost_usd,
                                     error="unparseable JSON")
                except Exception as e:  # timeout, HTTP error, connection refused
                    last = LLMResult(ok=False, model=p.model, provider=p.name, error=f"{type(e).__name__}: {e}"[:200])
        return last
