"""All tunables in one place. Every value can be overridden with an environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE per line). Real environment variables always win.
    The same file is read by docker compose, so one place holds local secrets; it is git-ignored."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # paths
    data_dir: Path = Path(_env("SGTE_DATA_DIR", str(ROOT / "data")))
    cache_dir: Path = Path(_env("SGTE_CACHE_DIR", str(ROOT / "cache")))
    log_dir: Path = Path(_env("SGTE_LOG_DIR", str(ROOT / "logs")))

    # models
    embed_model: str = _env("SGTE_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")  # SIIS + deeplinks
    # semantic-cache embedder: bge-small measured ~3x higher paraphrase recall than MiniLM at equal τ
    cache_embed_model: str = _env("SGTE_CACHE_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    rerank_model: str = _env("SGTE_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    torch_threads: int = _i("SGTE_TORCH_THREADS", 4)  # measured fastest on a 12-thread laptop CPU

    # LLM swap layer: "ollama" (default), "anthropic", "openai", or "none" (deterministic only)
    llm_provider: str = _env("SGTE_LLM_PROVIDER", "ollama")
    llm_fallback: str = _env("SGTE_LLM_FALLBACK", "")  # e.g. "anthropic"; used if primary fails
    ollama_url: str = _env("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = _env("SGTE_OLLAMA_MODEL", "qwen2.5:1.5b-instruct")
    anthropic_model: str = _env("SGTE_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    openai_model: str = _env("SGTE_OPENAI_MODEL", "gpt-4o-mini")
    llm_timeout_s: float = _f("SGTE_LLM_TIMEOUT_S", 25.0)
    llm_seed: int = _i("SGTE_LLM_SEED", 42)
    # USD per 1M tokens (input, output); local models cost 0
    price_anthropic: tuple = (_f("SGTE_PRICE_ANTHROPIC_IN", 1.0), _f("SGTE_PRICE_ANTHROPIC_OUT", 5.0))
    price_openai: tuple = (_f("SGTE_PRICE_OPENAI_IN", 0.15), _f("SGTE_PRICE_OPENAI_OUT", 0.60))

    # pipeline modes
    # auto: a fast (hosted) LLM runs inline; a CPU-local SLM (measured 11-17 s per call here) is kept off the
    # request path: extraction uses the deterministic grounded extractor and the SLM writes paraphrases
    # (extra cache keys) in the background.
    extract_mode: str = _env("SGTE_EXTRACT_MODE", "auto")  # auto | pointer | generative | rules
    enrich_mode: str = _env("SGTE_ENRICH_MODE", "auto")  # auto | llm | background | rules

    # semantic cache
    cache_tau: float = _f("SGTE_CACHE_TAU", 0.85)
    # article-anchored tier (zero LLM): on a key miss, a confident dense-only SIIS match whose article already
    # has a validated single-goal plan in the cache serves that plan. min dense / margin over the runner-up
    # article were chosen from a sweep on the held-out paraphrases (see metrics.md).
    anchor_cache: bool = os.getenv("SGTE_ANCHOR_CACHE", "1") not in ("0", "false", "no")
    anchor_min_dense: float = _f("SGTE_ANCHOR_MIN_DENSE", 0.54)
    anchor_margin: float = _f("SGTE_ANCHOR_MARGIN", 0.04)

    # SIIS retrieval (ranked by dense max-sim over article/sentences/title; measured best on the kit)
    # no-match gate: dense >= strong, OR dense >= min AND cross-encoder logit >= min_ce
    siis_top_k: int = _i("SGTE_SIIS_TOP_K", 3)
    siis_dense_strong: float = _f("SGTE_SIIS_DENSE_STRONG", 0.54)
    siis_min_dense: float = _f("SGTE_SIIS_MIN_DENSE", 0.45)
    siis_min_ce: float = _f("SGTE_SIIS_MIN_CE", 0.0)

    # grounding: rapidfuzz token_set_ratio (0-100) a step must reach against the source
    grounding_min: float = _f("SGTE_GROUNDING_MIN", 80.0)

    # deeplink engine
    rrf_k: int = _i("SGTE_RRF_K", 60)
    deeplink_top_k: int = _i("SGTE_DEEPLINK_TOP_K", 10)
    dl_high_score: float = _f("SGTE_DL_HIGH", 2.0)
    dl_high_margin: float = _f("SGTE_DL_HIGH_MARGIN", 0.5)
    dl_medium_score: float = _f("SGTE_DL_MEDIUM", -1.0)
    tie_break_delta: float = _f("SGTE_DL_TIE_DELTA", 0.75)

    # score calibration: score = a*raw + b. Fitted on the 5 gold samples (bench: offset-only fit, MAE 0.062;
    # a least-squares slope came out negative on 8 points, so the slope is kept at 1)
    score_a: float = _f("SGTE_SCORE_A", 1.0)
    score_b: float = _f("SGTE_SCORE_B", -0.037)

    dummy_deeplink: str = "bixby://dummy_positive"
    extra: dict = field(default_factory=dict)


SETTINGS = Settings()
