"""Lazy singletons for the embedding model and the cross-encoder reranker (CPU)."""
from __future__ import annotations

import os
import re
import threading
from functools import lru_cache

import numpy as np

from app.config import SETTINGS

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_lock = threading.Lock()

TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
BM25_STOP = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "at", "by", "is", "it", "my",
             "your", "you", "i", "me", "this", "that", "be", "are", "was", "from", "as", "so", "its", "phone",
             "tap", "open", "navigate", "settings", "turn", "select", "go"}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in BM25_STOP]


def _threads() -> None:
    import torch
    torch.set_num_threads(SETTINGS.torch_threads)


@lru_cache(maxsize=1)
def embedder():
    _threads()
    from sentence_transformers import SentenceTransformer
    with _lock:
        return SentenceTransformer(SETTINGS.embed_model, device="cpu")


@lru_cache(maxsize=1)
def cache_embedder():
    _threads()
    from sentence_transformers import SentenceTransformer
    if SETTINGS.cache_embed_model == SETTINGS.embed_model:
        return embedder()
    with _lock:
        return SentenceTransformer(SETTINGS.cache_embed_model, device="cpu")


@lru_cache(maxsize=1)
def reranker():
    _threads()
    from sentence_transformers import CrossEncoder
    with _lock:
        return CrossEncoder(SETTINGS.rerank_model, device="cpu")


def embed(texts: list[str]) -> np.ndarray:
    vecs = embedder().encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False,
                             convert_to_numpy=True)
    return vecs.astype("float32")


def embed_cache(texts: list[str]) -> np.ndarray:
    """Embeddings used only for the semantic cache keys / lookups."""
    vecs = cache_embedder().encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False,
                                   convert_to_numpy=True)
    return vecs.astype("float32")


def rerank(query: str, docs: list[str]) -> np.ndarray:
    if not docs:
        return np.zeros(0, dtype="float32")
    return np.asarray(reranker().predict([(query, d) for d in docs], show_progress_bar=False), dtype="float32")


def rerank_pairs(pairs: list[tuple[str, str]]) -> np.ndarray:
    if not pairs:
        return np.zeros(0, dtype="float32")
    return np.asarray(reranker().predict(pairs, batch_size=64, show_progress_bar=False), dtype="float32")


def warm() -> None:
    embed(["warm up"])
    embed_cache(["warm up"])
    rerank("warm up", ["warm up"])
