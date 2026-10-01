"""The ONLY module that knows the raw starter-kit file formats.

Everything else consumes the normalised dataclasses below. Two layouts are understood:

  official kit (data/):  input.txt (one complaint per line), siis_responses.json
                         ({"responses": [{"id", "original_query", "siis_response": {"title", "content"}}]}),
                         sample_output.json, schema.py
  synthetic kit (data/synthetic/): queries.json, siis_responses.json (plain text), paraphrases_heldout.json,
                         samples/*.json

The Settings deeplink catalog (deeplinks.json) is read from the kit folder or, if absent there, its parent
(the official kit ships no catalog; see data/README.md).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from app.config import SETTINGS
from app.siis_text import parse_article, payload_parts

# raw field name -> candidates, first match wins
ALIASES = {
    "deeplink": ["deeplink", "uri", "link", "deepLink"],
    "description": ["description", "desc"],
    "message": ["message", "msg"],
    "cna": ["cna_description", "qna_description", "cnaDescription", "qnaDescription"],
    "control": ["controlType", "control_type", "type"],
    "original_type": ["originalType", "original_type"],
    "path": ["path", "menu_path", "breadcrumb"],
    "validation": ["validation", "validationDeeplink", "validationRule", "validation_rule"],
}


@dataclass(frozen=True)
class CatalogEntry:
    uri: str
    description: str
    message: str
    cna: str
    control_type: str = "screen"
    path: tuple[str, ...] = ()
    validation: Optional[dict] = None  # {"deeplink": voiceassist://masked/val/..., "key", "resultType", ...}
    original_type: str = ""

    @property
    def text(self) -> str:
        """Metadata text used for matching. Never includes the masked URI."""
        return " . ".join(t for t in (self.description, self.message, self.cna) if t)

    @property
    def depth(self) -> int:
        return len(self.path) if self.path else 1

    @property
    def leaf(self) -> str:
        return self.path[-1] if self.path else ""

    @property
    def val_uri(self) -> Optional[str]:
        return (self.validation or {}).get("deeplink") or None


@dataclass(frozen=True)
class SiisDoc:
    id: str
    topic: str
    title: str
    text: str  # clean flat text: what retrieval indexes and what grounding checks use
    raw: str = ""  # the payload content as received (headings, one step per line); parsed for extraction

    @property
    def source(self) -> str:
        return self.raw or self.text


@dataclass(frozen=True)
class QueryRec:
    id: str
    query: str
    domain: str
    siis_ids: tuple[str, ...]
    type: str  # single | multi | no_match


@dataclass(frozen=True)
class Case:
    """One official evaluation row: the complaint and the SIIS payload the API receives with it."""
    id: str
    query: str
    siis_response: dict
    siis_id: str


@dataclass
class Kit:
    catalog: list[CatalogEntry]
    siis: list[SiisDoc]
    queries: list[QueryRec]
    heldout: list[dict] = field(default_factory=list)
    negatives: list[str] = field(default_factory=list)
    samples: dict[str, dict] = field(default_factory=dict)
    synthetic: bool = True
    cases: list[Case] = field(default_factory=list)
    reference_outputs: list[dict] = field(default_factory=list)  # official sample_output.json
    official: bool = False
    catalog_synthetic: bool = True


def _pick(rec: dict, key: str, default: Any = "") -> Any:
    for k in ALIASES[key]:
        if k in rec and rec[k] is not None:
            return rec[k]
    return default


def _records(obj: Any, *keys: str) -> tuple[list, dict]:
    """Accept a raw list or a {"metadata":…, "<key>": [...]} wrapper."""
    if isinstance(obj, list):
        return obj, {}
    for k in keys:
        if k in obj:
            return obj[k], obj.get("metadata", {})
    if all(isinstance(v, (str, dict)) for v in obj.values()):  # {id: text} style
        return [{"id": k, **(v if isinstance(v, dict) else {"text": v})} for k, v in obj.items()], {}
    raise ValueError(f"unrecognised file layout; expected one of {keys}")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _catalog_file(data_dir: Path) -> Path:
    for d in (data_dir, data_dir.parent):
        if (d / "deeplinks.json").exists():
            return d / "deeplinks.json"
    raise FileNotFoundError(f"deeplinks.json not found in {data_dir} or its parent")


def load_catalog(data_dir: Path) -> tuple[list[CatalogEntry], dict]:
    rows, meta = _records(_read(_catalog_file(data_dir)), "deeplinks", "entries", "data")
    out = []
    for r in rows:
        path = _pick(r, "path", ())
        if isinstance(path, str):
            path = [p.strip() for p in path.replace("›", ">").split(">")]
        out.append(CatalogEntry(uri=_pick(r, "deeplink"), description=_pick(r, "description"),
                                message=_pick(r, "message") or "", cna=_pick(r, "cna") or "",
                                control_type=str(_pick(r, "control", "screen")).lower(),
                                path=tuple(path), validation=_pick(r, "validation", None),
                                original_type=str(_pick(r, "original_type", "") or "")))
    return [e for e in out if e.uri], meta


def load_siis(data_dir: Path) -> tuple[list[SiisDoc], dict[str, str]]:
    """-> (unique articles, {row id: article id}). Official rows repeat the same article for several
    complaints; each distinct content becomes one SiisDoc, identified by the first row that carried it."""
    rows, _ = _records(_read(data_dir / "siis_responses.json"), "responses", "entries", "data")
    docs: list[SiisDoc] = []
    by_content: dict[str, str] = {}
    row_doc: dict[str, str] = {}
    for i, r in enumerate(rows):
        rid = str(r.get("id", f"siis_{i:03d}"))
        payload = r.get("siis_response")
        if isinstance(payload, dict):  # official: {"title", "content"}
            title, content = payload_parts(payload)
            article = parse_article(title, content)
            text, raw = article.text, content
        else:
            text = r.get("text") or payload or r.get("response") or r.get("content") or ""
            title, raw = r.get("title") or r.get("query") or text.split(".")[0][:60], ""
        key = re.sub(r"\s+", " ", raw or text).strip()
        if key in by_content:
            row_doc[rid] = by_content[key]
            continue
        by_content[key] = row_doc[rid] = rid
        docs.append(SiisDoc(id=rid, topic=r.get("topic", r.get("domain", "")), title=title, text=text, raw=raw))
    return docs, row_doc


def _clean_line(line: str) -> str:
    return line.strip().strip("﻿")


def load_cases(data_dir: Path, row_doc: dict[str, str]) -> list[Case]:
    """Pair every complaint in input.txt with the SIIS payload of the same row (the files are in the same order)."""
    p = data_dir / "input.txt"
    if not p.exists():
        return []
    lines = [_clean_line(x) for x in p.read_text(encoding="utf-8-sig").splitlines() if _clean_line(x)]
    rows, _ = _records(_read(data_dir / "siis_responses.json"), "responses", "entries", "data")
    rows = [r for r in rows if isinstance(r.get("siis_response"), dict)]
    if len(rows) != len(lines):
        raise ValueError(f"input.txt has {len(lines)} complaints but siis_responses.json has {len(rows)} payloads")
    return [Case(id=str(r["id"]), query=q, siis_response=r["siis_response"], siis_id=row_doc[str(r["id"])])
            for q, r in zip(lines, rows)]


def load_queries(data_dir: Path, cases: list[Case]) -> list[QueryRec]:
    p = data_dir / "queries.json"
    if not p.exists():  # official kit: the labelled queries are the input.txt complaints
        return [QueryRec(id=c.id, query=c.query, domain="", siis_ids=(c.siis_id,), type="single") for c in cases]
    rows, _ = _records(_read(p), "queries", "entries", "data")
    out = []
    for i, r in enumerate(rows):
        if isinstance(r, str):
            r = {"query": r}
        ids = r.get("siis_ids") or ([r["siis_id"]] if r.get("siis_id") else [])
        qtype = r.get("type") or ("no_match" if not ids else "multi" if len(ids) > 1 else "single")
        out.append(QueryRec(id=str(r.get("id", f"q{i:03d}")), query=r["query"], domain=r.get("domain", ""),
                            siis_ids=tuple(ids), type=qtype))
    return out


def load_samples(data_dir: Path) -> dict[str, dict]:
    sdir = data_dir / "samples"
    if not sdir.exists():
        return {}
    return {p.stem: _read(p) for p in sorted(sdir.glob("*.json"))}


def load_reference_outputs(data_dir: Path) -> list[dict]:
    p = data_dir / "sample_output.json"
    if not p.exists():
        return []
    obj = _read(p)
    return obj if isinstance(obj, list) else [obj]


def load_heldout(data_dir: Path) -> tuple[list[dict], list[str]]:
    p = data_dir / "paraphrases_heldout.json"
    if not p.exists():
        return [], []
    obj = _read(p)
    return obj.get("items", []), obj.get("negatives", [])


@lru_cache(maxsize=4)
def load_kit(data_dir: Optional[str] = None) -> Kit:
    d = Path(data_dir) if data_dir else SETTINGS.data_dir
    catalog, meta = load_catalog(d)
    heldout, negatives = load_heldout(d)
    siis, row_doc = load_siis(d)
    cases = load_cases(d, row_doc)
    official = bool(cases)
    return Kit(catalog=catalog, siis=siis, queries=load_queries(d, cases), heldout=heldout, negatives=negatives,
               samples=load_samples(d), synthetic=not official, cases=cases,
               reference_outputs=load_reference_outputs(d), official=official,
               catalog_synthetic=bool(meta.get("synthetic", False)))


SYNTHETIC_DIR = SETTINGS.data_dir / "synthetic"


def load_synthetic_kit() -> Kit:
    """The seeded regression kit (labelled queries, held-out paraphrases, gold samples)."""
    return load_kit(str(SYNTHETIC_DIR))
