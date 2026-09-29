"""The ONLY module that knows the raw starter-kit file formats.

Everything else consumes the normalised dataclasses below. To use the official kit,
drop its files into data/ and adjust the field aliases here if names differ.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from app.config import SETTINGS

# raw field name -> candidates, first match wins
ALIASES = {
    "deeplink": ["deeplink", "uri", "link", "deepLink"],
    "description": ["description", "desc"],
    "message": ["message", "msg"],
    "cna": ["cna_description", "qna_description", "cnaDescription", "qnaDescription"],
    "control": ["controlType", "control_type", "type", "originalType"],
    "path": ["path", "menu_path", "breadcrumb"],
    "validation": ["validation", "validationRule", "validation_rule"],
}


@dataclass(frozen=True)
class CatalogEntry:
    uri: str
    description: str
    message: str
    cna: str
    control_type: str = "screen"
    path: tuple[str, ...] = ()
    validation: Optional[dict] = None

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


@dataclass(frozen=True)
class SiisDoc:
    id: str
    topic: str
    title: str
    text: str


@dataclass(frozen=True)
class QueryRec:
    id: str
    query: str
    domain: str
    siis_ids: tuple[str, ...]
    type: str  # single | multi | no_match


@dataclass
class Kit:
    catalog: list[CatalogEntry]
    siis: list[SiisDoc]
    queries: list[QueryRec]
    heldout: list[dict] = field(default_factory=list)
    negatives: list[str] = field(default_factory=list)
    samples: dict[str, dict] = field(default_factory=dict)
    synthetic: bool = True


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


def load_catalog(data_dir: Path) -> tuple[list[CatalogEntry], dict]:
    rows, meta = _records(_read(data_dir / "deeplinks.json"), "deeplinks", "entries", "data")
    out = []
    for r in rows:
        path = _pick(r, "path", ())
        if isinstance(path, str):
            path = [p.strip() for p in path.replace("›", ">").split(">")]
        out.append(CatalogEntry(uri=_pick(r, "deeplink"), description=_pick(r, "description"),
                                message=_pick(r, "message") or "", cna=_pick(r, "cna") or "",
                                control_type=str(_pick(r, "control", "screen")).lower(),
                                path=tuple(path), validation=_pick(r, "validation", None)))
    return [e for e in out if e.uri], meta


def load_siis(data_dir: Path) -> list[SiisDoc]:
    rows, _ = _records(_read(data_dir / "siis_responses.json"), "responses", "entries", "data")
    out = []
    for i, r in enumerate(rows):
        text = r.get("text") or r.get("siis_response") or r.get("response") or r.get("content") or ""
        title = r.get("title") or r.get("query") or text.split(".")[0][:60]
        out.append(SiisDoc(id=str(r.get("id", f"siis_{i:03d}")), topic=r.get("topic", r.get("domain", "")),
                           title=title, text=text))
    return out


def load_queries(data_dir: Path) -> list[QueryRec]:
    rows, _ = _records(_read(data_dir / "queries.json"), "queries", "entries", "data")
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
    return Kit(catalog=catalog, siis=load_siis(d), queries=load_queries(d), heldout=heldout,
               negatives=negatives, samples=load_samples(d), synthetic=bool(meta.get("synthetic", False)))
