"""SIIS payload -> clean, sectioned article (the grounding boundary).

The official kit sends `siis_response` as {"title": ..., "content": ...}. The content starts with a
product-category prefix ("Smartphone,Tablet,... <title> ( Smartphone,Tablet,...): ") followed by markdown-ish
text: `#`/`##`/`###` headings ("## Step 2: Verify Your Phone's Internet Connection"), one paragraph or one
step per line, glossaries, notes, and occasionally garbled lines with the spaces stripped out. This module
only removes and splits text; it never adds words.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional

from app.textutils import scrub_urls, split_sentences

HEADING_RE = re.compile(r"^\s*(#{1,6})\s*(.*?)\s*#*\s*$")
STEP_PREFIX_RE = re.compile(r"^(?:step\s*\d+\s*[:.\-–]\s*|\d+\s*[.)]\s+)", re.IGNORECASE)
GLUED_RE = re.compile(r"(?<=[a-z0-9)\"'])\.(?=[A-Z][a-z])")  # "apps.To open" -> "apps. To open"
SKIP_SECTIONS = re.compile(r"^(glossary|useful app pairing ideas)$", re.IGNORECASE)
SKIP_LINES = re.compile(r"^(?:note|glossary|important notes?:?|\"\"|''|[-*#\s]*)$", re.IGNORECASE)
MAX_TOKEN = 24  # a whitespace token longer than this is a line whose spaces were lost: unusable

# a titled section with at most this many paragraphs / characters is scored for relevance as one block
SECTION_BLOCK_PARAS = 6
SECTION_BLOCK_CHARS = 1500


@dataclass
class Section:
    heading: str
    level: int
    paragraphs: list[str] = field(default_factory=list)
    numbered: bool = False  # "## Step 2: ..." / "### 3. ..." -> part of a step-by-step procedure


@dataclass
class Block:
    """A unit of relevance: a short section, or one paragraph (step lists and intro lines merged)."""
    section: int
    heading: str
    paragraphs: list[str]

    @property
    def text(self) -> str:
        return " ".join(self.paragraphs)

    @property
    def scoring_text(self) -> str:
        return f"{self.heading}. {self.text}" if self.heading else self.text


@dataclass
class Article:
    title: str
    sections: list[Section]
    raw: str

    @property
    def text(self) -> str:
        """Flat clean text (headings as their own sentences) used for retrieval and grounding checks."""
        parts = []
        for s in self.sections:
            if s.heading:
                parts.append(s.heading.rstrip(".:") + ".")
            parts.extend(s.paragraphs)
        return " ".join(parts)

    @property
    def headed(self) -> bool:
        return any(s.heading for s in self.sections)

    @property
    def procedural(self) -> bool:
        """A numbered troubleshooting procedure (Step 1..N): every step belongs to the answer."""
        return sum(s.numbered for s in self.sections if s.paragraphs) >= 2

    @property
    def digest(self) -> str:
        return content_hash(self.text)

    def blocks(self) -> list[Block]:
        out: list[Block] = []
        for si, s in enumerate(self.sections):
            if not s.paragraphs:
                continue
            short = len(s.paragraphs) <= SECTION_BLOCK_PARAS and sum(map(len, s.paragraphs)) <= SECTION_BLOCK_CHARS
            if s.heading and short:
                out.append(Block(si, s.heading, list(s.paragraphs)))
                continue
            cur: Optional[Block] = None
            for p in s.paragraphs:
                single = len(split_sentences(p)) == 1 and len(p.split()) <= 25
                if cur and (cur.paragraphs[-1].endswith(":") or
                            (single and len(split_sentences(cur.paragraphs[-1])) == 1)):
                    cur.paragraphs.append(p)
                else:
                    cur = Block(si, s.heading, [p])
                    out.append(cur)
        return out


def content_hash(text: str) -> str:
    return hashlib.sha1(re.sub(r"\s+", " ", text or "").strip().lower().encode("utf-8")).hexdigest()[:12]


def payload_parts(payload: Any) -> tuple[str, str]:
    """Accept the official {"title", "content"} object, a legacy plain string, or None."""
    if payload is None:
        return "", ""
    if isinstance(payload, str):
        return "", payload
    if isinstance(payload, dict):
        title = str(payload.get("title") or "")
        content = payload.get("content")
        if content is None:
            content = payload.get("text") or payload.get("siis_response") or ""
        if isinstance(content, dict):  # a whole {"id", "original_query", "siis_response": {...}} row
            return payload_parts(content)
        return title, str(content)
    return "", str(payload)


def strip_prefix(title: str, content: str) -> str:
    """Drop the '<categories> <title> ( <categories>): ' header that precedes every official article."""
    head = content[:1200]
    if title:
        i = head.find(f"{title} (")
        if i >= 0:
            j = head.find("):", i)
            if j >= 0:
                return content[j + 2:].lstrip()
    m = re.match(r"^[^\n]{0,800}?\([^()\n]{0,600}\):\s*", content)
    if m and "," in m.group(0):
        return content[m.end():]
    return content


def _clean_heading(h: str) -> str:
    h = STEP_PREFIX_RE.sub("", h.strip()).strip(" :#")
    return scrub_urls(h)


def _clean_line(line: str) -> str:
    line = scrub_urls(line.replace(" ", " ").strip())
    line = GLUED_RE.sub(". ", line)
    line = re.sub(r"\s+", " ", line).strip()
    if not line or SKIP_LINES.match(line):
        return ""
    # keep the readable sentences of a partly garbled line, drop the ones whose spaces were lost
    sentences = [s for s in split_sentences(line) if not any(len(tok) > MAX_TOKEN for tok in s.split())]
    line = " ".join(sentences)
    if not line:
        return ""
    if not line.endswith((".", "!", "?", ":")):
        line += "."
    return line


@lru_cache(maxsize=256)
def parse_article(title: str, content: str) -> Article:
    body = strip_prefix(title, content or "")
    sections: list[Section] = [Section("", 0)]
    skipping = False
    for raw in body.splitlines():
        m = HEADING_RE.match(raw)
        if m and raw.lstrip().startswith("#"):
            heading = _clean_heading(m.group(2))
            skipping = bool(SKIP_SECTIONS.match(heading))
            sections.append(Section(heading, len(m.group(1)), numbered=bool(STEP_PREFIX_RE.match(m.group(2).strip()))))
            continue
        if skipping:
            continue
        line = _clean_line(raw)
        if line == "Glossary.":
            skipping = True
            continue
        if line:
            sections[-1].paragraphs.append(line)
    sections = [s for s in sections if s.paragraphs or s.heading]
    return Article(title=title.strip(), sections=sections, raw=content or "")


def article_from_payload(payload: Any) -> Article:
    title, content = payload_parts(payload)
    return parse_article(title, content)
