"""Deeplink ground truth derived from SIIS menu paths (evaluation only).

For every screen-level action the rules extractor finds in the SIIS text, the expected
catalog entry is the one whose `path` equals the menu path in the text (or the toggle
child named in a "Turn on X" step). The engine itself never sees `path` for matching.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from app.data_loader import Kit
from app.deeplinks import ActionQuery
from app.steps import rules_extract


@dataclass
class DLCase:
    siis_id: str
    query: ActionQuery
    gold_uri: Optional[str]  # None -> screen not in catalog (expect dummy_positive)
    parents: set[str]


def build_cases(kit: Kit) -> list[DLCase]:
    by_path = {tuple(e.path): e for e in kit.catalog if e.path}
    known = {" > ".join(p) for p in by_path}
    cases = []
    for doc in kit.siis:
        for a in rules_extract(doc.text, known):
            if not a.path:
                continue
            q = ActionQuery(a.name, a.steps, a.path)
            gold = by_path.get(a.path)
            for s in a.steps:
                m = re.match(r"Turn (?:on|off) ([A-Z][^.,]*)", s)
                if m:
                    child = by_path.get(a.path + (re.split(r" and | if | when ", m.group(1))[0].strip(),))
                    gold = child or gold
            parents = {by_path[a.path[:i]].uri for i in range(1, len(a.path)) if a.path[:i] in by_path}
            if gold and gold.path != a.path:
                parents.add(by_path[a.path].uri)
            cases.append(DLCase(doc.id, q, gold.uri if gold else None, parents))
    return cases


def relevance(pred_uri: Optional[str], case: DLCase, dummy: str) -> int:
    """2 = exact screen, 1 = parent/sibling on the same path (or dummy when screen missing), 0 = wrong/none."""
    if case.gold_uri is None:
        return 2 if pred_uri == dummy else 0
    if pred_uri == case.gold_uri:
        return 2
    if pred_uri in case.parents or pred_uri == dummy:
        return 1
    return 0
