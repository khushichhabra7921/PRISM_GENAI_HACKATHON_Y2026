"""⑤ Ordering: classify auto / manual / critical by rules, sort auto -> manual -> critical (stable)."""
from __future__ import annotations

from app.steps import DraftAction, categorise

ORDER = {"auto": 0, "manual": 1, "critical": 2}


def classify(action: DraftAction) -> str:
    """Rules decide the category, whatever an LLM suggested: critical keywords win,
    a settings screen reachable by path is auto, anything physical/other is manual."""
    if any(u.category == "critical" for u in action.units):
        return "critical"
    return categorise(" ".join(action.steps), action.path)


def order_actions(actions: list[DraftAction]) -> list[DraftAction]:
    for a in actions:
        a.category = classify(a)
    return sorted(actions, key=lambda a: ORDER[a.category])  # sorted() is stable: source order kept per category
