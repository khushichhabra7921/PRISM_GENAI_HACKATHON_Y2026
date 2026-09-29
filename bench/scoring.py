"""Scoring helpers against gold plans.

step accuracy (0-3) = completeness + correctness + ordering, each 0-1:
  completeness  share of gold steps matched one-to-one by a predicted step (fuzzy >= 85)
  correctness   share of predicted steps that match a gold step (penalises extra/invented steps)
  ordering      pairwise order agreement of the matched steps (critical-last is implied)
deeplink relevance (0-2) per gold action: 2 exact target screen, 1 parent/child on the same menu
  branch (or dummy_positive where the gold has a real link), 0 wrong or missing; manual actions
  score 2 when they correctly carry no deeplink.
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional

from rapidfuzz import fuzz

MATCH = 85


def _steps(goal: dict) -> list[str]:
    return [s for a in goal.get("actions", []) for sg in a["stepGroups"] for s in sg["steps"]]


def _match(gold: list[str], pred: list[str]) -> list[tuple[int, int]]:
    """Greedy one-to-one matching by similarity (gold index, pred index)."""
    pairs = sorted(((fuzz.token_set_ratio(g.lower(), p.lower()), gi, pi) for gi, g in enumerate(gold)
                    for pi, p in enumerate(pred)), reverse=True)
    used_g, used_p, out = set(), set(), []
    for score, gi, pi in pairs:
        if score < MATCH:
            break
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            out.append((gi, pi))
    return out


def step_accuracy(pred_goal: Optional[dict], gold_goal: dict) -> dict:
    gold, pred = _steps(gold_goal), _steps(pred_goal or {})
    if not gold:
        return {"score": 3.0 if not pred else 0.0, "completeness": 1.0, "correctness": 1.0, "ordering": 1.0}
    m = _match(gold, pred)
    comp = len(m) / len(gold)
    corr = len(m) / len(pred) if pred else 0.0
    pairs = list(combinations(sorted(m), 2))
    order = (sum(1 for (g1, p1), (g2, p2) in pairs if p1 < p2) / len(pairs)) if pairs else (1.0 if m else 0.0)
    return {"score": round(comp + corr + order, 3), "completeness": round(comp, 3), "correctness": round(corr, 3),
            "ordering": round(order, 3)}


def _paths(catalog) -> dict[str, tuple]:
    return {e.uri: tuple(e.path) for e in catalog}


def deeplink_relevance(pred_goal: Optional[dict], gold_goal: dict, catalog, dummy: str) -> dict:
    paths = _paths(catalog)
    pred_actions = (pred_goal or {}).get("actions", [])
    scores, notes = [], []
    for ga in gold_goal.get("actions", []):
        gsteps = [s for sg in ga["stepGroups"] for s in sg["steps"]]
        gdl = ga["stepGroups"][0].get("actionableDeeplink")
        best, best_sim = None, 0.0
        for pa in pred_actions:
            psteps = [s for sg in pa["stepGroups"] for s in sg["steps"]]
            sim = fuzz.token_set_ratio(" ".join(gsteps).lower(), " ".join(psteps).lower())
            if sim > best_sim:
                best, best_sim = pa, sim
        pdl = best["stepGroups"][0].get("actionableDeeplink") if best and best_sim >= 70 else None
        if gdl is None:
            s = 2 if pdl is None else 0
        elif pdl is None:
            s = 0
        elif pdl["deeplink"] == gdl["deeplink"]:
            s = 2
        else:
            gp, pp = paths.get(gdl["deeplink"], ()), paths.get(pdl["deeplink"], ())
            same_branch = bool(gp and pp and (gp[:len(pp)] == pp or pp[:len(gp)] == gp))
            s = 1 if same_branch or pdl["deeplink"] == dummy else 0
        scores.append(s)
        notes.append({"gold_action": ga["actionName"], "pred_action": best["actionName"] if best else None,
                      "gold": (gdl or {}).get("description"), "pred": (pdl or {}).get("description"), "score": s})
    return {"score": round(sum(scores) / len(scores), 3) if scores else 2.0, "per_action": notes}
