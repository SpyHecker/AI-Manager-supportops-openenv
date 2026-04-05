from typing import Dict, List

from app.models import SupportState
from app.utils import contains_any_keyword


def _score_response_keywords(
    response_text: str | None,
    required_keywords: List[str],
    prohibited_keywords: List[str],
) -> float:
    """
    Scores drafted response content:
    - full credit if at least one required keyword is present
      and no prohibited keyword appears
    - partial credit if no response exists or required terms are missing
    """
    if not response_text:
        return 0.0

    has_required = contains_any_keyword(response_text, required_keywords) if required_keywords else True
    has_prohibited = contains_any_keyword(response_text, prohibited_keywords) if prohibited_keywords else False

    if has_required and not has_prohibited:
        return 1.0
    if has_prohibited:
        return 0.0
    return 0.5


def _score_final_outcome(state: SupportState) -> float:
    """
    Final outcome depends on task policy:
    - if escalation required, escalated is good, direct resolution is bad
    - if more info required, direct resolution without asking is bad
    - if resolution allowed, proper resolution is good
    """
    if state.escalation_required:
        if state.escalated and not state.resolved:
            return 1.0
        if state.resolved:
            return 0.0
        return 0.25

    if state.more_info_required:
        if state.info_requested and not state.resolved:
            return 1.0
        if state.resolved:
            return 0.0
        return 0.25

    if state.resolution_allowed:
        if state.resolved:
            return 1.0
        return 0.25

    if state.closed_without_resolution:
        return 0.0

    return 0.0


def grade_episode(state: SupportState) -> Dict:
    """
    Deterministic grading rubric with score in [0.0, 1.0].
    Returns both total score and a detailed breakdown.
    """

    breakdown = {
        "category": 0.0,
        "priority": 0.0,
        "team": 0.0,
        "info_request": 0.0,
        "escalation": 0.0,
        "response_quality": 0.0,
        "final_outcome": 0.0,
    }

    # 1. Category correctness
    if state.current_category == state.ground_truth_category:
        breakdown["category"] = 1.0

    # 2. Priority correctness
    if state.current_priority == state.ground_truth_priority:
        breakdown["priority"] = 1.0

    # 3. Team correctness
    if state.assigned_team == state.correct_team:
        breakdown["team"] = 1.0

    # 4. Info request correctness
    if state.more_info_required:
        breakdown["info_request"] = 1.0 if state.info_requested else 0.0
    else:
        # if not required, agent gets full credit for not overcomplicating
        breakdown["info_request"] = 1.0 if not state.info_requested else 0.5

    # 5. Escalation correctness
    if state.escalation_required:
        breakdown["escalation"] = 1.0 if state.escalated else 0.0
    else:
        breakdown["escalation"] = 1.0 if not state.escalated else 0.5

    # 6. Drafted response quality
    breakdown["response_quality"] = _score_response_keywords(
        response_text=state.last_drafted_response,
        required_keywords=state.required_response_keywords,
        prohibited_keywords=state.prohibited_response_keywords,
    )

    # 7. Final outcome correctness
    breakdown["final_outcome"] = _score_final_outcome(state)

    # Weighted total
    weights = {
        "category": 0.20,
        "priority": 0.15,
        "team": 0.15,
        "info_request": 0.10,
        "escalation": 0.15,
        "response_quality": 0.10,
        "final_outcome": 0.15,
    }

    total_score = 0.0
    for key, value in breakdown.items():
        total_score += value * weights[key]

    total_score = max(0.0, min(1.0, total_score))

    return {
        "score": round(total_score, 4),
        "breakdown": breakdown,
    }