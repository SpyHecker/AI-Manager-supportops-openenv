from app.models import SupportAction, SupportState


def compute_step_reward(
    prev_state: SupportState,
    new_state: SupportState,
    action: SupportAction,
    action_valid: bool,
) -> float:
    """
    Dense reward shaping for trajectory-level learning.
    Returns a float reward, typically kept within [-0.2, 0.25] per step.
    """
    reward = 0.0

    if not action_valid:
        return -0.10

    # Reward newly correct classification
    if (
        new_state.current_category == new_state.ground_truth_category
        and prev_state.current_category != new_state.current_category
    ):
        reward += 0.15

    # Reward newly correct priority
    if (
        new_state.current_priority == new_state.ground_truth_priority
        and prev_state.current_priority != new_state.current_priority
    ):
        reward += 0.10

    # Reward newly correct team assignment
    if (
        new_state.assigned_team == new_state.correct_team
        and prev_state.assigned_team != new_state.assigned_team
    ):
        reward += 0.10

    # Reward asking for more info only when needed
    if action.action_type == "request_more_info":
        if new_state.more_info_required and not prev_state.info_requested and new_state.info_requested:
            reward += 0.15
        elif not new_state.more_info_required:
            reward -= 0.05

    # Reward drafting a response, but only lightly here
    if action.action_type == "draft_response":
        if new_state.last_drafted_response and not prev_state.last_drafted_response:
            reward += 0.05

    # Reward/penalize escalation
    if action.action_type == "escalate_ticket":
        if new_state.escalation_required and not prev_state.escalated and new_state.escalated:
            reward += 0.20
        elif not new_state.escalation_required:
            reward -= 0.10

    # Reward correct resolution or penalize premature/wrong resolution
    if action.action_type == "resolve_ticket":
        if new_state.resolution_allowed and not new_state.more_info_required and not new_state.escalation_required:
            reward += 0.15
        else:
            reward -= 0.20

    # Penalize closing without resolution
    if action.action_type == "close_without_resolution":
        reward -= 0.20

    # Penalize repeated no-progress actions
    if _is_no_progress_repeat(prev_state, new_state, action):
        reward -= 0.05

    # Clamp per-step reward for stability
    reward = max(-0.25, min(0.25, reward))
    return round(reward, 4)


def _is_no_progress_repeat(
    prev_state: SupportState,
    new_state: SupportState,
    action: SupportAction,
) -> bool:
    """
    Penalize repeating the same kind of action without changing useful state.
    """
    if not prev_state.action_history:
        return False

    last_action = prev_state.action_history[-1]

    same_action_type = last_action.startswith(action.action_type)
    no_state_change = (
        prev_state.current_category == new_state.current_category
        and prev_state.current_priority == new_state.current_priority
        and prev_state.assigned_team == new_state.assigned_team
        and prev_state.info_requested == new_state.info_requested
        and prev_state.escalated == new_state.escalated
        and prev_state.resolved == new_state.resolved
        and prev_state.closed_without_resolution == new_state.closed_without_resolution
        and prev_state.last_drafted_response == new_state.last_drafted_response
    )

    return same_action_type and no_state_change