from copy import deepcopy
from typing import Optional

from app.graders import grade_episode
from app.models import SupportAction, SupportObservation, SupportState
from app.rewards import compute_step_reward
from app.tasks import load_task_by_difficulty, load_task_by_id
from app.utils import (
    action_to_log_string,
    append_history,
    make_info_dict,
    validate_action,
)


class SupportEnv:
    def __init__(self, max_steps: int = 6):
        self._state: Optional[SupportState] = None
        self._task = None
        self._last_action_result = "Environment not started."
        self._max_steps_default = max_steps

    def reset(self, difficulty: str = "easy", task_id: Optional[str] = None) -> SupportObservation:
        if task_id:
            task = load_task_by_id(task_id)
        else:
            task = load_task_by_difficulty(difficulty)

        ticket = task["ticket"]
        gt = task["ground_truth"]

        self._task = task
        self._last_action_result = "Environment reset."

        self._state = SupportState(
            ticket_id=ticket["ticket_id"],
            task_id=task["task_id"],
            difficulty=task["difficulty"],
            step_count=0,
            max_steps=self._max_steps_default,
            current_category=None,
            current_priority=None,
            assigned_team=None,
            info_requested=False,
            response_drafted=False,
            escalated=False,
            resolved=False,
            closed_without_resolution=False,
            action_history=[],
            cumulative_reward=0.0,
            done=False,
            ground_truth_category=gt["category"],
            ground_truth_priority=gt["priority"],
            correct_team=gt["team"],
            escalation_required=gt["escalation_required"],
            more_info_required=gt["more_info_required"],
            resolution_allowed=gt["resolution_allowed"],
            last_drafted_response=None,
            required_response_keywords=gt.get("required_response_keywords", []),
            prohibited_response_keywords=gt.get("prohibited_response_keywords", []),
        )

        return self._get_observation()

    def step(self, action: SupportAction):
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self._state.done:
            final_grade = grade_episode(self._state)
            return (
                self._get_observation(),
                0.0,
                True,
                make_info_dict(
                    success=True,
                    message="Episode already finished.",
                    score=final_grade["score"],
                    error=None,
                ),
            )

        prev_state = deepcopy(self._state)

        is_valid, validation_message = validate_action(action)
        if not is_valid:
            self._state.step_count += 1
            self._last_action_result = f"Invalid action: {validation_message}"
            self._state.action_history = append_history(
                self._state.action_history,
                f"invalid:{action.action_type}",
            )

            reward = compute_step_reward(prev_state, self._state, action, action_valid=False)
            self._state.cumulative_reward += reward

            if self._state.step_count >= self._state.max_steps:
                self._state.done = True

            info = make_info_dict(
                success=False,
                message="Action validation failed.",
                error=validation_message,
            )
            return self._get_observation(), reward, self._state.done, info

        self._state.step_count += 1
        self._apply_action(action)

        action_log = action_to_log_string(action)
        self._state.action_history = append_history(self._state.action_history, action_log)

        reward = compute_step_reward(prev_state, self._state, action, action_valid=True)
        self._state.cumulative_reward += reward

        self._update_done_status()

        info = {}
        if self._state.done:
            final_grade = grade_episode(self._state)
            info = make_info_dict(
                success=final_grade["score"] > 0.5,
                message="Episode completed.",
                score=final_grade["score"],
                error=None,
            )
            info["grading_breakdown"] = final_grade["breakdown"]
        else:
            info = make_info_dict(
                success=True,
                message=self._last_action_result,
                score=None,
                error=None,
            )

        return self._get_observation(), reward, self._state.done, info

    def state(self) -> SupportState:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state

    def _apply_action(self, action: SupportAction) -> None:
        assert self._state is not None

        if action.action_type == "classify_ticket":
            self._state.current_category = action.category
            self._last_action_result = f"Category set to '{action.category}'."
            return

        if action.action_type == "set_priority":
            self._state.current_priority = action.priority
            self._last_action_result = f"Priority set to '{action.priority}'."
            return

        if action.action_type == "assign_team":
            self._state.assigned_team = action.team
            self._last_action_result = f"Assigned to '{action.team}'."
            return

        if action.action_type == "request_more_info":
            self._state.info_requested = True
            self._last_action_result = "Requested more information from customer."
            return

        if action.action_type == "draft_response":
            self._state.response_drafted = True
            self._state.last_drafted_response = action.message
            self._last_action_result = "Drafted customer response."
            return

        if action.action_type == "escalate_ticket":
            self._state.escalated = True
            self._last_action_result = (
                f"Ticket escalated. Reason: {action.escalation_reason or 'not provided'}."
            )
            return

        if action.action_type == "resolve_ticket":
            self._state.resolved = True
            self._last_action_result = "Ticket marked resolved."
            return

        if action.action_type == "close_without_resolution":
            self._state.closed_without_resolution = True
            self._last_action_result = "Ticket closed without resolution."
            return

        self._last_action_result = f"Unhandled action type: {action.action_type}"

    def _update_done_status(self) -> None:
        assert self._state is not None

        if self._state.closed_without_resolution:
            self._state.done = True
            return

        if self._state.escalation_required and self._state.escalated:
            self._state.done = True
            return

        if self._state.more_info_required and self._state.info_requested:
            self._state.done = True
            return

        if self._state.resolution_allowed and self._state.resolved:
            self._state.done = True
            return

        if self._state.resolved and not self._state.resolution_allowed:
            self._state.done = True
            return

        if self._state.step_count >= self._state.max_steps:
            self._state.done = True

    def _get_observation(self) -> SupportObservation:
        if self._state is None or self._task is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        ticket = self._task["ticket"]

        ready_for_resolution = (
            not self._state.more_info_required
            and not self._state.escalation_required
            and self._state.current_category is not None
            and self._state.current_priority is not None
            and self._state.assigned_team is not None
        )

        return SupportObservation(
            ticket_id=self._state.ticket_id,
            task_id=self._state.task_id,
            difficulty=self._state.difficulty,
            customer_message=ticket["customer_message"],
            conversation_history=ticket.get("conversation_history", []),
            customer_tier=ticket["customer_tier"],
            product_area=ticket["product_area"],
            current_category=self._state.current_category,
            current_priority=self._state.current_priority,
            assigned_team=self._state.assigned_team,
            last_action_result=self._last_action_result,
            allowed_actions=[
                "classify_ticket",
                "set_priority",
                "assign_team",
                "request_more_info",
                "draft_response",
                "escalate_ticket",
                "resolve_ticket",
                "close_without_resolution",
            ],
            step_count=self._state.step_count,
            max_steps=self._state.max_steps,
            ready_for_resolution=ready_for_resolution,
            done=self._state.done,
        )   