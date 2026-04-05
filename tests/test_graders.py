from app.graders import grade_episode
from app.models import SupportState


def make_base_state():
    return SupportState(
        ticket_id="T1",
        task_id="support_easy_001",
        difficulty="easy",
        step_count=1,
        max_steps=6,
        ground_truth_category="billing",
        ground_truth_priority="medium",
        correct_team="billing_team",
        escalation_required=False,
        more_info_required=False,
        resolution_allowed=True,
    )


def test_grade_score_range():
    state = make_base_state()
    result = grade_episode(state)
    assert 0.0 <= result["score"] <= 1.0


def test_correct_state_scores_higher():
    wrong_state = make_base_state()

    correct_state = make_base_state()
    correct_state.current_category = "billing"
    correct_state.current_priority = "medium"
    correct_state.assigned_team = "billing_team"
    correct_state.resolved = True

    wrong_score = grade_episode(wrong_state)["score"]
    correct_score = grade_episode(correct_state)["score"]

    assert correct_score >= wrong_score