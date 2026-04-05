from app.environment import SupportEnv
from app.models import SupportAction


def test_reset_returns_observation():
    env = SupportEnv()
    obs = env.reset("easy")

    assert obs.ticket_id is not None
    assert obs.task_id is not None
    assert obs.done is False


def test_easy_task_happy_path():
    env = SupportEnv()
    env.reset("easy")

    obs, reward, done, info = env.step(
        SupportAction(action_type="classify_ticket", category="billing")
    )
    assert reward >= 0.0
    assert done is False

    obs, reward, done, info = env.step(
        SupportAction(action_type="set_priority", priority="medium")
    )
    assert done is False

    obs, reward, done, info = env.step(
        SupportAction(action_type="assign_team", team="billing_team")
    )
    assert done is False

    obs, reward, done, info = env.step(
        SupportAction(action_type="resolve_ticket")
    )
    assert done is True
    assert "score" in info


def test_invalid_action_penalty():
    env = SupportEnv()
    env.reset("easy")

    obs, reward, done, info = env.step(
        SupportAction(action_type="assign_team", team="wrong_team")
    )
    assert reward <= 0.0 or info["success"] is False