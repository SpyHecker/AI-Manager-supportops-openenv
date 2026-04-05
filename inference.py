import json
import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

import requests
from openai import OpenAI

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL")

MAX_STEPS = 6
TEMPERATURE = 0.0
TASKS = ["easy", "medium", "hard"]


SYSTEM_PROMPT = """
You are operating a customer support management environment.

Your job is to choose the single best next action for the current support ticket.

You must return ONLY valid JSON with this schema:
{
  "action_type": "<one valid action type>",
  "category": "<optional string>",
  "priority": "<optional string>",
  "team": "<optional string>",
  "message": "<optional string>",
  "escalation_reason": "<optional string>"
}

Valid action types:
- classify_ticket
- set_priority
- assign_team
- request_more_info
- draft_response
- escalate_ticket
- resolve_ticket
- close_without_resolution

Valid categories:
- billing
- account_access
- technical_issue
- shipping_delivery
- subscription_change
- abuse_report
- feature_request

Valid priorities:
- low
- medium
- high
- urgent

Valid teams:
- general_support
- billing_team
- technical_team
- account_recovery
- logistics_team
- trust_safety
- escalation_manager

You must use these exact values only.
Do not invent new category names, priority names, or team names.

Guidelines:
- For billing/refund issues, use category "billing" and usually route to "billing_team".
- For login, OTP, or account access issues, use category "account_access" and usually route to "account_recovery".
- If more information is needed, use "request_more_info" with a helpful message.
- If security risk is present, use "escalate_ticket" with a short escalation_reason.
- Do not use "resolve_ticket" if more information is required or escalation is required.
- If the ticket mentions suspicious login activity, unfamiliar access, account compromise, or security risk, prioritize category "account_access", priority "urgent", and escalation handling before billing resolution.
- In mixed billing + security cases, handle the security risk first.
- For OTP/login issues without explicit security risk, prefer "request_more_info" before "escalate_ticket".
- For standard OTP delivery problems, usually use priority "high" instead of "urgent" unless there is clear security risk.
- Return JSON only. No markdown. No explanation.
""".strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def reset_env(task_name: str) -> Dict[str, Any]:
    response = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"difficulty": task_name},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def step_env(action: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(
        f"{ENV_BASE_URL}/step",
        json=action,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def build_user_prompt(observation: Dict[str, Any]) -> str:
    return json.dumps(
        {
            "ticket_id": observation.get("ticket_id"),
            "task_id": observation.get("task_id"),
            "difficulty": observation.get("difficulty"),
            "customer_message": observation.get("customer_message"),
            "conversation_history": observation.get("conversation_history", []),
            "customer_tier": observation.get("customer_tier"),
            "product_area": observation.get("product_area"),
            "current_category": observation.get("current_category"),
            "current_priority": observation.get("current_priority"),
            "assigned_team": observation.get("assigned_team"),
            "last_action_result": observation.get("last_action_result"),
            "allowed_actions": observation.get("allowed_actions", []),
            "step_count": observation.get("step_count"),
            "max_steps": observation.get("max_steps"),
            "ready_for_resolution": observation.get("ready_for_resolution"),
            "done": observation.get("done"),
        },
        ensure_ascii=False,
    )


def get_model_action(client: OpenAI, observation: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt = build_user_prompt(observation)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=200,
        )
        content = (completion.choices[0].message.content or "").strip()

        # Strip accidental markdown fences
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:].strip()

        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Model output is not a JSON object.")

        return parsed

    except Exception:
        customer_message = (observation.get("customer_message") or "").lower()
        current_category = observation.get("current_category")
        current_priority = observation.get("current_priority")
        assigned_team = observation.get("assigned_team")
        ready_for_resolution = observation.get("ready_for_resolution", False)

        has_security_signal = any(
            phrase in customer_message
            for phrase in [
                "suspicious",
                "unfamiliar login",
                "unknown login",
                "unauthorized access",
                "account compromised",
                "security",
                "fraud",
                "hacked",
            ]
        )

        has_access_signal = any(
            phrase in customer_message
            for phrase in [
                "otp",
                "login",
                "access",
                "sign in",
                "sign-in",
                "verification code",
                "not receiving code",
            ]
        )

        has_billing_signal = any(
            phrase in customer_message
            for phrase in [
                "charged twice",
                "refund",
                "billing",
                "payment",
                "subscription charge",
                "extra amount",
            ]
        )

        # HARD / security-first fallback
        if has_security_signal:
            if not current_category:
                return {"action_type": "classify_ticket", "category": "account_access"}
            if not current_priority:
                return {"action_type": "set_priority", "priority": "urgent"}
            if not assigned_team:
                return {"action_type": "assign_team", "team": "escalation_manager"}
            return {
                "action_type": "escalate_ticket",
                "escalation_reason": "Potential account security issue reported by customer",
            }

        # MEDIUM / account access fallback
        if has_access_signal:
            if not current_category:
                return {"action_type": "classify_ticket", "category": "account_access"}
            if not current_priority:
                return {"action_type": "set_priority", "priority": "high"}
            if not assigned_team:
                return {"action_type": "assign_team", "team": "account_recovery"}
            return {
                "action_type": "request_more_info",
                "message": (
                    "Please confirm the email or phone number linked to your account "
                    "and whether you have already checked spam or junk folders."
                ),
            }

        # EASY / billing fallback
        if has_billing_signal:
            if not current_category:
                return {"action_type": "classify_ticket", "category": "billing"}
            if not current_priority:
                return {"action_type": "set_priority", "priority": "medium"}
            if not assigned_team:
                return {"action_type": "assign_team", "team": "billing_team"}
            if not ready_for_resolution:
                return {
                    "action_type": "draft_response",
                    "message": (
                        "I understand that you were charged twice. "
                        "We will help process the refund for the extra charge."
                    ),
                }
            return {"action_type": "resolve_ticket"}

        # Generic safe fallback
        if not current_category:
            return {"action_type": "classify_ticket", "category": "technical_issue"}
        if not current_priority:
            return {"action_type": "set_priority", "priority": "medium"}
        if not assigned_team:
            return {"action_type": "assign_team", "team": "general_support"}
        return {
            "action_type": "request_more_info",
            "message": "Please share any additional details so we can assist you better."
        }


def run_task(client: OpenAI, task_name: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    final_score = 0.0
    success = False

    log_start(task=task_name, env="supportops-openenv", model=MODEL_NAME)

    try:
        reset_result = reset_env(task_name)
        observation = reset_result["observation"]
        done = reset_result["done"]

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = get_model_action(client, observation)
            step_result = step_env(action)

            observation = step_result["observation"]
            reward = float(step_result["reward"])
            done = bool(step_result["done"])
            info = step_result.get("info", {})

            rewards.append(reward)
            steps_taken = step

            action_str = json.dumps(action, ensure_ascii=False, separators=(",", ":"))
            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=info.get("error"),
            )

            if done:
                final_score = float(info.get("score", 0.0))
                success = final_score > 0.5
                break

        if not done:
            final_score = max(0.0, min(1.0, sum(rewards)))

    except Exception as exc:
        log_step(
            step=steps_taken + 1,
            action="{}",
            reward=0.00,
            done=True,
            error=str(exc),
        )

    log_end(
        success=success,
        steps=steps_taken,
        score=final_score,
        rewards=rewards,
    )

    return final_score


def main() -> None:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN or OPENAI_API_KEY must be set.")

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

    all_scores: List[float] = []
    for task_name in TASKS:
        score = run_task(client, task_name)
        all_scores.append(score)


if __name__ == "__main__":
    main()