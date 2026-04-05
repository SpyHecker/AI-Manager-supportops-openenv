from typing import Any, Dict, Optional, Tuple

from app.models import SupportAction


VALID_ACTION_TYPES = {
    "classify_ticket",
    "set_priority",
    "assign_team",
    "request_more_info",
    "draft_response",
    "escalate_ticket",
    "resolve_ticket",
    "close_without_resolution",
}

VALID_CATEGORIES = {
    "billing",
    "account_access",
    "technical_issue",
    "shipping_delivery",
    "subscription_change",
    "abuse_report",
    "feature_request",
}

VALID_PRIORITIES = {"low", "medium", "high", "urgent"}

VALID_TEAMS = {
    "general_support",
    "billing_team",
    "technical_team",
    "account_recovery",
    "logistics_team",
    "trust_safety",
    "escalation_manager",
}


def normalize_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip().lower()
    return cleaned if cleaned else None


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def validate_category(category: Optional[str]) -> bool:
    return normalize_text(category) in VALID_CATEGORIES


def validate_priority(priority: Optional[str]) -> bool:
    return normalize_text(priority) in VALID_PRIORITIES


def validate_team(team: Optional[str]) -> bool:
    return normalize_text(team) in VALID_TEAMS


def validate_action(action: SupportAction) -> Tuple[bool, str]:
    action_type = normalize_text(action.action_type)

    if action_type not in VALID_ACTION_TYPES:
        return False, f"unsupported action_type: {action.action_type}"

    if action_type == "classify_ticket":
        if not action.category:
            return False, "category is required for classify_ticket"
        if not validate_category(action.category):
            return False, f"invalid category: {action.category}"
        return True, "valid action"

    if action_type == "set_priority":
        if not action.priority:
            return False, "priority is required for set_priority"
        if not validate_priority(action.priority):
            return False, f"invalid priority: {action.priority}"
        return True, "valid action"

    if action_type == "assign_team":
        if not action.team:
            return False, "team is required for assign_team"
        if not validate_team(action.team):
            return False, f"invalid team: {action.team}"
        return True, "valid action"

    if action_type == "request_more_info":
        if not action.message or not safe_str(action.message):
            return False, "message is required for request_more_info"
        return True, "valid action"

    if action_type == "draft_response":
        if not action.message or not safe_str(action.message):
            return False, "message is required for draft_response"
        return True, "valid action"

    if action_type == "escalate_ticket":
        if not action.escalation_reason or not safe_str(action.escalation_reason):
            return False, "escalation_reason is required for escalate_ticket"
        return True, "valid action"

    if action_type in {"resolve_ticket", "close_without_resolution"}:
        return True, "valid action"

    return False, f"unsupported action_type: {action.action_type}"


def parse_action_payload(payload: Dict[str, Any]) -> SupportAction:
    cleaned = {
        "action_type": safe_str(payload.get("action_type")),
        "category": normalize_text(payload.get("category")),
        "priority": normalize_text(payload.get("priority")),
        "team": normalize_text(payload.get("team")),
        "message": safe_str(payload.get("message")) or None,
        "escalation_reason": safe_str(payload.get("escalation_reason")) or None,
    }
    return SupportAction(**cleaned)


def action_to_log_string(action: SupportAction) -> str:
    parts = [f"action_type={action.action_type}"]

    if action.category:
        parts.append(f"category={action.category}")
    if action.priority:
        parts.append(f"priority={action.priority}")
    if action.team:
        parts.append(f"team={action.team}")
    if action.message:
        message = action.message.replace("\n", " ").strip()
        parts.append(f"message={message}")
    if action.escalation_reason:
        reason = action.escalation_reason.replace("\n", " ").strip()
        parts.append(f"escalation_reason={reason}")

    return " | ".join(parts)


def append_history(history: list[str], entry: str, max_items: int = 10) -> list[str]:
    updated = history + [entry]
    return updated[-max_items:]


def has_keyword(text: Optional[str], keyword: str) -> bool:
    if not text:
        return False
    return keyword.lower() in text.lower()


def contains_any_keyword(text: Optional[str], keywords: list[str]) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)


def make_info_dict(
    success: bool,
    message: str,
    score: Optional[float] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "success": success,
        "message": message,
        "error": error,
    }
    if score is not None:
        info["score"] = round(float(score), 4)
    return info