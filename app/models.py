from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class SupportAction(BaseModel):
    action_type: Literal[
        "classify_ticket",
        "set_priority",
        "assign_team",
        "request_more_info",
        "draft_response",
        "escalate_ticket",
        "resolve_ticket",
        "close_without_resolution",
    ]
    category: Optional[str] = None
    priority: Optional[str] = None
    team: Optional[str] = None
    message: Optional[str] = None
    escalation_reason: Optional[str] = None


class SupportObservation(BaseModel):
    ticket_id: str
    task_id: str
    difficulty: str

    customer_message: str
    conversation_history: List[str]

    customer_tier: str
    product_area: str

    current_category: Optional[str] = None
    current_priority: Optional[str] = None
    assigned_team: Optional[str] = None

    last_action_result: str
    allowed_actions: List[str]

    step_count: int
    max_steps: int

    ready_for_resolution: bool
    done: bool


class SupportState(BaseModel):
    # task / episode identity
    ticket_id: str
    task_id: str
    difficulty: str

    # episode progress
    step_count: int
    max_steps: int

    # agent-updated fields
    current_category: Optional[str] = None
    current_priority: Optional[str] = None
    assigned_team: Optional[str] = None

    info_requested: bool = False
    response_drafted: bool = False
    escalated: bool = False
    resolved: bool = False
    closed_without_resolution: bool = False

    # tracking
    action_history: List[str] = Field(default_factory=list)
    cumulative_reward: float = 0.0
    done: bool = False

    # response tracking
    last_drafted_response: Optional[str] = None

    # hidden grading ground truth
    ground_truth_category: str
    ground_truth_priority: str
    correct_team: str

    escalation_required: bool
    more_info_required: bool
    resolution_allowed: bool

    required_response_keywords: List[str] = Field(default_factory=list)
    prohibited_response_keywords: List[str] = Field(default_factory=list)