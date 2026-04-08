from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.environment import SupportEnv
from app.tasks import list_tasks
from app.utils import parse_action_payload


app = FastAPI(
    title="SupportOps-OpenEnv",
    description="Customer Support AI Manager environment built with OpenEnv-compatible API patterns.",
    version="0.1.0",
)

env = SupportEnv(max_steps=6)

# ------------------ MODELS ------------------

class Action(BaseModel):
    action_type: str = "classify_ticket"
    category: Optional[str] = "billing"
    priority: Optional[str] = "medium"
    team: Optional[str] = "billing_team"
    message: Optional[str] = None
    escalation_reason: Optional[str] = None


class StepRequest(BaseModel):
    action: Optional[Action] = Action()   # 🔥 DEFAULT ACTION


class ResetRequest(BaseModel):
    difficulty: Optional[str] = Field(default="easy", example="easy")
    task_id: Optional[str] = Field(default="support_easy_001", example="support_easy_001")


# ------------------ ROUTES ------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
def tasks() -> Dict[str, Any]:
    return {"tasks": list_tasks()}


@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()) -> Dict[str, Any]:
    try:
        print("START")  # REQUIRED LOG

        observation = env.reset(
            difficulty=request.difficulty or "easy",
            task_id=request.task_id or "support_easy_001"
        )

        return {
            "observation": observation.model_dump(),
            "done": False,
        }

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step")
def step(request: StepRequest) -> Dict[str, Any]:
    try:
        print("STEP")

        # 🔥 AUTO RESET IF NOT INITIALIZED
        try:
            env.state()
        except Exception:
            print("AUTO RESET TRIGGERED")
            env.reset(difficulty="easy", task_id="support_easy_001")

        action_data = request.action.model_dump() if request.action else Action().model_dump()

        parsed_action = parse_action_payload(action_data)

        observation, reward, done, info = env.step(parsed_action)

        if done:
            print("END")

        return {
            "observation": observation.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }

    except Exception as exc:
        print("ERROR:", str(exc))
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/")
def root():
    return {"message": "AI Manager SupportOps OpenEnv is running"}


@app.get("/state")
def state() -> Dict[str, Any]:
    try:
        return env.state().model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
