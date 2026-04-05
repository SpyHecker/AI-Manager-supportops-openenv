from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.environment import SupportEnv
from app.tasks import list_tasks
from app.utils import parse_action_payload


app = FastAPI(
    title="SupportOps-OpenEnv",
    description="Customer Support AI Manager environment built with OpenEnv-compatible API patterns.",
    version="0.1.0",
)

env = SupportEnv(max_steps=6)


class ResetRequest(BaseModel):
    difficulty: Optional[str] = "easy"
    task_id: Optional[str] = None


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
def tasks() -> Dict[str, Any]:
    return {"tasks": list_tasks()}


@app.post("/reset")
def reset(request: ResetRequest) -> Dict[str, Any]:
    try:
        observation = env.reset(
            difficulty=request.difficulty or "easy",
            task_id=request.task_id,
        )
        return {
            "observation": observation.model_dump(),
            "done": False,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step")
def step(action: Dict[str, Any]) -> Dict[str, Any]:
    try:
        parsed_action = parse_action_payload(action)
        observation, reward, done, info = env.step(parsed_action)

        return {
            "observation": observation.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state")
def state() -> Dict[str, Any]:
    try:
        return env.state().model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))