import json
from pathlib import Path
from typing import Dict, List


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


TASK_FILES = {
    "easy": "easy_task.json",
    "medium": "medium_task.json",
    "hard": "hard_task.json",
}


def _load_json_file(filename: str) -> Dict:
    file_path = DATA_DIR / filename
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_task_by_difficulty(difficulty: str) -> Dict:
    difficulty = difficulty.strip().lower()
    if difficulty not in TASK_FILES:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    return _load_json_file(TASK_FILES[difficulty])


def load_task_by_id(task_id: str) -> Dict:
    for filename in TASK_FILES.values():
        task = _load_json_file(filename)
        if task.get("task_id") == task_id:
            return task
    raise ValueError(f"Task not found: {task_id}")


def list_tasks() -> List[Dict]:
    tasks = []
    for filename in TASK_FILES.values():
        task = _load_json_file(filename)
        tasks.append(
            {
                "task_id": task["task_id"],
                "difficulty": task["difficulty"],
                "description": task.get("description", ""),
            }
        )
    return tasks