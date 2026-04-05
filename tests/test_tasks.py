from app.tasks import load_task_by_difficulty, list_tasks


def test_load_easy_task():
    task = load_task_by_difficulty("easy")
    assert task["difficulty"] == "easy"
    assert "ticket" in task
    assert "ground_truth" in task


def test_load_medium_task():
    task = load_task_by_difficulty("medium")
    assert task["difficulty"] == "medium"


def test_load_hard_task():
    task = load_task_by_difficulty("hard")
    assert task["difficulty"] == "hard"


def test_list_tasks():
    tasks = list_tasks()
    assert len(tasks) == 3