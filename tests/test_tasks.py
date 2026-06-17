from benchmark.tasks import load_tasks
from benchmark.types import Task


def test_load_tasks_code_returns_tasks():
    tasks = load_tasks(["code"])
    assert len(tasks) >= 3
    for t in tasks:
        assert isinstance(t, Task)
        assert t.category == "code"
        assert t.id
        assert t.prompt
        assert t.judge_hints


def test_load_tasks_multiple_categories():
    tasks = load_tasks(["code", "reasoning"])
    categories = {t.category for t in tasks}
    assert "code" in categories
    assert "reasoning" in categories


def test_load_tasks_all_categories():
    tasks = load_tasks(["code", "reasoning", "summarization", "qa_pt"])
    assert len(tasks) >= 12
