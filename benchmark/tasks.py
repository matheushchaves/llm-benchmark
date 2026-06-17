from pathlib import Path

import yaml

from benchmark.types import Task

TASKS_DIR = Path(__file__).parent.parent / "tasks"


def load_tasks(categories: list[str]) -> list[Task]:
    tasks = []
    for category in categories:
        path = TASKS_DIR / f"{category}.yaml"
        with open(path) as f:
            raw = yaml.safe_load(f)
        for item in raw:
            tasks.append(Task(
                id=item["id"],
                prompt=item["prompt"],
                category=item["category"],
                judge_hints=item.get("judge_hints", ""),
            ))
    return tasks
