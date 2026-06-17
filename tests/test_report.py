from pathlib import Path

import pytest

from benchmark.report import generate_report
from benchmark.types import JudgeResult, ModelResponse, Task, TaskResult


def _make_result(task_id: str, category: str, score_c: float, score_g: float, winner: str) -> TaskResult:
    task = Task(id=task_id, prompt="prompt", category=category, judge_hints="hints")
    claude_r = ModelResponse("claude answer", 500.0, 10, 50, 0.001)
    gemma_r = ModelResponse("gemma answer", 800.0, 0, 60, 0.0)
    judge = JudgeResult(score_c, score_g, winner, "justification text here")
    return TaskResult(task=task, claude_response=claude_r, gemma_response=gemma_r, judge_result=judge)


def test_generate_report_returns_markdown_string():
    results = [
        _make_result("code_fib", "code", 5.0, 3.0, "claude"),
        _make_result("reason_1", "reasoning", 3.0, 4.0, "gemma"),
    ]
    md = generate_report(results)

    assert "# Benchmark" in md
    assert "code" in md
    assert "reasoning" in md
    assert "Claude" in md
    assert "Gemma" in md
    assert "Latência" in md


def test_generate_report_saves_md_and_json(tmp_path: Path):
    results = [_make_result("code_fib", "code", 5.0, 3.0, "claude")]
    output = tmp_path / "report.md"

    generate_report(results, output)

    assert output.exists()
    json_path = output.with_suffix(".json")
    assert json_path.exists()

    import json
    data = json.loads(json_path.read_text())
    assert len(data) == 1
    assert data[0]["task_id"] == "code_fib"
    assert data[0]["winner"] == "claude"


def test_generate_report_correct_winner_in_summary():
    results = [
        _make_result("t1", "code", 5.0, 2.0, "claude"),
        _make_result("t2", "code", 4.0, 3.0, "claude"),
    ]
    md = generate_report(results)
    # código tem avg claude=4.5, gemma=2.5 → Claude vence
    assert "Claude" in md
