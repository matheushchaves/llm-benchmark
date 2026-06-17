from unittest.mock import AsyncMock, patch

import pytest

from benchmark.runner import run_benchmark, run_task
from benchmark.types import JudgeResult, ModelResponse, Task, TaskResult


@pytest.fixture
def sample_task() -> Task:
    return Task(id="t1", prompt="test", category="code", judge_hints="hints")


@pytest.fixture
def mock_claude_resp() -> ModelResponse:
    return ModelResponse("claude answer", 500.0, 10, 50, 0.001)


@pytest.fixture
def mock_gemma_resp() -> ModelResponse:
    return ModelResponse("gemma answer", 800.0, 0, 60, 0.0)


@pytest.fixture
def mock_judge_result() -> JudgeResult:
    return JudgeResult(5.0, 3.0, "claude", "claude was better")


@pytest.mark.asyncio
async def test_run_task_returns_task_result(
    sample_task, mock_claude_resp, mock_gemma_resp, mock_judge_result
):
    with (
        patch("benchmark.runner.call_claude", AsyncMock(return_value=mock_claude_resp)),
        patch("benchmark.runner.call_ollama", AsyncMock(return_value=mock_gemma_resp)),
        patch("benchmark.runner.judge_responses", AsyncMock(return_value=mock_judge_result)),
    ):
        result = await run_task(sample_task, "claude-haiku-4-5-20251001")

    assert isinstance(result, TaskResult)
    assert result.task == sample_task
    assert result.claude_response == mock_claude_resp
    assert result.gemma_response == mock_gemma_resp
    assert result.judge_result == mock_judge_result


@pytest.mark.asyncio
async def test_run_benchmark_returns_all_results(
    sample_task, mock_claude_resp, mock_gemma_resp, mock_judge_result
):
    tasks = [sample_task, sample_task]

    with (
        patch("benchmark.runner.call_claude", AsyncMock(return_value=mock_claude_resp)),
        patch("benchmark.runner.call_ollama", AsyncMock(return_value=mock_gemma_resp)),
        patch("benchmark.runner.judge_responses", AsyncMock(return_value=mock_judge_result)),
    ):
        results = await run_benchmark(tasks, "claude-haiku-4-5-20251001")

    assert len(results) == 2
    assert all(isinstance(r, TaskResult) for r in results)
