import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchmark.judge import judge_responses
from benchmark.types import JudgeResult, ModelResponse, Task


@pytest.fixture
def sample_task() -> Task:
    return Task(
        id="t1",
        prompt="Escreva fibonacci",
        category="code",
        judge_hints="Verificar uso de memoização",
    )


@pytest.fixture
def claude_resp() -> ModelResponse:
    return ModelResponse(
        content="def fib(n): ...",
        latency_ms=500,
        input_tokens=10,
        output_tokens=50,
        estimated_cost_usd=0.001,
    )


@pytest.fixture
def gemma_resp() -> ModelResponse:
    return ModelResponse(
        content="fibonacci solution here",
        latency_ms=800,
        input_tokens=0,
        output_tokens=60,
        estimated_cost_usd=0.0,
    )


@pytest.mark.asyncio
async def test_judge_claude_wins(sample_task, claude_resp, gemma_resp):
    judge_json = json.dumps({
        "score_a": 5,
        "score_b": 3,
        "winner": "A",
        "justification": "A usou lru_cache corretamente",
    })
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=judge_json)]

    with (
        patch("benchmark.judge.AsyncAnthropic") as mock_cls,
        patch("benchmark.judge.random.random", return_value=0.0),  # a_is_claude=True
    ):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=mock_message)

        result = await judge_responses(sample_task, claude_resp, gemma_resp)

    assert isinstance(result, JudgeResult)
    assert result.score_claude == 5
    assert result.score_gemma == 3
    assert result.winner == "claude"
    assert "lru_cache" in result.justification


@pytest.mark.asyncio
async def test_judge_gemma_wins_with_swapped_ab(sample_task, claude_resp, gemma_resp):
    # random.random >= 0.5 → a_is_claude=False → gemma is A
    judge_json = json.dumps({
        "score_a": 4,
        "score_b": 2,
        "winner": "A",
        "justification": "A foi mais claro",
    })
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=judge_json)]

    with (
        patch("benchmark.judge.AsyncAnthropic") as mock_cls,
        patch("benchmark.judge.random.random", return_value=0.9),  # a_is_claude=False
    ):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=mock_message)

        result = await judge_responses(sample_task, claude_resp, gemma_resp)

    # gemma foi A, venceu → winner deve ser "gemma"
    assert result.score_gemma == 4
    assert result.score_claude == 2
    assert result.winner == "gemma"
