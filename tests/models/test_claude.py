from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchmark.models.claude import call_claude
from benchmark.types import ModelResponse


@pytest.mark.asyncio
async def test_call_claude_returns_model_response():
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="def fib(n): return n")]
    mock_message.usage.input_tokens = 50
    mock_message.usage.output_tokens = 20

    with patch("benchmark.models.claude.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=mock_message)

        result = await call_claude("write fibonacci", "claude-haiku-4-5-20251001")

    assert isinstance(result, ModelResponse)
    assert result.content == "def fib(n): return n"
    assert result.input_tokens == 50
    assert result.output_tokens == 20
    assert result.estimated_cost_usd > 0
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_call_claude_cost_is_positive_for_known_model():
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="answer")]
    mock_message.usage.input_tokens = 100
    mock_message.usage.output_tokens = 100

    with patch("benchmark.models.claude.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=mock_message)

        result = await call_claude("test", "claude-sonnet-4-6")

    assert result.estimated_cost_usd > 0
