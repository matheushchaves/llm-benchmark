import json
from unittest.mock import AsyncMock, patch

import pytest

from benchmark.models.claude import call_claude
from benchmark.types import ModelResponse

_MOCK_CLI_RESPONSE = {
    "type": "result",
    "is_error": False,
    "result": "def fib(n): return n",
    "duration_ms": 800,
    "total_cost_usd": 0.005,
    "usage": {"input_tokens": 50, "output_tokens": 20},
}


@pytest.mark.asyncio
async def test_call_claude_returns_model_response():
    with patch("benchmark.models.claude._run_claude_cli", new=AsyncMock(return_value=_MOCK_CLI_RESPONSE)):
        result = await call_claude("write fibonacci", "claude-haiku-4-5-20251001")

    assert isinstance(result, ModelResponse)
    assert result.content == "def fib(n): return n"
    assert result.input_tokens == 50
    assert result.output_tokens == 20
    assert result.estimated_cost_usd == 0.005
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_call_claude_uses_cost_from_cli():
    mock_response = {**_MOCK_CLI_RESPONSE, "total_cost_usd": 0.042}
    with patch("benchmark.models.claude._run_claude_cli", new=AsyncMock(return_value=mock_response)):
        result = await call_claude("test", "claude-sonnet-4-6")

    assert result.estimated_cost_usd == 0.042
