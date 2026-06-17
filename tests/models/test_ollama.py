import httpx
import pytest
import respx

from benchmark.models.ollama import call_ollama
from benchmark.types import ModelResponse


@pytest.mark.asyncio
async def test_call_ollama_returns_model_response():
    mock_body = {
        "message": {"content": "Here is the fibonacci solution..."},
        "eval_count": 150,
        "prompt_eval_count": 30,
    }

    with respx.mock:
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, json=mock_body)
        )
        result = await call_ollama("write fibonacci")

    assert isinstance(result, ModelResponse)
    assert result.content == "Here is the fibonacci solution..."
    assert result.output_tokens == 150
    assert result.input_tokens == 30
    assert result.estimated_cost_usd == 0.0
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_call_ollama_custom_base_url():
    mock_body = {
        "message": {"content": "ok"},
        "eval_count": 10,
        "prompt_eval_count": 5,
    }

    with respx.mock:
        respx.post("http://localhost:9999/api/chat").mock(
            return_value=httpx.Response(200, json=mock_body)
        )
        result = await call_ollama("test", base_url="http://localhost:9999")

    assert result.content == "ok"
