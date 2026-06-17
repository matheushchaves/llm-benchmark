import time

import httpx

from benchmark.types import ModelResponse

OLLAMA_MODEL = "gemma4:26b-mlx"


async def call_ollama(
    prompt: str,
    base_url: str = "http://localhost:11434",
) -> ModelResponse:
    async with httpx.AsyncClient(timeout=600.0) as client:
        start = time.perf_counter()
        response = await client.post(
            f"{base_url}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
        )
        response.raise_for_status()
        latency_ms = (time.perf_counter() - start) * 1000

    data = response.json()
    return ModelResponse(
        content=data["message"]["content"],
        latency_ms=latency_ms,
        input_tokens=data.get("prompt_eval_count", 0),
        output_tokens=data.get("eval_count", 0),
        estimated_cost_usd=0.0,
    )
