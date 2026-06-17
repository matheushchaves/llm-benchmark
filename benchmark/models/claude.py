import time

from anthropic import AsyncAnthropic

from benchmark.types import ModelResponse

_MODEL_COSTS: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"input": 0.80 / 1_000_000, "output": 4.00 / 1_000_000},
    "claude-sonnet-4-6": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
}
_DEFAULT_COST = {"input": 1.00 / 1_000_000, "output": 5.00 / 1_000_000}


async def call_claude(prompt: str, model: str) -> ModelResponse:
    client = AsyncAnthropic()
    start = time.perf_counter()
    response = await client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = (time.perf_counter() - start) * 1000

    costs = _MODEL_COSTS.get(model, _DEFAULT_COST)
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = input_tokens * costs["input"] + output_tokens * costs["output"]

    return ModelResponse(
        content=response.content[0].text,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=cost,
    )
