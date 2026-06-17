import asyncio
import json
import time

from benchmark.types import ModelResponse


async def _run_claude_cli(prompt: str, model: str) -> dict:
    proc = await asyncio.create_subprocess_exec(
        "claude", "-p", prompt,
        "--output-format", "json",
        "--model", model,
        "--no-session-persistence",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {stderr.decode().strip()}")
    return json.loads(stdout.decode())


async def call_claude(prompt: str, model: str) -> ModelResponse:
    start = time.perf_counter()
    data = await _run_claude_cli(prompt, model)
    latency_ms = (time.perf_counter() - start) * 1000

    if data.get("is_error"):
        raise RuntimeError(f"claude CLI returned error: {data}")

    usage = data.get("usage", {})
    return ModelResponse(
        content=data["result"],
        latency_ms=latency_ms,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        estimated_cost_usd=data.get("total_cost_usd", 0.0),
    )
