import asyncio
import json
import random
import re

from benchmark.types import JudgeResult, ModelResponse, Task

_MAX_JUDGE_RETRIES = 2

_JUDGE_PROMPT = """\
Você é um avaliador imparcial de modelos de linguagem.

Tarefa original: {prompt}

Critérios de avaliação: {hints}

Resposta A:
{response_a}

Resposta B:
{response_b}

Avalie cada resposta de 1 a 5 (1=muito ruim, 5=excelente) e declare um vencedor.

Responda SOMENTE com JSON válido, sem texto adicional, sem markdown fences, neste formato exato:
{{
  "score_a": <1-5>,
  "score_b": <1-5>,
  "winner": "<A|B|tie>",
  "justification": "<um parágrafo explicando os scores>"
}}"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1).strip()
    return json.loads(text)


async def _run_judge_cli(prompt: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        "claude", "-p", prompt,
        "--output-format", "json",
        "--no-session-persistence",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI (judge) failed: {stderr.decode().strip()}")
    data = json.loads(stdout.decode())
    if data.get("is_error"):
        raise RuntimeError(f"claude CLI (judge) returned error: {data}")
    return data["result"]


async def judge_responses(
    task: Task,
    claude_resp: ModelResponse,
    gemma_resp: ModelResponse,
) -> JudgeResult:
    a_is_claude = random.random() < 0.5
    resp_a = claude_resp if a_is_claude else gemma_resp
    resp_b = gemma_resp if a_is_claude else claude_resp

    prompt = _JUDGE_PROMPT.format(
        prompt=task.prompt,
        hints=task.judge_hints,
        response_a=resp_a.content,
        response_b=resp_b.content,
    )

    last_error: Exception | None = None
    last_raw: str = ""
    for _ in range(_MAX_JUDGE_RETRIES + 1):
        last_raw = await _run_judge_cli(prompt)
        try:
            data = _extract_json(last_raw)
            break
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
    else:
        raise ValueError(
            f"Judge returned non-JSON after {_MAX_JUDGE_RETRIES + 1} attempts. "
            f"Last response: {last_raw!r}"
        ) from last_error

    raw_winner = data["winner"]

    if a_is_claude:
        score_claude = data["score_a"]
        score_gemma = data["score_b"]
        if raw_winner == "A":
            winner = "claude"
        elif raw_winner == "B":
            winner = "gemma"
        elif raw_winner == "tie":
            winner = "tie"
        else:
            raise ValueError(f"Unexpected winner value from judge: {raw_winner!r}")
    else:
        score_claude = data["score_b"]
        score_gemma = data["score_a"]
        if raw_winner == "A":
            winner = "gemma"
        elif raw_winner == "B":
            winner = "claude"
        elif raw_winner == "tie":
            winner = "tie"
        else:
            raise ValueError(f"Unexpected winner value from judge: {raw_winner!r}")

    return JudgeResult(
        score_claude=score_claude,
        score_gemma=score_gemma,
        winner=winner,
        justification=data["justification"],
    )
