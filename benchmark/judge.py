import json
import random

from anthropic import AsyncAnthropic

from benchmark.types import JudgeResult, ModelResponse, Task

JUDGE_MODEL = "claude-haiku-4-5-20251001"

_JUDGE_PROMPT = """\
Você é um avaliador imparcial de modelos de linguagem.

Tarefa original: {prompt}

Critérios de avaliação: {hints}

Resposta A:
{response_a}

Resposta B:
{response_b}

Avalie cada resposta de 1 a 5 (1=muito ruim, 5=excelente) e declare um vencedor.

Responda SOMENTE com JSON válido neste formato exato:
{{
  "score_a": <1-5>,
  "score_b": <1-5>,
  "winner": "<A|B|tie>",
  "justification": "<um parágrafo explicando os scores>"
}}"""


async def judge_responses(
    task: Task,
    claude_resp: ModelResponse,
    gemma_resp: ModelResponse,
) -> JudgeResult:
    client = AsyncAnthropic()

    a_is_claude = random.random() < 0.5
    resp_a = claude_resp if a_is_claude else gemma_resp
    resp_b = gemma_resp if a_is_claude else claude_resp

    prompt = _JUDGE_PROMPT.format(
        prompt=task.prompt,
        hints=task.judge_hints,
        response_a=resp_a.content,
        response_b=resp_b.content,
    )

    response = await client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    data = json.loads(response.content[0].text)
    raw_winner = data["winner"]

    if a_is_claude:
        score_claude = data["score_a"]
        score_gemma = data["score_b"]
        winner = "claude" if raw_winner == "A" else ("gemma" if raw_winner == "B" else "tie")
    else:
        score_claude = data["score_b"]
        score_gemma = data["score_a"]
        winner = "gemma" if raw_winner == "A" else ("claude" if raw_winner == "B" else "tie")

    return JudgeResult(
        score_claude=score_claude,
        score_gemma=score_gemma,
        winner=winner,
        justification=data["justification"],
    )
