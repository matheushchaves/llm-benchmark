import json
import os
import random
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

from benchmark.types import JudgeResult, ModelResponse, Task

load_dotenv(override=True)

_MAX_JUDGE_RETRIES = 2
_JUDGE_MODEL = "gemini-2.5-flash"

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


def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY não encontrada. "
            "Configure em .env ou como variável de ambiente. "
            "Obtenha gratuitamente em https://aistudio.google.com/apikey"
        )
    # Remove GOOGLE_API_KEY do processo para evitar conflito com o shell
    os.environ.pop("GOOGLE_API_KEY", None)
    return genai.Client(api_key=api_key)


def _extract_json(text: str) -> dict:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1).strip()
    return json.loads(text)


async def _run_judge(prompt: str) -> str:
    client = _get_client()
    response = await client.aio.models.generate_content(
        model=_JUDGE_MODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.0,
        ),
    )
    return response.text


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
        last_raw = await _run_judge(prompt)
        try:
            data = _extract_json(last_raw)
            break
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
    else:
        raise ValueError(
            f"Gemini judge retornou JSON inválido após {_MAX_JUDGE_RETRIES + 1} tentativas. "
            f"Última resposta: {last_raw!r}"
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
            raise ValueError(f"Valor inesperado de winner do juiz: {raw_winner!r}")
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
            raise ValueError(f"Valor inesperado de winner do juiz: {raw_winner!r}")

    return JudgeResult(
        score_claude=score_claude,
        score_gemma=score_gemma,
        winner=winner,
        justification=data["justification"],
    )
