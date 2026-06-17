# LLM Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** CLI harness que roda tarefas em paralelo no Claude e no Gemma, avalia as respostas com um judge LLM e gera relatório Markdown + JSON.

**Architecture:** Adapters isolados por modelo → runner asyncio → judge (Claude Haiku) → report. Tasks declaradas em YAMLs por categoria, sem acoplamento com o runner.

**Tech Stack:** Python 3.12+, `uv`, `anthropic`, `httpx`, `pyyaml`, `rich`, `pytest`, `pytest-asyncio`, `respx`

## Global Constraints

- Python >= 3.12
- Gerenciador de pacotes: `uv` (não pip diretamente)
- Modelo Gemma fixo: `gemma4:26b-mlx` via `http://localhost:11434`
- Modelo judge fixo: `claude-haiku-4-5-20251001`
- Modelo Claude configurável via `--claude-model` (default: `claude-haiku-4-5-20251001`)
- Tudo async (asyncio), Claude e Gemma chamados em paralelo por task
- `results/` é gitignored — nunca commitar resultados
- Sem banco de dados, sem UI

---

## File Map

```
llm-benchmark/
├── benchmark/
│   ├── __init__.py          # vazio
│   ├── types.py             # dataclasses ModelResponse, Task, JudgeResult, TaskResult
│   ├── tasks.py             # load_tasks(categories) → list[Task]
│   ├── judge.py             # judge_responses(task, claude_resp, gemma_resp) → JudgeResult
│   ├── runner.py            # run_task, run_benchmark
│   ├── report.py            # generate_report(results, output_path) → str
│   └── models/
│       ├── __init__.py      # vazio
│       ├── claude.py        # call_claude(prompt, model) → ModelResponse
│       └── ollama.py        # call_ollama(prompt, base_url) → ModelResponse
├── tasks/
│   ├── code.yaml
│   ├── reasoning.yaml
│   ├── summarization.yaml
│   └── qa_pt.yaml
├── tests/
│   ├── conftest.py
│   ├── test_tasks.py
│   ├── test_judge.py
│   ├── test_runner.py
│   ├── test_report.py
│   └── models/
│       ├── test_claude.py
│       └── test_ollama.py
├── results/                 # gitignored
├── main.py
├── pyproject.toml
└── .env.example
```

---

## Task 1: Scaffold — pyproject.toml, types, estrutura de pastas

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `benchmark/__init__.py`
- Create: `benchmark/models/__init__.py`
- Create: `benchmark/types.py`
- Create: `tests/conftest.py`
- Create: `tests/models/` (pasta vazia com `__init__.py`)

**Interfaces:**
- Produces:
  - `ModelResponse(content: str, latency_ms: float, input_tokens: int, output_tokens: int, estimated_cost_usd: float)`
  - `Task(id: str, prompt: str, category: str, judge_hints: str)`
  - `JudgeResult(score_claude: float, score_gemma: float, winner: str, justification: str)`
  - `TaskResult(task: Task, claude_response: ModelResponse, gemma_response: ModelResponse, judge_result: JudgeResult)`

- [ ] **Step 1: Criar pyproject.toml**

```toml
[project]
name = "llm-benchmark"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "anthropic>=0.40.0",
    "httpx>=0.27.0",
    "pyyaml>=6.0",
    "rich>=13.0",
    "python-dotenv>=1.0",
]

[tool.uv]
dev-dependencies = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "respx>=0.21",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 2: Criar .env.example**

```env
ANTHROPIC_API_KEY=sk-ant-...
BENCHMARK_CLAUDE_MODEL=claude-haiku-4-5-20251001
OLLAMA_BASE_URL=http://localhost:11434
```

- [ ] **Step 3: Criar benchmark/types.py**

```python
from dataclasses import dataclass


@dataclass
class ModelResponse:
    content: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


@dataclass
class Task:
    id: str
    prompt: str
    category: str
    judge_hints: str


@dataclass
class JudgeResult:
    score_claude: float
    score_gemma: float
    winner: str  # "claude" | "gemma" | "tie"
    justification: str


@dataclass
class TaskResult:
    task: Task
    claude_response: ModelResponse
    gemma_response: ModelResponse
    judge_result: JudgeResult
```

- [ ] **Step 4: Criar arquivos vazios restantes**

```bash
touch benchmark/__init__.py
touch benchmark/models/__init__.py
mkdir -p tests/models
touch tests/__init__.py
touch tests/models/__init__.py
touch tests/conftest.py
```

- [ ] **Step 5: Instalar dependências**

```bash
uv sync
```

Expected: `Resolved X packages` sem erros.

- [ ] **Step 6: Verificar que types são importáveis**

```bash
uv run python -c "from benchmark.types import ModelResponse, Task, JudgeResult, TaskResult; print('OK')"
```

Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .env.example benchmark/ tests/
git commit -m "feat: scaffold project — types, pyproject, estrutura de pastas"
```

---

## Task 2: Task YAML files + loader

**Files:**
- Create: `tasks/code.yaml`
- Create: `tasks/reasoning.yaml`
- Create: `tasks/summarization.yaml`
- Create: `tasks/qa_pt.yaml`
- Create: `benchmark/tasks.py`
- Create: `tests/test_tasks.py`

**Interfaces:**
- Consumes: `Task` de `benchmark.types`
- Produces: `load_tasks(categories: list[str]) -> list[Task]`

- [ ] **Step 1: Escrever teste primeiro**

`tests/test_tasks.py`:

```python
from benchmark.tasks import load_tasks
from benchmark.types import Task


def test_load_tasks_code_returns_tasks():
    tasks = load_tasks(["code"])
    assert len(tasks) >= 3
    for t in tasks:
        assert isinstance(t, Task)
        assert t.category == "code"
        assert t.id
        assert t.prompt
        assert t.judge_hints


def test_load_tasks_multiple_categories():
    tasks = load_tasks(["code", "reasoning"])
    categories = {t.category for t in tasks}
    assert "code" in categories
    assert "reasoning" in categories


def test_load_tasks_all_categories():
    tasks = load_tasks(["code", "reasoning", "summarization", "qa_pt"])
    assert len(tasks) >= 12
```

- [ ] **Step 2: Rodar testes — confirmar FAIL**

```bash
uv run pytest tests/test_tasks.py -v
```

Expected: `ImportError` ou `FileNotFoundError`

- [ ] **Step 3: Criar benchmark/tasks.py**

```python
from pathlib import Path

import yaml

from benchmark.types import Task

TASKS_DIR = Path(__file__).parent.parent / "tasks"


def load_tasks(categories: list[str]) -> list[Task]:
    tasks = []
    for category in categories:
        path = TASKS_DIR / f"{category}.yaml"
        with open(path) as f:
            raw = yaml.safe_load(f)
        for item in raw:
            tasks.append(Task(
                id=item["id"],
                prompt=item["prompt"],
                category=item["category"],
                judge_hints=item.get("judge_hints", ""),
            ))
    return tasks
```

- [ ] **Step 4: Criar tasks/code.yaml**

```yaml
- id: code_fibonacci
  prompt: "Escreva uma função Python que retorna o n-ésimo número de Fibonacci usando memoização. Mostre um exemplo de uso."
  category: code
  judge_hints: "Verificar: uso correto de memoização (dict, @lru_cache ou similar), função correta, complexidade O(n), código limpo e legível."

- id: code_binary_search
  prompt: "Implemente busca binária em Python para uma lista ordenada de inteiros. Retorne o índice do elemento ou -1 se não encontrado."
  category: code
  judge_hints: "Verificar: lógica correta (mid, low, high), trata lista vazia, retorna -1 quando não encontra, complexidade O(log n)."

- id: code_palindrome
  prompt: "Escreva uma função Python que verifica se uma string é palíndromo, ignorando espaços e maiúsculas/minúsculas."
  category: code
  judge_hints: "Verificar: normalização (strip, lower), ignora espaços, lógica correta, trata string vazia."

- id: code_fizzbuzz
  prompt: "Escreva FizzBuzz em Python: para números de 1 a 100, imprima 'Fizz' para múltiplos de 3, 'Buzz' para múltiplos de 5, 'FizzBuzz' para múltiplos de ambos, e o número caso contrário."
  category: code
  judge_hints: "Verificar: ordem das condições (FizzBuzz antes de Fizz/Buzz), cobre todos os casos, código simples."

- id: code_flatten_list
  prompt: "Escreva uma função Python que achata uma lista aninhada de qualquer profundidade. Exemplo: [[1,[2,3]],[4]] deve retornar [1,2,3,4]."
  category: code
  judge_hints: "Verificar: recursão ou iteração correta, trata listas vazias, trata profundidade arbitrária, resultado correto."
```

- [ ] **Step 5: Criar tasks/reasoning.yaml**

```yaml
- id: reason_handshakes
  prompt: "Em uma festa com 10 pessoas, cada pessoa cumprimenta todas as outras exatamente uma vez. Quantos cumprimentos acontecem no total? Explique seu raciocínio passo a passo."
  category: reasoning
  judge_hints: "Resposta correta: 45 (combinação C(10,2) = 45). Verificar: raciocínio correto, explicação clara, chegou ao resultado certo."

- id: reason_trains
  prompt: "Dois trens partem simultaneamente de cidades que distam 300 km, se aproximando um do outro. O trem A viaja a 80 km/h e o trem B a 70 km/h. Em quanto tempo se encontrarão? Mostre os cálculos."
  category: reasoning
  judge_hints: "Resposta correta: 300/(80+70) = 2 horas. Verificar: cálculo correto da velocidade relativa, resultado em horas."

- id: reason_logic_ages
  prompt: "Ana é mais velha que Bruno. Carlos é mais novo que Ana mas mais velho que Bruno. Diana é mais velha que Ana. Ordene as 4 pessoas da mais velha para a mais nova."
  category: reasoning
  judge_hints: "Resposta correta: Diana > Ana > Carlos > Bruno. Verificar: ordem correta, raciocínio lógico claro."

- id: reason_sequence
  prompt: "Encontre o número que falta na sequência: 2, 6, 12, 20, 30, __, 56. Explique o padrão."
  category: reasoning
  judge_hints: "Resposta correta: 42. Padrão: n*(n+1) onde n=1,2,3,... Verificar: resposta correta e padrão identificado."

- id: reason_coins
  prompt: "Você tem 8 moedas idênticas visualmente, mas uma é mais pesada que as outras. Com uma balança de dois pratos e apenas 2 pesagens, como você identifica a moeda mais pesada? Descreva a estratégia."
  category: reasoning
  judge_hints: "Estratégia ótima: 1ª pesagem: 3 vs 3. Se pender, a mais pesada está no grupo de 3 (pesagem 2: 1 vs 1 dessas 3). Se equilibrar, a mais pesada é uma das 2 restantes (pesagem 2: 1 vs 1). Verificar: estratégia correta em exatamente 2 pesagens."
```

- [ ] **Step 6: Criar tasks/summarization.yaml**

```yaml
- id: summ_extract_json
  prompt: |
    Extraia as seguintes informações do texto abaixo e retorne como JSON com as chaves: nome, cargo, empresa, email, telefone.

    "Olá, meu nome é Roberto Ferreira e trabalho como Gerente de Projetos na TechBrasil Soluções LTDA. Pode entrar em contato pelo e-mail roberto.ferreira@techbrasil.com.br ou pelo telefone (11) 98765-4321."
  category: summarization
  judge_hints: "Verificar: JSON válido, todos os 5 campos presentes, valores corretos: nome=Roberto Ferreira, cargo=Gerente de Projetos, empresa=TechBrasil Soluções LTDA, email=roberto.ferreira@techbrasil.com.br, telefone=(11) 98765-4321."

- id: summ_bullet_points
  prompt: |
    Resuma o texto abaixo em até 4 bullets concisos em português:

    "A inteligência artificial generativa está transformando rapidamente o mercado de trabalho global. Estudos recentes indicam que até 30% das tarefas atuais podem ser automatizadas nos próximos 5 anos, especialmente em áreas como atendimento ao cliente, análise de dados e criação de conteúdo. Por outro lado, novas profissões estão emergindo, como engenheiros de prompt, especialistas em ética de IA e auditores de algoritmos. Especialistas recomendam que profissionais invistam em habilidades complementares à IA, como pensamento crítico, criatividade e inteligência emocional."
  category: summarization
  judge_hints: "Verificar: 4 bullets ou menos, captura ideias principais (automação, novas profissões, recomendações), conciso e em português correto."

- id: summ_sentiment
  prompt: |
    Analise o sentimento do comentário abaixo e classifique como POSITIVO, NEGATIVO ou NEUTRO. Justifique em uma frase.

    "O produto chegou no prazo, embalagem estava ok. Mas a qualidade do material não é o que eu esperava pelo preço. O atendimento quando reclamei foi demorado. Não sei se compraria de novo."
  category: summarization
  judge_hints: "Classificação esperada: NEGATIVO (ou MISTO tendendo a negativo). Verificar: classificação razoável, justificativa coerente com o texto."

- id: summ_titles
  prompt: |
    Gere 3 títulos alternativos para o artigo abaixo, do mais formal ao mais informal:

    "Pesquisadores da Universidade de São Paulo desenvolveram uma técnica que usa bactérias modificadas geneticamente para decompor plásticos em 48 horas, processo que naturalmente levaria centenas de anos."
  category: summarization
  judge_hints: "Verificar: exatamente 3 títulos, progressão clara de formal a informal, todos relevantes ao conteúdo do artigo."
```

- [ ] **Step 7: Criar tasks/qa_pt.yaml**

```yaml
- id: qa_capital
  prompt: "Qual é a capital do Brasil e por que ela foi construída no interior do país? Responda em português."
  category: qa_pt
  judge_hints: "Resposta correta: Brasília. Motivos: integração do interior, desenvolvimento do Centro-Oeste, decisão de JK (inaugurada em 1960). Verificar: capital correta, explicação histórica precisa, português fluente."

- id: qa_amazon
  prompt: "Explique em 3 parágrafos a importância da Floresta Amazônica para o clima global. Responda em português."
  category: qa_pt
  judge_hints: "Verificar: menciona regulação climática, ciclo de água/chuvas, sequestro de carbono, biodiversidade. 3 parágrafos distintos, factualmente correto, português claro."

- id: qa_formula1
  prompt: "Como funciona o sistema de pontuação atual da Fórmula 1? Quem detém o recorde de mais títulos mundiais de pilotos? Responda em português."
  category: qa_pt
  judge_hints: "Sistema: 25-18-15-12-10-8-6-4-2-1 para top 10 + 1 ponto volta mais rápida (se no top 10). Recordistas: Lewis Hamilton e Michael Schumacher (7 títulos cada). Verificar: pontuação correta, recordistas corretos, português fluente."

- id: qa_python_vs_js
  prompt: "Quais são as principais diferenças entre Python e JavaScript? Quando você escolheria cada um? Responda em português."
  category: qa_pt
  judge_hints: "Verificar: menciona tipagem dinâmica em ambos, usos típicos (Python: data science/backend; JS: web/frontend/Node), diferenças relevantes de sintaxe/ecossistema, comparação equilibrada, português correto."
```

- [ ] **Step 8: Rodar testes — confirmar PASS**

```bash
uv run pytest tests/test_tasks.py -v
```

Expected: 3 testes PASS

- [ ] **Step 9: Commit**

```bash
git add tasks/ benchmark/tasks.py tests/test_tasks.py
git commit -m "feat: task YAMLs (16 tasks) e loader"
```

---

## Task 3: Claude adapter

**Files:**
- Create: `benchmark/models/claude.py`
- Create: `tests/models/test_claude.py`

**Interfaces:**
- Consumes: `ModelResponse` de `benchmark.types`
- Produces: `async call_claude(prompt: str, model: str) -> ModelResponse`

- [ ] **Step 1: Escrever teste primeiro**

`tests/models/test_claude.py`:

```python
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
```

- [ ] **Step 2: Rodar testes — confirmar FAIL**

```bash
uv run pytest tests/models/test_claude.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Criar benchmark/models/claude.py**

```python
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
```

- [ ] **Step 4: Rodar testes — confirmar PASS**

```bash
uv run pytest tests/models/test_claude.py -v
```

Expected: 2 testes PASS

- [ ] **Step 5: Commit**

```bash
git add benchmark/models/claude.py tests/models/test_claude.py
git commit -m "feat: claude adapter com custo estimado por modelo"
```

---

## Task 4: Ollama adapter

**Files:**
- Create: `benchmark/models/ollama.py`
- Create: `tests/models/test_ollama.py`

**Interfaces:**
- Consumes: `ModelResponse` de `benchmark.types`
- Produces: `async call_ollama(prompt: str, base_url: str = "http://localhost:11434") -> ModelResponse`

- [ ] **Step 1: Escrever teste primeiro**

`tests/models/test_ollama.py`:

```python
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
```

- [ ] **Step 2: Rodar testes — confirmar FAIL**

```bash
uv run pytest tests/models/test_ollama.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Criar benchmark/models/ollama.py**

```python
import time

import httpx

from benchmark.types import ModelResponse

OLLAMA_MODEL = "gemma4:26b-mlx"


async def call_ollama(
    prompt: str,
    base_url: str = "http://localhost:11434",
) -> ModelResponse:
    async with httpx.AsyncClient(timeout=180.0) as client:
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
```

- [ ] **Step 4: Rodar testes — confirmar PASS**

```bash
uv run pytest tests/models/test_ollama.py -v
```

Expected: 2 testes PASS

- [ ] **Step 5: Commit**

```bash
git add benchmark/models/ollama.py tests/models/test_ollama.py
git commit -m "feat: ollama adapter (gemma4:26b-mlx, custo zero)"
```

---

## Task 5: Judge

**Files:**
- Create: `benchmark/judge.py`
- Create: `tests/test_judge.py`

**Interfaces:**
- Consumes: `Task`, `ModelResponse`, `JudgeResult` de `benchmark.types`; `AsyncAnthropic`
- Produces: `async judge_responses(task: Task, claude_resp: ModelResponse, gemma_resp: ModelResponse) -> JudgeResult`

- [ ] **Step 1: Escrever teste primeiro**

`tests/test_judge.py`:

```python
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
```

- [ ] **Step 2: Rodar testes — confirmar FAIL**

```bash
uv run pytest tests/test_judge.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Criar benchmark/judge.py**

```python
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
```

- [ ] **Step 4: Rodar testes — confirmar PASS**

```bash
uv run pytest tests/test_judge.py -v
```

Expected: 2 testes PASS

- [ ] **Step 5: Commit**

```bash
git add benchmark/judge.py tests/test_judge.py
git commit -m "feat: judge LLM com aleatorização A/B para evitar viés de posição"
```

---

## Task 6: Runner

**Files:**
- Create: `benchmark/runner.py`
- Create: `tests/test_runner.py`

**Interfaces:**
- Consumes: `call_claude`, `call_ollama`, `judge_responses`, `Task`, `TaskResult`
- Produces:
  - `async run_task(task: Task, claude_model: str) -> TaskResult`
  - `async run_benchmark(tasks: list[Task], claude_model: str) -> list[TaskResult]`

- [ ] **Step 1: Escrever teste primeiro**

`tests/test_runner.py`:

```python
from unittest.mock import AsyncMock, patch

import pytest

from benchmark.runner import run_benchmark, run_task
from benchmark.types import JudgeResult, ModelResponse, Task, TaskResult


@pytest.fixture
def sample_task() -> Task:
    return Task(id="t1", prompt="test", category="code", judge_hints="hints")


@pytest.fixture
def mock_claude_resp() -> ModelResponse:
    return ModelResponse("claude answer", 500.0, 10, 50, 0.001)


@pytest.fixture
def mock_gemma_resp() -> ModelResponse:
    return ModelResponse("gemma answer", 800.0, 0, 60, 0.0)


@pytest.fixture
def mock_judge_result() -> JudgeResult:
    return JudgeResult(5.0, 3.0, "claude", "claude was better")


@pytest.mark.asyncio
async def test_run_task_returns_task_result(
    sample_task, mock_claude_resp, mock_gemma_resp, mock_judge_result
):
    with (
        patch("benchmark.runner.call_claude", AsyncMock(return_value=mock_claude_resp)),
        patch("benchmark.runner.call_ollama", AsyncMock(return_value=mock_gemma_resp)),
        patch("benchmark.runner.judge_responses", AsyncMock(return_value=mock_judge_result)),
    ):
        result = await run_task(sample_task, "claude-haiku-4-5-20251001")

    assert isinstance(result, TaskResult)
    assert result.task == sample_task
    assert result.claude_response == mock_claude_resp
    assert result.gemma_response == mock_gemma_resp
    assert result.judge_result == mock_judge_result


@pytest.mark.asyncio
async def test_run_benchmark_returns_all_results(
    sample_task, mock_claude_resp, mock_gemma_resp, mock_judge_result
):
    tasks = [sample_task, sample_task]

    with (
        patch("benchmark.runner.call_claude", AsyncMock(return_value=mock_claude_resp)),
        patch("benchmark.runner.call_ollama", AsyncMock(return_value=mock_gemma_resp)),
        patch("benchmark.runner.judge_responses", AsyncMock(return_value=mock_judge_result)),
    ):
        results = await run_benchmark(tasks, "claude-haiku-4-5-20251001")

    assert len(results) == 2
    assert all(isinstance(r, TaskResult) for r in results)
```

- [ ] **Step 2: Rodar testes — confirmar FAIL**

```bash
uv run pytest tests/test_runner.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Criar benchmark/runner.py**

```python
import asyncio

from rich.console import Console

from benchmark.judge import judge_responses
from benchmark.models.claude import call_claude
from benchmark.models.ollama import call_ollama
from benchmark.types import Task, TaskResult

console = Console()


async def run_task(task: Task, claude_model: str) -> TaskResult:
    console.print(f"  [dim]→ {task.id}[/dim]", end=" ")
    claude_resp, gemma_resp = await asyncio.gather(
        call_claude(task.prompt, claude_model),
        call_ollama(task.prompt),
    )
    judge_result = await judge_responses(task, claude_resp, gemma_resp)
    winner = judge_result.winner.capitalize()
    console.print(f"[bold]{winner}[/bold] wins ({judge_result.score_claude} vs {judge_result.score_gemma})")
    return TaskResult(
        task=task,
        claude_response=claude_resp,
        gemma_response=gemma_resp,
        judge_result=judge_result,
    )


async def run_benchmark(tasks: list[Task], claude_model: str) -> list[TaskResult]:
    results = []
    for task in tasks:
        result = await run_task(task, claude_model)
        results.append(result)
    return results
```

- [ ] **Step 4: Rodar testes — confirmar PASS**

```bash
uv run pytest tests/test_runner.py -v
```

Expected: 2 testes PASS

- [ ] **Step 5: Commit**

```bash
git add benchmark/runner.py tests/test_runner.py
git commit -m "feat: runner asyncio com chamadas paralelas por task"
```

---

## Task 7: Report

**Files:**
- Create: `benchmark/report.py`
- Create: `tests/test_report.py`

**Interfaces:**
- Consumes: `TaskResult`, `JudgeResult`, `ModelResponse`, `Task`
- Produces: `generate_report(results: list[TaskResult], output_path: Path | None = None) -> str`

- [ ] **Step 1: Escrever teste primeiro**

`tests/test_report.py`:

```python
from pathlib import Path

import pytest

from benchmark.report import generate_report
from benchmark.types import JudgeResult, ModelResponse, Task, TaskResult


def _make_result(task_id: str, category: str, score_c: float, score_g: float, winner: str) -> TaskResult:
    task = Task(id=task_id, prompt="prompt", category=category, judge_hints="hints")
    claude_r = ModelResponse("claude answer", 500.0, 10, 50, 0.001)
    gemma_r = ModelResponse("gemma answer", 800.0, 0, 60, 0.0)
    judge = JudgeResult(score_c, score_g, winner, "justification text here")
    return TaskResult(task=task, claude_response=claude_r, gemma_response=gemma_r, judge_result=judge)


def test_generate_report_returns_markdown_string():
    results = [
        _make_result("code_fib", "code", 5.0, 3.0, "claude"),
        _make_result("reason_1", "reasoning", 3.0, 4.0, "gemma"),
    ]
    md = generate_report(results)

    assert "# Benchmark" in md
    assert "code" in md
    assert "reasoning" in md
    assert "Claude" in md
    assert "Gemma" in md
    assert "Latência" in md


def test_generate_report_saves_md_and_json(tmp_path: Path):
    results = [_make_result("code_fib", "code", 5.0, 3.0, "claude")]
    output = tmp_path / "report.md"

    generate_report(results, output)

    assert output.exists()
    json_path = output.with_suffix(".json")
    assert json_path.exists()

    import json
    data = json.loads(json_path.read_text())
    assert len(data) == 1
    assert data[0]["task_id"] == "code_fib"
    assert data[0]["winner"] == "claude"


def test_generate_report_correct_winner_in_summary():
    results = [
        _make_result("t1", "code", 5.0, 2.0, "claude"),
        _make_result("t2", "code", 4.0, 3.0, "claude"),
    ]
    md = generate_report(results)
    # código tem avg claude=4.5, gemma=2.5 → Claude vence
    assert "Claude" in md
```

- [ ] **Step 2: Rodar testes — confirmar FAIL**

```bash
uv run pytest tests/test_report.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Criar benchmark/report.py**

```python
import json
from datetime import datetime
from pathlib import Path

from benchmark.types import TaskResult


def generate_report(
    results: list[TaskResult],
    output_path: Path | None = None,
) -> str:
    timestamp = datetime.now().isoformat(timespec="seconds")

    categories: dict[str, dict] = {}
    for r in results:
        cat = r.task.category
        if cat not in categories:
            categories[cat] = {
                "claude_scores": [],
                "gemma_scores": [],
                "latency_claude": [],
                "latency_gemma": [],
            }
        categories[cat]["claude_scores"].append(r.judge_result.score_claude)
        categories[cat]["gemma_scores"].append(r.judge_result.score_gemma)
        categories[cat]["latency_claude"].append(r.claude_response.latency_ms)
        categories[cat]["latency_gemma"].append(r.gemma_response.latency_ms)

    lines = [f"# Benchmark — {timestamp}", ""]

    lines += ["## Resumo por Categoria", ""]
    lines += ["| Categoria | Claude Avg | Gemma Avg | Vencedor |"]
    lines += ["|-----------|-----------|-----------|---------|"]

    total_claude: list[float] = []
    total_gemma: list[float] = []
    for cat, data in categories.items():
        avg_c = sum(data["claude_scores"]) / len(data["claude_scores"])
        avg_g = sum(data["gemma_scores"]) / len(data["gemma_scores"])
        total_claude.append(avg_c)
        total_gemma.append(avg_g)
        winner = "Claude" if avg_c > avg_g else ("Gemma" if avg_g > avg_c else "Empate")
        lines.append(f"| {cat} | {avg_c:.1f} | {avg_g:.1f} | {winner} |")

    grand_c = sum(total_claude) / len(total_claude)
    grand_g = sum(total_gemma) / len(total_gemma)
    grand_w = "Claude" if grand_c > grand_g else ("Gemma" if grand_g > grand_c else "Empate")
    lines.append(f"| **Total** | **{grand_c:.1f}** | **{grand_g:.1f}** | **{grand_w}** |")

    all_c_lat = [r.claude_response.latency_ms for r in results]
    all_g_lat = [r.gemma_response.latency_ms for r in results]
    lines += ["", "## Latência Média (ms/task)", ""]
    lines += ["| Modelo | ms/task |"]
    lines += ["|--------|---------|"]
    lines.append(f"| Claude | {sum(all_c_lat)/len(all_c_lat):.0f} |")
    lines.append(f"| Gemma  | {sum(all_g_lat)/len(all_g_lat):.0f} |")

    lines += ["", "## Detalhes por Task", ""]
    lines += ["| ID | Categoria | Claude | Gemma | Vencedor | Justificativa |"]
    lines += ["|----|-----------|--------|-------|----------|---------------|"]
    for r in results:
        jr = r.judge_result
        w = jr.winner.capitalize()
        just = (jr.justification[:80] + "...").replace("|", "/")
        lines.append(f"| {r.task.id} | {r.task.category} | {jr.score_claude} | {jr.score_gemma} | {w} | {just} |")

    md = "\n".join(lines)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md, encoding="utf-8")

        raw = [
            {
                "task_id": r.task.id,
                "category": r.task.category,
                "prompt": r.task.prompt,
                "claude_response": r.claude_response.content,
                "gemma_response": r.gemma_response.content,
                "claude_latency_ms": r.claude_response.latency_ms,
                "gemma_latency_ms": r.gemma_response.latency_ms,
                "claude_tokens": r.claude_response.output_tokens,
                "gemma_tokens": r.gemma_response.output_tokens,
                "claude_cost_usd": r.claude_response.estimated_cost_usd,
                "score_claude": r.judge_result.score_claude,
                "score_gemma": r.judge_result.score_gemma,
                "winner": r.judge_result.winner,
                "justification": r.judge_result.justification,
            }
            for r in results
        ]
        output_path.with_suffix(".json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return md
```

- [ ] **Step 4: Rodar testes — confirmar PASS**

```bash
uv run pytest tests/test_report.py -v
```

Expected: 3 testes PASS

- [ ] **Step 5: Commit**

```bash
git add benchmark/report.py tests/test_report.py
git commit -m "feat: report — Markdown + JSON com resumo por categoria e detalhes por task"
```

---

## Task 8: CLI main.py + suite completa de testes

**Files:**
- Create: `main.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: `load_tasks`, `run_benchmark`, `generate_report`
- Produces: entrypoint `uv run main.py [--categories ...] [--claude-model ...] [--output ...]`

- [ ] **Step 1: Escrever teste de parsing de args**

`tests/test_cli.py`:

```python
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def _import_parse_args():
    import importlib
    import main as m
    return m.parse_args


def test_parse_args_defaults():
    with patch("sys.argv", ["main.py"]):
        import main
        args = main.parse_args()
    assert args.categories == ["code", "reasoning", "summarization", "qa_pt"]
    assert args.claude_model == "claude-haiku-4-5-20251001"
    assert args.output is None


def test_parse_args_custom_categories():
    with patch("sys.argv", ["main.py", "--categories", "code", "reasoning"]):
        import main
        args = main.parse_args()
    assert args.categories == ["code", "reasoning"]


def test_parse_args_custom_model():
    with patch("sys.argv", ["main.py", "--claude-model", "claude-sonnet-4-6"]):
        import main
        args = main.parse_args()
    assert args.claude_model == "claude-sonnet-4-6"
```

- [ ] **Step 2: Rodar testes — confirmar FAIL**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: `ModuleNotFoundError: No module named 'main'`

- [ ] **Step 3: Criar main.py**

```python
import argparse
import asyncio
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from benchmark.report import generate_report
from benchmark.runner import run_benchmark
from benchmark.tasks import load_tasks

load_dotenv()

console = Console()

ALL_CATEGORIES = ["code", "reasoning", "summarization", "qa_pt"]
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM Benchmark: Claude vs Gemma")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=ALL_CATEGORIES,
        choices=ALL_CATEGORIES,
        metavar="CATEGORY",
        help=f"Categorias a rodar. Opções: {', '.join(ALL_CATEGORIES)}",
    )
    parser.add_argument(
        "--claude-model",
        default=DEFAULT_MODEL,
        help=f"Modelo Claude a usar (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Caminho do relatório .md (default: results/<timestamp>.md)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output = args.output or Path(f"results/{timestamp}.md")

    console.print("\n[bold]LLM Benchmark[/bold] — Claude vs Gemma\n")
    console.print(f"  Claude model : [cyan]{args.claude_model}[/cyan]")
    console.print(f"  Gemma model  : [green]gemma4:26b-mlx[/green]")
    console.print(f"  Categories   : {', '.join(args.categories)}")
    console.print(f"  Output       : [dim]{output}[/dim]\n")

    tasks = load_tasks(args.categories)
    console.print(f"[bold]{len(tasks)} tasks[/bold] carregadas\n")

    results = await run_benchmark(tasks, args.claude_model)

    md = generate_report(results, output)
    console.print("\n" + md)
    console.print(f"\n[bold green]✓[/bold green] Relatório salvo em [cyan]{output}[/cyan]")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: Rodar testes de CLI — confirmar PASS**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: 3 testes PASS

- [ ] **Step 5: Rodar suite completa**

```bash
uv run pytest -v
```

Expected: todos os testes PASS (mínimo 14 testes)

- [ ] **Step 6: Verificar que main.py é executável (dry run — sem API key real)**

```bash
uv run main.py --help
```

Expected: exibe help com `--categories`, `--claude-model`, `--output`

- [ ] **Step 7: Commit final**

```bash
git add main.py tests/test_cli.py
git commit -m "feat: CLI main.py com argparse, rich output e entrypoint completo"
```

---

## Checklist Final

- [ ] `uv run pytest -v` → todos PASS
- [ ] `uv run main.py --help` → exibe usage sem erros
- [ ] `results/` está no `.gitignore`
- [ ] `.env` não commitado (apenas `.env.example`)
