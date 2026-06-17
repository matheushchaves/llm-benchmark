# Design: LLM Benchmark — Claude vs Gemma

**Data:** 2026-06-17  
**Status:** Aprovado

---

## Objetivo

Harness CLI em Python para comparar respostas do Claude (Anthropic API) e do Gemma (`gemma4:26b-mlx` via Ollama local) em quatro categorias de tarefas, com avaliação automática de qualidade por um modelo judge.

---

## Estrutura de Pastas

```
llm-benchmark/
├── tasks/
│   ├── code.yaml
│   ├── reasoning.yaml
│   ├── summarization.yaml
│   └── qa_pt.yaml
├── benchmark/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── claude.py
│   │   └── ollama.py
│   ├── runner.py
│   ├── judge.py
│   └── report.py
├── results/              # gitignored
├── main.py
├── pyproject.toml
└── .env.example
```

---

## Categorias de Tarefas

Cada categoria tem seu YAML com 3–5 tasks. Formato de cada task:

```yaml
- id: code_fibonacci
  prompt: "Escreva uma função Python que retorna o n-ésimo Fibonacci com memoização."
  category: code
  judge_hints: "Verificar: uso de cache/dict ou @lru_cache, complexidade O(n), código correto e legível."
```

Campos:
- `id` — slug único
- `prompt` — enviado diretamente ao modelo
- `category` — `code | reasoning | summarization | qa_pt`
- `judge_hints` — contexto adicional para o judge avaliar com critérios da categoria

---

## Arquitetura e Fluxo

```
main.py
  └── runner.py
        ├── carrega YAMLs das categorias selecionadas
        ├── para cada task, chama claude.py e ollama.py em paralelo (asyncio)
        └── passa ambas as respostas para judge.py
              └── retorna scores + justificativa
  └── report.py
        ├── salva results/<timestamp>.json
        └── salva results/<timestamp>.md
```

### Adapters

**`claude.py`**
- SDK: `anthropic`
- Modelo: configurável via `--claude-model` (default: `claude-haiku-4-5-20251001`)
- Captura: `latency_ms`, `input_tokens`, `output_tokens`, `estimated_cost_usd`

**`ollama.py`**
- HTTP direto: `POST http://localhost:11434/api/chat`
- Modelo: `gemma4:26b-mlx` (fixo)
- Captura: `latency_ms`, `tokens_generated` (via campo `eval_count`)
- Custo: R$0 (local), registrado como `0.0`

---

## Judge

- Modelo: `claude-haiku-4-5-20251001` (custo baixo)
- As respostas A/B são embaralhadas aleatoriamente por run (evita viés de posição)
- O judge recebe: prompt original, `resposta_A`, `resposta_B`, `judge_hints`
- Retorna JSON estruturado via `response_format` / parsing:

```json
{
  "score_a": 4,
  "score_b": 3,
  "winner": "A",
  "justification": "Resposta A usou @lru_cache corretamente..."
}
```

- Scores de 1–5 (1=muito ruim, 5=excelente)
- O campo `winner` usa a letra embaralhada; o report resolve pra nome do modelo real

---

## CLI

```bash
# rodar todas as categorias
uv run main.py

# filtrar categorias
uv run main.py --categories code reasoning

# escolher modelo claude específico
uv run main.py --claude-model claude-sonnet-4-6

# salvar relatório em arquivo específico
uv run main.py --output results/meu-teste.md
```

---

## Relatório Markdown

```markdown
# Benchmark — 2026-06-17T15:30:00

## Resumo por Categoria
| Categoria     | Claude Avg | Gemma Avg | Vencedor  |
|---------------|-----------|-----------|-----------|
| code          | 4.2       | 3.8       | Claude    |
| reasoning     | 3.5       | 4.0       | Gemma     |
| summarization | 4.5       | 3.2       | Claude    |
| qa_pt         | 4.0       | 3.6       | Claude    |
| **Total**     | **4.1**   | **3.7**   | **Claude**|

## Latência Média
| Modelo | ms/task |
|--------|---------|
| Claude | 820     |
| Gemma  | 1240    |

## Detalhes por Task
[tabela com id, categoria, scores individuais, winner, latência, justificativa]
```

---

## Dependências

- `anthropic` — SDK Claude
- `httpx` — chamadas async ao Ollama
- `pyyaml` — leitura das tasks
- `rich` — output no terminal durante execução
- `python-dotenv` — carrega `ANTHROPIC_API_KEY` do `.env`

---

## Variáveis de Ambiente

```env
ANTHROPIC_API_KEY=sk-...
BENCHMARK_CLAUDE_MODEL=claude-haiku-4-5-20251001  # opcional
OLLAMA_BASE_URL=http://localhost:11434             # opcional
```

---

## O que está fora do escopo

- Interface web
- Suporte a outros modelos Ollama além do gemma4:26b-mlx
- Persistência em banco de dados
- CI/CD automatizado
