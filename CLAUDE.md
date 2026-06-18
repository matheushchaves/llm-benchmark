# CLAUDE.md

## Setup

```bash
uv sync
ollama pull gemma4:26b-mlx
```

Claude Code deve estar instalado e autenticado — o benchmark chama `claude -p` como subprocess.

## Rodar benchmark

```bash
uv run python main.py
uv run python main.py --categories code reasoning
```

## Testes

```bash
uv run pytest -v
```

## Estrutura relevante

- `benchmark/models/claude.py` — subprocess do `claude` CLI via stdin
- `benchmark/models/ollama.py` — httpx para Ollama REST; timeout 600s; modelo em `OLLAMA_MODEL`
- `benchmark/judge.py` — randomização A/B, retry, parse JSON robusto
- `benchmark/runner.py` — sequencial entre tasks, paralelo dentro de cada task
- `tasks/*.yaml` — adicione tasks aqui para expandir o benchmark

## Convenções

- Dataclasses imutáveis em `benchmark/types.py` — não adicione lógica lá
- Tasks YAML precisam dos campos: `id`, `category`, `judge_hints`, `prompt`
- Resultados vão para `results/` como `<timestamp>.md` e `<timestamp>.json`
