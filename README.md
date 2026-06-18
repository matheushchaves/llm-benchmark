# LLM Benchmark — Claude vs Gemma

![Claude Sonnet](https://img.shields.io/badge/Claude%20Sonnet-4.8%2F5.0-brightgreen)
![Gemma 4 26B](https://img.shields.io/badge/Gemma%204%2026B-4.7%2F5.0-orange)
![Winner](https://img.shields.io/badge/winner-Claude%20%28por%200.1%29-blue)
![Tasks](https://img.shields.io/badge/tasks-18-lightgrey)
![Judge](https://img.shields.io/badge/judge-Gemini%202.5%20Flash-red)

Estudo comparativo entre **Claude Sonnet** (via Claude Code CLI) e **Gemma 4 26B** (via Ollama local) em quatro categorias de tarefas: código, raciocínio, sumarização e Q&A em português.

## Motivação

Ao usar Claude Code como ferramenta principal de desenvolvimento, surge a pergunta: **vale a pena usar um modelo local gratuito como o Gemma para economizar?** Este projeto quantifica essa troca com um benchmark reprodutível e um juiz LLM imparcial.

## Resultados — Juiz Neutro: Gemini 2.5 Flash

| Categoria | Claude | Gemma | Vencedor |
|-----------|--------|-------|----------|
| Código | 4.8 | 4.8 | Empate |
| Raciocínio | 4.8 | 4.8 | Empate |
| Sumarização | 5.0 | 4.2 | Claude |
| Q&A em PT | 4.8 | **5.0** | **Gemma** |
| **Total** | **4.8** | **4.7** | **Claude** |

**Latência média por task:**
- Claude CLI: ~10s
- Gemma (Ollama local): ~72s

### Interpretação

Com um juiz neutro (Gemini), a margem cai para **apenas 0.1 ponto** — um empate técnico em 3 de 4 categorias. O único diferencial real do Claude foi sumarização. Gemma venceu Q&A em português e empatou em código e raciocínio.

> **Nota metodológica:** em um run anterior com Claude como próprio juiz, a margem era de 0.4 pontos a favor de Claude — evidência direta de auto-favorecimento. O juiz neutro revelou uma competição muito mais equilibrada.

O resultado detalhado com justificativas do juiz está em [`results/20260618T135436.md`](results/20260618T135436.md).

## Arquitetura

```
main.py                  # CLI (argparse)
benchmark/
  runner.py              # Orquestra tasks em sequência; Claude + Gemma em paralelo por task
  judge.py               # Juiz LLM com randomização A/B para evitar position bias
  report.py              # Gera relatório .md e snapshot .json
  models/
    claude.py            # Chama `claude -p --output-format json` via stdin
    ollama.py            # Chama Ollama REST API via httpx
  types.py               # Dataclasses: Task, ModelResponse, JudgeResult, TaskResult
tasks/                   # Tasks em YAML por categoria
  code.yaml
  reasoning.yaml
  summarization.yaml
  qa_pt.yaml
results/                 # Snapshots de runs anteriores (.md + .json)
tests/                   # Testes com pytest + respx
```

### Decisões de design

**Por que `claude` CLI em vez do SDK Anthropic?**
O SDK exige `ANTHROPIC_API_KEY` como variável de ambiente. O Claude Code guarda a chave internamente — quem já usa Claude Code pagaria duas vezes. Usar `claude -p --output-format json` aproveita a sessão ativa sem custo adicional.

**Por que tarefas sequenciais?**
Gemma processa uma requisição por vez localmente. Executar todas as tasks em paralelo via `asyncio.gather` causava timeout. A solução: sequencial entre tasks, paralelo dentro de cada task (Claude + Gemma simultaneamente).

**Por que Gemini como juiz?**
Claude e Gemma são os competidores — usar um deles como juiz criaria conflito de interesse. O Gemini 2.5 Flash atua como árbitro neutro de terceiro. Comprovamos empiricamente que o Claude como auto-juiz infla a própria margem em 0.3 pontos.

**Por que randomização A/B no juiz?**
O juiz LLM tem position bias — tende a favorecer a resposta que aparece primeiro. Sortear qual modelo é "Resposta A" e qual é "Resposta B" a cada avaliação elimina esse viés sistematicamente.

**Por que prompt via stdin e não argumento CLI?**
Prompts longos (com as respostas dos dois modelos) passados como argumento de linha de comando causavam falha silenciosa no processo filho. Stdin não tem limite de tamanho.

## Requisitos

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — gerenciador de pacotes
- [Claude Code](https://claude.ai/code) — instalado e autenticado (`claude` disponível no PATH)
- [Ollama](https://ollama.com/) — rodando localmente com o modelo desejado
- **Gemini API key** — gratuita em [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

```bash
ollama pull gemma4:26b-mlx   # ou outro modelo
```

## Instalação

```bash
git clone https://github.com/matheushchaves/llm-benchmark
cd llm-benchmark
uv sync
cp .env.example .env
# edite .env e adicione sua GEMINI_API_KEY
```

## Uso

```bash
# Todas as categorias
uv run python main.py

# Categorias específicas
uv run python main.py --categories code reasoning

# Modelo Claude diferente
uv run python main.py --claude-model opus

# Arquivo de saída customizado
uv run python main.py --output results/meu_teste.md
```

O relatório é salvo em `results/` como `.md` (leitura humana) e `.json` (processamento programático).

## Testes

```bash
uv run pytest -v
```

## Adicionando tasks

Cada arquivo YAML em `tasks/` segue este formato:

```yaml
- id: minha_task
  category: code
  judge_hints: "Avaliar: correção, legibilidade, tratamento de edge cases"
  prompt: |
    Escreva uma função Python que ...
```

## Limitações do estudo

- **Amostra pequena**: 18 tasks não é suficiente para conclusões estatisticamente robustas
- **Viés do juiz**: Gemini 2.5 Flash é neutro entre os competidores, mas qualquer LLM tem preferências de estilo não totalmente elimináveis
- **Modelo fixo de Gemma**: testamos apenas `gemma4:26b-mlx`; outros tamanhos e quantizações darão resultados diferentes
- **Tarefas em português**: o benchmark tem viés para PT-BR; resultados podem diferir em inglês
- **Sem temperatura controlada**: não fixamos temperatura nas chamadas, o que adiciona variância

## Licença

MIT
