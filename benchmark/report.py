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

    for cat, data in categories.items():
        avg_c = sum(data["claude_scores"]) / len(data["claude_scores"])
        avg_g = sum(data["gemma_scores"]) / len(data["gemma_scores"])
        winner = "Claude" if avg_c > avg_g else ("Gemma" if avg_g > avg_c else "Empate")
        lines.append(f"| {cat} | {avg_c:.1f} | {avg_g:.1f} | {winner} |")

    all_task_claude = [r.judge_result.score_claude for r in results]
    all_task_gemma = [r.judge_result.score_gemma for r in results]
    grand_c = sum(all_task_claude) / len(all_task_claude)
    grand_g = sum(all_task_gemma) / len(all_task_gemma)
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
    for r in results:
        jr = r.judge_result
        w = jr.winner.capitalize()
        lines += [
            f"### {r.task.id}",
            f"**Categoria:** {r.task.category} | "
            f"**Claude:** {jr.score_claude} | "
            f"**Gemma:** {jr.score_gemma} | "
            f"**Vencedor:** {w}",
            "",
            jr.justification,
            "",
        ]

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
