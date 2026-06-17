import argparse
import asyncio
from datetime import datetime
from pathlib import Path

from rich.console import Console

from benchmark.report import generate_report
from benchmark.runner import run_benchmark
from benchmark.tasks import load_tasks

console = Console()

ALL_CATEGORIES = ["code", "reasoning", "summarization", "qa_pt"]
DEFAULT_MODEL = "sonnet"


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
        help=f"Modelo Claude a usar via CLI (default: {DEFAULT_MODEL})",
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
