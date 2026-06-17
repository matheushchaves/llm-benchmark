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
