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
