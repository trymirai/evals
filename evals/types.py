from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InternalEvalRecord:
    id: str
    question: str
    answer: str
    options: list[str] | None = None
    answer_index: int | None = None
    reasoning: str | None = None
    category: str | None = None
    metadata: dict[str, str] | None = None


@dataclass(frozen=True)
class DatasetMetadata:
    lalamo_version: str
    name: str
    repo: str
    splits: tuple[str, ...]
    schema_version: str
    total_examples: dict[str, int]


@dataclass(frozen=True)
class PromptMessage:
    role: str  # "system", "user", "assistant"
    content: str


@dataclass(frozen=True)
class EvalPrompt:
    id: str
    messages: list[PromptMessage]
    category: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class InferenceOutput:
    id: str
    response: str
    chain_of_thought: str | None = None


@dataclass(frozen=True)
class PredictionRecord:
    id: str
    model_output: str
    extracted_answer: str | None = None
    is_correct: bool | None = None
    metadata: dict[str, str] | None = None


@dataclass(frozen=True)
class BenchmarkMetrics:
    eval_name: str
    model_name: str
    split: str
    overall_accuracy: float
    total_examples: int
    correct: int
    incorrect: int
    category_metrics: dict[str, float] | None = None
    custom_metrics: dict[str, float] | None = None
