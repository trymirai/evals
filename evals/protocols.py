from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from evals.types import (
    BenchmarkMetrics,
    EvalPrompt,
    InternalEvalRecord,
    PredictionRecord,
)


@dataclass(frozen=True)
class EvalAdapter(ABC):
    @abstractmethod
    def convert_record(self, record: dict) -> InternalEvalRecord:
        """Convert single HuggingFace record to internal format."""
        ...

    @abstractmethod
    def convert_split(self, parquet_path: Path) -> list[InternalEvalRecord]:
        """Convert entire HuggingFace split to internal format."""
        ...

    @abstractmethod
    def format_prompts(
        self,
        records: list[InternalEvalRecord],
        few_shot_source: list[InternalEvalRecord] | None = None,
        num_few_shot: int = 5,
    ) -> list[EvalPrompt]:
        """Generate inference prompts from internal records."""
        ...

    @abstractmethod
    def prepare_for_benchmark(
        self,
        predictions: list[PredictionRecord],
        ground_truth: list[InternalEvalRecord],
        output_dir: Path,
    ) -> Path:
        """Convert predictions to format expected by official eval code."""
        ...

    @abstractmethod
    def run_benchmark(
        self,
        prepared_data_path: Path,
        eval_name: str,
        model_name: str,
        split: str,
    ) -> BenchmarkMetrics:
        """Run official benchmark code and return metrics."""
        ...
