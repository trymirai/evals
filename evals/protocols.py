from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from evals.types import (
    BenchmarkMetrics,
    DatasetLoadConfig,
    EvalPrompt,
    InferenceConfig,
    InferenceOutput,
    InternalEvalRecord,
)


@dataclass(frozen=True)
class EvalAdapter(ABC):
    @classmethod
    @abstractmethod
    def download_split(
        cls,
        repo_id: str,
        split: str,
        temp_dir: Path,
    ) -> list[InternalEvalRecord]:
        """Download and convert a dataset split."""
        ...

    @abstractmethod
    def convert_record(self, record: dict) -> InternalEvalRecord:
        """Convert single HuggingFace record to internal format."""
        ...

    @abstractmethod
    def convert_split(self, parquet_path: Path) -> list[InternalEvalRecord]:
        """Convert entire HuggingFace split to internal format."""
        ...

    @abstractmethod
    def get_benchmark_split(self) -> str:
        """Return the split name to use for benchmarking."""
        ...

    @abstractmethod
    def get_loading_config(self, limit: int | None) -> list[DatasetLoadConfig]:
        """Return list of dataset loading configs."""
        ...

    @abstractmethod
    def get_inference_config(self) -> InferenceConfig:
        """Return inference parameters from reference implementation. """
        ...

    @abstractmethod
    def format_prompts(
        self,
        datasets: dict[str, list[InternalEvalRecord]],
    ) -> list[EvalPrompt]:
        """Generate inference prompts from loaded datasets. """
        ...

    @abstractmethod
    def prepare_for_benchmark(
        self,
        predictions: list["InferenceOutput"],
        output_dir: Path,
    ) -> Path:
        """Convert predictions to format expected by official eval code. """
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
