import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import nltk
import pyarrow.parquet as pq

from evals.base_adapters import HFDatasetsAdapter
from evals.types import (
    BenchmarkMetrics,
    DatasetLoadConfig,
    EvalPrompt,
    InternalEvalRecord,
    MessageRole,
    PredictionRecord,
    PromptMessage,
)


def _ensure_nltk_data() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


@dataclass(frozen=True)
class IFEvalAdapter(HFDatasetsAdapter):
    def get_benchmark_split(self) -> str:
        return "train"

    def get_loading_config(self, limit: int | None) -> list[DatasetLoadConfig]:
        return [
            DatasetLoadConfig(split="train", limit=limit),
        ]

    def convert_record(self, record: dict) -> InternalEvalRecord:
        return InternalEvalRecord(
            id=str(record["key"]),
            question=record["prompt"],
            answer=None,  # IFEval is constraint-based, no "correct answer"
            metadata={
                "instruction_id_list": json.dumps(record["instruction_id_list"]),
                "kwargs": json.dumps(record["kwargs"]),
            },
        )

    def convert_split(self, parquet_path: Path) -> list[InternalEvalRecord]:
        table = pq.read_table(parquet_path)
        records = table.to_pydict()

        if not records:
            return []

        num_rows = len(next(iter(records.values())))
        return [self.convert_record({key: records[key][i] for key in records}) for i in range(num_rows)]

    def format_prompts(
        self,
        datasets: dict[str, list[InternalEvalRecord]],
    ) -> list[EvalPrompt]:
        records = datasets["train"]
        return [
            EvalPrompt(
                id=record.id,
                messages=[
                    PromptMessage(role=MessageRole.USER, content=record.question),
                ],
            )
            for record in records
        ]

    def prepare_for_benchmark(
        self,
        predictions: list[PredictionRecord],
        ground_truth: list[InternalEvalRecord],
        output_dir: Path,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create input_data.jsonl (original dataset format)
        input_data_path = output_dir / "input_data.jsonl"
        with open(input_data_path, "w") as f:
            for gt in ground_truth:
                metadata = json.loads(gt.metadata) if isinstance(gt.metadata, str) else gt.metadata

                # Filter kwargs to remove null values (vendored code expects only non-null fields)
                kwargs_list = json.loads(metadata["kwargs"])
                filtered_kwargs = [
                    {k: v for k, v in kwargs_dict.items() if v is not None} for kwargs_dict in kwargs_list
                ]

                entry = {
                    "key": int(gt.id),
                    "prompt": gt.question,
                    "instruction_id_list": json.loads(metadata["instruction_id_list"]),
                    "kwargs": filtered_kwargs,
                }
                f.write(json.dumps(entry) + "\n")

        # Create input_response_data.jsonl (prompt + model response)
        input_response_path = output_dir / "input_response_data.jsonl"
        with open(input_response_path, "w") as f:
            for pred, gt in zip(predictions, ground_truth, strict=True):
                if pred.id != gt.id:
                    raise ValueError(
                        f"ID mismatch: prediction.id={pred.id!r} != ground_truth.id={gt.id!r}. "
                        "Predictions and ground truth must be in the same order with matching IDs.",
                    )

                entry = {
                    "prompt": gt.question,
                    "response": pred.model_output,
                }
                f.write(json.dumps(entry) + "\n")

        return output_dir

    def run_benchmark(
        self,
        prepared_data_path: Path,
        eval_name: str,
        model_name: str,
        split: str,
        mode: str = "strict",
    ) -> BenchmarkMetrics:
        _ensure_nltk_data()

        input_data_path = prepared_data_path / "input_data.jsonl"
        input_response_path = prepared_data_path / "input_response_data.jsonl"
        output_dir = prepared_data_path / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        vendored_dir = Path(__file__).parent.parent / "vendored/instruction_following_eval"

        # Run the vendored evaluation script directly
        # Note: We run it as a script (not a module) because the vendored code
        # uses absolute imports like "from instruction_following_eval import ..."
        cmd = [
            sys.executable,
            str(vendored_dir / "evaluation_main.py"),
            f"--input_data={input_data_path}",
            f"--input_response_data={input_response_path}",
            f"--output_dir={output_dir}",
        ]

        # Add vendored directory to PYTHONPATH so imports work
        env = {**os.environ, "PYTHONPATH": str(vendored_dir.parent)}

        result = subprocess.run(
            cmd,
            check=False, env=env,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"IFEval evaluation failed:\n{result.stderr}")

        # Parse results from output files
        # The vendored code creates: eval_results_strict.jsonl and eval_results_loose.jsonl
        metrics = self._parse_output_files(output_dir, mode)

        return BenchmarkMetrics(
            eval_name=eval_name,
            model_name=model_name,
            split=split,
            overall_accuracy=metrics["prompt_level_accuracy"],
            total_examples=metrics["total_examples"],
            correct=metrics["correct"],
            incorrect=metrics["incorrect"],
            custom_metrics={
                "instruction_level_accuracy": metrics["instruction_level_accuracy"],
                "prompt_level_strict_accuracy": metrics.get("prompt_level_strict_accuracy"),
                "prompt_level_loose_accuracy": metrics.get("prompt_level_loose_accuracy"),
                "instruction_level_strict_accuracy": metrics.get("instruction_level_strict_accuracy"),
                "instruction_level_loose_accuracy": metrics.get("instruction_level_loose_accuracy"),
            },
        )

    def _parse_output_files(self, output_dir: Path, mode: str) -> dict:
        strict_path = output_dir / "eval_results_strict.jsonl"
        loose_path = output_dir / "eval_results_loose.jsonl"

        strict_results = []
        if strict_path.exists():
            with open(strict_path) as f:
                strict_results = [json.loads(line) for line in f]

        loose_results = []
        if loose_path.exists():
            with open(loose_path) as f:
                loose_results = [json.loads(line) for line in f]

        if mode == "strict":
            return self._calculate_metrics(strict_results, prefix="")
        if mode == "loose":
            return self._calculate_metrics(loose_results, prefix="")
        if mode == "both":
            strict_metrics = self._calculate_metrics(strict_results, prefix="strict_")
            loose_metrics = self._calculate_metrics(loose_results, prefix="loose_")
            return {
                "prompt_level_accuracy": strict_metrics["prompt_level_accuracy"],
                "instruction_level_accuracy": strict_metrics["instruction_level_accuracy"],
                "total_examples": strict_metrics["total_examples"],
                "correct": strict_metrics["correct"],
                "incorrect": strict_metrics["incorrect"],
                "prompt_level_strict_accuracy": strict_metrics["prompt_level_accuracy"],
                "instruction_level_strict_accuracy": strict_metrics["instruction_level_accuracy"],
                "prompt_level_loose_accuracy": loose_metrics["prompt_level_accuracy"],
                "instruction_level_loose_accuracy": loose_metrics["instruction_level_accuracy"],
            }
        raise ValueError(f"Invalid mode: {mode}. Must be 'strict', 'loose', or 'both'")

    def _calculate_metrics(self, results: list[dict], prefix: str = "") -> dict:
        if not results:
            return {
                f"{prefix}prompt_level_accuracy": 0.0,
                f"{prefix}instruction_level_accuracy": 0.0,
                f"{prefix}total_examples": 0,
                f"{prefix}correct": 0,
                f"{prefix}incorrect": 0,
            }

        # Prompt-level accuracy: all instructions followed
        prompt_correct = sum(1 for r in results if r.get("follow_all_instructions", False))
        total_prompts = len(results)
        prompt_level_accuracy = prompt_correct / total_prompts if total_prompts > 0 else 0.0

        # Instruction-level accuracy: individual instruction compliance
        total_instructions = 0
        correct_instructions = 0
        for result in results:
            follow_list = result.get("follow_instruction_list", [])
            total_instructions += len(follow_list)
            correct_instructions += sum(1 for followed in follow_list if followed)

        instruction_level_accuracy = (
            correct_instructions / total_instructions if total_instructions > 0 else 0.0
        )

        return {
            f"{prefix}prompt_level_accuracy": prompt_level_accuracy,
            f"{prefix}instruction_level_accuracy": instruction_level_accuracy,
            f"{prefix}total_examples": total_prompts,
            f"{prefix}correct": prompt_correct,
            f"{prefix}incorrect": total_prompts - prompt_correct,
        }

