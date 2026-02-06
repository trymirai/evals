import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from evals.base_adapters import ParquetBasedAdapter
from evals.types import (
    BenchmarkMetrics,
    DatasetLoadConfig,
    EvalPrompt,
    InternalEvalRecord,
    PredictionRecord,
    PromptMessage,
)
from evals.vendored.mmlu_pro.prompts import format_cot_example


@dataclass(frozen=True)
class MMLUProAdapter(ParquetBasedAdapter):
    num_few_shot: int = 5

    def get_benchmark_split(self) -> str:
        return "test"

    def get_loading_config(self, limit: int | None) -> list[DatasetLoadConfig]:
        return [
            DatasetLoadConfig(split="test", limit=limit),
            DatasetLoadConfig(split="validation", limit=None),
        ]

    def convert_record(self, record: dict) -> InternalEvalRecord:
        return InternalEvalRecord(
            id=str(record["question_id"]),
            question=record["question"],
            answer=record["answer"],
            metadata={
                "src": record.get("src", ""),
                "options": record["options"],
                "answer_index": record["answer_index"],
                "reasoning": record.get("cot_content"),
                "category": record.get("category"),
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
        records = datasets["test"]
        few_shot_source = datasets.get("validation")

        prompts = []
        for record in records:
            full_prompt = ""

            category = record.metadata.get("category") if record.metadata else None
            full_prompt += self._load_system_prompt(category or "general")
            full_prompt += "\n"

            few_shot = self._select_few_shot(record, few_shot_source, self.num_few_shot)
            for example in few_shot:
                example_dict = self._to_vendored_format(example)
                full_prompt += format_cot_example(example_dict, including_answer=True)

            test_dict = self._to_vendored_format(record)
            full_prompt += format_cot_example(test_dict, including_answer=False)

            prompts.append(
                EvalPrompt(
                    id=record.id,
                    messages=[
                        PromptMessage(role="user", content=full_prompt),
                    ],
                    category=category,
                ),
            )

        return prompts

    def _load_system_prompt(self, category: str) -> str:
        prompt_path = Path(__file__).parent.parent / "vendored/mmlu_pro/cot_prompt_lib/initial_prompt.txt"
        with open(prompt_path) as f:
            prompt = f.read()
        return prompt.replace("{$}", category)

    def _select_few_shot(
        self,
        test_record: InternalEvalRecord,
        few_shot_source: list[InternalEvalRecord] | None,
        num_few_shot: int,
    ) -> list[InternalEvalRecord]:
        if not few_shot_source or num_few_shot == 0:
            return []

        category = test_record.metadata.get("category") if test_record.metadata else None
        same_category = [
            r for r in few_shot_source
            if (r.metadata and r.metadata.get("category") == category)
        ]

        return same_category[:num_few_shot]

    def _to_vendored_format(self, record: InternalEvalRecord) -> dict:
        metadata = record.metadata or {}
        return {
            "question": record.question,
            "options": metadata.get("options"),
            "answer": record.answer,
            "cot_content": metadata.get("reasoning", ""),
            "category": metadata.get("category"),
        }

    def prepare_for_benchmark(
        self,
        predictions: list[PredictionRecord],
        ground_truth: list[InternalEvalRecord],
        output_dir: Path,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        by_category: dict[str, list[dict]] = {}
        for pred, gt in zip(predictions, ground_truth, strict=True):
            if pred.id != gt.id:
                raise ValueError(
                    f"ID mismatch at position: prediction.id={pred.id!r} != ground_truth.id={gt.id!r}. "
                    "Predictions and ground truth must be in the same order with matching IDs.",
                )

            category = (gt.metadata.get("category") if gt.metadata else None) or "other"
            if category not in by_category:
                by_category[category] = []

            by_category[category].append(
                {
                    "model_outputs": pred.model_output,
                    "answer": gt.answer,
                },
            )

        for category, entries in by_category.items():
            category_file = output_dir / f"{category}.json"
            with open(category_file, "w") as f:
                json.dump(entries, f, indent=2)

        return output_dir

    def run_benchmark(
        self,
        prepared_data_path: Path,
        eval_name: str,
        model_name: str,
        split: str,
    ) -> BenchmarkMetrics:
        random.seed(12345)

        vendored_dir = Path(__file__).parent.parent / "vendored/mmlu_pro"

        cmd = [
            sys.executable,
            str(vendored_dir / "compute_accuracy.py"),
            str(prepared_data_path),
        ]

        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"MMLU-Pro evaluation failed:\n{result.stderr}")

        metrics = self._parse_stdout(result.stdout, prepared_data_path)

        return BenchmarkMetrics(
            eval_name=eval_name,
            model_name=model_name,
            split=split,
            overall_accuracy=metrics["overall_accuracy"],
            total_examples=metrics["total_examples"],
            correct=metrics["correct"],
            incorrect=metrics["incorrect"],
            category_metrics=metrics["category_metrics"],
        )

    def _parse_stdout(self, stdout: str, prepared_data_path: Path) -> dict:
        lines = stdout.strip().split("\n")

        # Find Level 2 section
        level2_start = None
        for i, line in enumerate(lines):
            if "Level 2 regex" in line:
                level2_start = i + 1
                break

        if level2_start is None:
            raise RuntimeError("Could not find Level 2 results in output")

        # Parse: "path/to/category.json accuracy"
        category_accuracies = {}
        for line in lines[level2_start:]:
            if not line.strip() or "Level" in line:
                break
            parts = line.strip().split()
            if len(parts) >= 2:
                filepath = parts[0]
                accuracy = float(parts[1])
                category = Path(filepath).stem
                category_accuracies[category] = accuracy

        total_correct = 0
        total_examples = 0
        category_metrics = {}

        for category, accuracy in category_accuracies.items():
            json_path = prepared_data_path / f"{category}.json"
            with open(json_path) as f:
                entries = json.load(f)
                count = len(entries)
                correct = int(round(accuracy * count))

                category_metrics[category] = accuracy
                total_correct += correct
                total_examples += count

        overall_accuracy = total_correct / total_examples if total_examples > 0 else 0.0

        return {
            "overall_accuracy": overall_accuracy,
            "total_examples": total_examples,
            "correct": total_correct,
            "incorrect": total_examples - total_correct,
            "category_metrics": category_metrics,
        }
