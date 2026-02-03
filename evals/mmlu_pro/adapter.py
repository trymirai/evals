import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from ..protocols import EvalAdapter
from ..types import (
    BenchmarkMetrics,
    EvalPrompt,
    InternalEvalRecord,
    PredictionRecord,
    PromptMessage,
)
from ..vendored.mmlu_pro import extract_answer


@dataclass(frozen=True)
class MMLUProAdapter(EvalAdapter):
    def convert_record(self, record: dict) -> InternalEvalRecord:
        return InternalEvalRecord(
            id=str(record["question_id"]),
            question=record["question"],
            answer=record["answer"],
            options=record["options"],
            answer_index=record["answer_index"],
            reasoning=record.get("cot_content"),
            category=record.get("category"),
            metadata={
                "src": record.get("src", ""),
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
        records: list[InternalEvalRecord],
        few_shot_source: list[InternalEvalRecord] | None = None,
        num_few_shot: int = 5,
    ) -> list[EvalPrompt]:
        """Generate MMLU-Pro prompts using vendored formatting logic.

        Matches reference implementation: creates single prompt string with
        initial_prompt + few-shot examples + test question.
        """
        from ..vendored.mmlu_pro.prompts import format_cot_example

        prompts = []
        for record in records:
            # Build full prompt as single string (matches reference implementation)
            full_prompt = ""

            # 1. Initial prompt with category
            full_prompt += self._load_system_prompt(record.category or "general")
            full_prompt += "\n"

            # 2. Few-shot examples with answers
            few_shot = self._select_few_shot(record, few_shot_source, num_few_shot)
            for example in few_shot:
                example_dict = self._to_vendored_format(example)
                full_prompt += format_cot_example(example_dict, including_answer=True)

            # 3. Test question without answer
            test_dict = self._to_vendored_format(record)
            full_prompt += format_cot_example(test_dict, including_answer=False)

            prompts.append(
                EvalPrompt(
                    id=record.id,
                    messages=[
                        PromptMessage(role="user", content=full_prompt),
                    ],
                    category=record.category,
                )
            )

        return prompts

    def _load_system_prompt(self, category: str) -> str:
        prompt_path = Path(__file__).parent.parent / "vendored/mmlu_pro/initial_prompt.txt"
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

        category = test_record.category
        same_category = [r for r in few_shot_source if r.category == category]

        return same_category[:num_few_shot]

    def _to_vendored_format(self, record: InternalEvalRecord) -> dict:
        return {
            "question": record.question,
            "options": record.options,
            "answer": record.answer,
            "cot_content": record.reasoning or "",
            "category": record.category,
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
                    "Predictions and ground truth must be in the same order with matching IDs."
                )

            category = gt.category or "other"
            if category not in by_category:
                by_category[category] = []

            by_category[category].append(
                {
                    "model_outputs": pred.model_output,
                    "answer": gt.answer,
                }
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
        import random

        random.seed(12345)
        results = {}

        for file_path in prepared_data_path.glob("*.json"):
            category = file_path.stem
            succ, fail = 0, 0

            with open(file_path) as f:
                entries = json.load(f)
                for e in entries:
                    pred = extract_answer(e["model_outputs"], level="l2")
                    if pred is None:
                        pred = random.choice(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])

                    if pred == e["answer"]:
                        succ += 1
                    else:
                        fail += 1

            total = succ + fail
            accuracy = succ / total if total > 0 else 0.0

            results[category] = {
                "accuracy": accuracy,
                "correct": succ,
                "total": total,
            }

        total_correct = sum(r["correct"] for r in results.values())
        total_examples = sum(r["total"] for r in results.values())
        overall_accuracy = total_correct / total_examples if total_examples > 0 else 0.0

        category_metrics = {category: r["accuracy"] for category, r in results.items()}

        return BenchmarkMetrics(
            eval_name=eval_name,
            model_name=model_name,
            split=split,
            overall_accuracy=overall_accuracy,
            total_examples=total_examples,
            correct=total_correct,
            incorrect=total_examples - total_correct,
            category_metrics=category_metrics,
        )
