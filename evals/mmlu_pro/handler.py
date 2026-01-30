import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from ..common import EvalHandler
from ..formats import BenchmarkMetrics, InternalEvalRecord, PredictionRecord
from ..vendored.mmlu_pro import extract_answer


@dataclass(frozen=True)
class MMLUProHandler(EvalHandler):
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

            by_category[category].append({
                "model_outputs": pred.model_output,
                "answer": gt.answer,
            })

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

        # Use extraction logic (level 2 regex with fallbacks)
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

    def evaluate_with_vllm(
        self,
        model_path: str,
        ground_truth: list[InternalEvalRecord],
        output_dir: Path,
        ntrain: int = 5,
        gpu_util: float = 0.8,
        lora_path: str | None = None,
    ) -> dict[str, dict]:
        from ..vendored.mmlu_pro.evaluate_from_local import (
            eval_cot,
            load_mmlu_pro,
            select_by_category,
        )
        import torch
        import transformers
        from vllm import LLM, SamplingParams  # type: ignore
        from vllm.lora.request import LoRARequest  # type: ignore

        output_dir.mkdir(parents=True, exist_ok=True)

        llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_util,
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=4096,
            trust_remote_code=True,
            enable_lora=True if lora_path else False,
        )
        sampling_params = SamplingParams(temperature=0, max_tokens=2048, stop=["Question:"])
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        lora_request = None
        if lora_path:
            lora_request = LoRARequest(lora_name="lora", lora_path=lora_path, lora_int_id=1)

        model = (llm, sampling_params, lora_request)

        # Load validation data for few-shot examples
        _, full_val_df = load_mmlu_pro()

        # Convert InternalEvalRecord to format expected by vendored code
        test_data_by_category = {}
        for record in ground_truth:
            category = record.category or "other"
            if category not in test_data_by_category:
                test_data_by_category[category] = []

            test_data_by_category[category].append(
                {
                    "question_id": record.id,
                    "question": record.question,
                    "options": record.options,
                    "answer": record.answer,
                    "answer_index": record.answer_index,
                    "category": category,
                    "cot_content": record.reasoning or "",
                }
            )

        # Run evaluation for each category
        results = {}
        for category, test_df in sorted(test_data_by_category.items()):
            val_df = select_by_category(full_val_df, category)
            output_path = output_dir / f"{category}.json"

            accu, corr, wrong = eval_cot(
                subject=category,
                model=model,
                tokenizer=tokenizer,
                val_df=val_df,
                test_df=test_df,
                output_path=str(output_path),
                ntrain=ntrain,
            )

            results[category] = {"accu": accu, "corr": corr, "wrong": wrong}

        # Compute overall statistics
        total_corr = sum(r["corr"] for r in results.values())
        total_wrong = sum(r["wrong"] for r in results.values())
        total_accu = (
            total_corr / (total_corr + total_wrong) if (total_corr + total_wrong) > 0 else 0.0
        )

        results["total"] = {"accu": total_accu, "corr": total_corr, "wrong": total_wrong}

        return results
