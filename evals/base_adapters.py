from pathlib import Path

import huggingface_hub
from datasets import load_dataset

from evals.protocols import EvalAdapter
from evals.types import InternalEvalRecord


def _list_parquet_files_for_split(repo_id: str, split: str) -> list[str]:
    all_files = huggingface_hub.list_repo_files(repo_id, repo_type="dataset")
    parquet_files = [
        filename
        for filename in all_files
        if filename.endswith(".parquet")
        and (filename.startswith((f"{split}/", f"{split}-")) or f"/{split}-" in filename)
    ]
    return parquet_files


class ParquetBasedAdapter(EvalAdapter):
    @classmethod
    def download_split(
        cls,
        repo_id: str,
        split: str,
        temp_dir: Path,
    ) -> list[InternalEvalRecord]:
        parquet_files = _list_parquet_files_for_split(repo_id, split)
        all_records = []

        for filename in parquet_files:
            downloaded_path = huggingface_hub.hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                local_dir=str(temp_dir),
            )
            records = cls().convert_split(Path(downloaded_path))
            all_records.extend(records)

        return all_records


class HFDatasetsAdapter(EvalAdapter):
    @classmethod
    def download_split(
        cls,
        repo_id: str,
        split: str,
        temp_dir: Path,  # noqa: ARG003
    ) -> list[InternalEvalRecord]:
        dataset = load_dataset(repo_id, split=split)

        all_records = []
        for record in dataset:
            converted = cls().convert_record(record)
            all_records.append(converted)

        return all_records
