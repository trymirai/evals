# Vendored MMLU-Pro Code

This directory contains vendored code from the MMLU-Pro repository.

## Source

- **Repository:** https://github.com/TIGER-AI-Lab/MMLU-Pro
- **Commit:** `7eca4cee303b85033bf371bb9f00fe9278d0e30d`
- **Date:** 2026-01-30
- **License:** Apache License 2.0

## Why Vendored?

The MMLU-Pro code is written as scripts (not importable modules). To use the exact reference implementation in our evaluation framework, we vendor the code with minimal modifications to make it importable.

## Changes from Original

**Minimal edits for importability:**

### `compute_accuracy.py`
- Added module docstring with source attribution
- Wrapped script execution code in `if __name__ == "__main__"` block
- Functions `extract_answer()`, `extract_again()`, `extract_final()` remain **unchanged**

### `evaluate_from_local.py`
- Added `ntrain=5` parameter to `eval_cot()` function (was reading from global `args.ntrain`)
- Updated `eval_cot()` call in `main()` to pass `args.ntrain`
- All other functions remain unchanged

### `LICENSE`
- Copied verbatim from original repository

## Usage

```python
from evals.vendored.mmlu_pro.compute_accuracy import extract_answer, extract_again, extract_final

pred = extract_answer(model_output, level="l2")
```
