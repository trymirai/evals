"""
Vendored MMLU-Pro evaluation code.

Source: https://github.com/TIGER-AI-Lab/MMLU-Pro
Commit: 7eca4cee303b85033bf371bb9f00fe9278d0e30d
License: Apache License 2.0
"""

from evals.vendored.mmlu_pro.compute_accuracy import extract_answer, extract_again, extract_final

__all__ = [
    "extract_answer",
    "extract_again",
    "extract_final",
]
