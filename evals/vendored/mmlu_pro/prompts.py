"t"Prompt formatting functions extracted from evaluate_from_local.py.

This module contains only the prompt formatting logic without heavy dependencies
like torch, vllm, etc. Use this for lightweight prompt generation.

For full evaluation with vLLM, use evaluate_from_local.py directly.
"""

# Choice letters for multiple choice options
choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]


def format_cot_example(example, including_answer=True):
    """Format a single example in chain-of-thought format.

    Args:
        example: Dict with keys: question, options, cot_content (if including_answer)
        including_answer: Whether to include the answer and reasoning

    Returns:
        Formatted prompt string

    Source: https://github.com/TIGER-AI-Lab/MMLU-Pro (evaluate_from_local.py)
    """
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        cot_content = example["cot_content"].replace(
            "A: Let's think step by step.",
            "Answer: Let's think step by step."
        )
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def select_by_category(records, category):
    """Select records matching a specific category.

    Args:
        records: List of dicts with 'category' key
        category: Category to filter by

    Returns:
        List of records matching the category

    Source: https://github.com/TIGER-AI-Lab/MMLU-Pro (evaluate_from_local.py)
    """
    res = []
    for each in records:
        if each["category"] == category:
            res.append(each)
    return res
