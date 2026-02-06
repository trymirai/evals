<a href="https://artifacts.trymirai.com/social/about_us.mp3"><img src="https://img.shields.io/badge/Listen-Podcast-red" alt="Listen to our podcast"></a>
<a href="https://docsend.com/v/76bpr/mirai2025"><img src="https://img.shields.io/badge/View-Deck-red" alt="View our deck"></a>
<a href="https://discord.com/invite/trymirai"><img src="https://img.shields.io/discord/1377764166764462120?label=Discord" alt="Discord"></a>
<a href="mailto:contact@getmirai.co?subject=Interested%20in%20Mirai"><img src="https://img.shields.io/badge/Send-Email-green" alt="Contact us"></a>
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

# Evals

Evaluation adapters for [lalamo](https://github.com/trymirai/lalamo) with vendored reference implementations. Adapters normalize different benchmark formats into a common pipeline.

## Adding a new adapter

1. Create `evals/<benchmark_name>/adapter.py`
2. Implement `EvalAdapter` protocol (see `evals/protocols.py`)
3. Vendor the official evaluation code in `evals/vendored/<benchmark_name>/`

See `evals/ifeval/adapter.py` or `evals/mmlu_pro/adapter.py` for examples.
