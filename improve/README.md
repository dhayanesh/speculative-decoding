# Improve: Acceptance and Resource Check

## Goal
Identify when speculative acceptance is high (prompt type/length), and a simple resource check for:
- Base (`Qwen/Qwen3-8B`) for latency baseline
- Draft speculative (`Qwen/Qwen3-0.6B`)
- EAGLE3 speculative (`Tengyunw/qwen3_8b_eagle3`)

## Setup
- Shared settings:
  - `temperature=0.0`, `top_p=1.0`
  - `num_speculative_tokens=4` (speculative runs)
  - `max_model_len=4096`
- Prompt suite: 8 cases with varied task type and input length

## Per-Case Acceptance and Time

| Case | Category | Prompt tokens | Base time (s) | Draft time (s) | Draft acc (%) | Eagle3 time (s) | Eagle3 acc (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| short_definition | factual_short | 11 | 2.801 | 2.207 | 39.86 | 1.835 | 22.50 |
| structured_json | structured_output | 41 | 2.805 | 1.696 | 64.81 | 1.594 | 31.98 |
| code_generation | code | 23 | 5.590 | 3.216 | 51.59 | 3.433 | 25.26 |
| creative_writing | creative_openended | 18 | 7.446 | 3.601 | 65.14 | 4.193 | 29.49 |
| reasoning_math | reasoning | 48 | 6.516 | 3.624 | 60.98 | 4.180 | 23.06 |
| translation | translation | 23 | 2.803 | 1.585 | 51.61 | 1.884 | 20.67 |
| summary_medium | long_context_summary | 622 | 7.552 | 3.643 | 66.20 | 3.264 | 48.28 |
| summary_long | long_context_summary | 2350 | 7.845 | 4.177 | 60.00 | 4.289 | 35.71 |


## Simple Resource Check (Draft vs Eagle3)
Method:
- Sampled `nvidia-smi` once per second during each full script run.

| Metric | Draft | Eagle3 |
|---|---:|---:|
| GPU util avg (%) | 43.01 | 50.79 |
| GPU util peak (%) | 100.00 | 100.00 |
| GPU mem avg (MiB) | 26092.69 | 26820.46 |
| GPU mem peak (MiB) | 41649.00 | 41855.00 |
