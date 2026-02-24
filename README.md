# Speculative Decoding Experiments

This repo contains two experiment tracks:

- `compare/`: direct comparison of `base` vs `draft_model` vs `eagle3`
- `improve/`: prompt-type/length study and simple GPU resource checks

## Models Used

- Target model: `Qwen/Qwen3-8B`
- Draft model (`draft_model`): `Qwen/Qwen3-0.6B`
- EAGLE3 model (`eagle3`): `Tengyunw/qwen3_8b_eagle3`

## Shared Core Config

- `TENSOR_PARALLEL_SIZE=1`
- `DRAFT_TENSOR_PARALLEL_SIZE=1`
- `GPU_MEMORY_UTILIZATION=0.88`
- `MAX_MODEL_LEN=4096`
- `num_speculative_tokens=4` (speculative runs)
- Sampling: `temperature=0.0`, `top_p=1.0`

## draft_model vs eagle3 - Comparison Results (`compare/README.md`)

### Normal inference
| Technique | Inference time (s) | Acceptance | Accepted / Draft tokens | Mean acceptance length |
|---|---:|---:|---:|---:|
| base | 14.868 | n/a | n/a | n/a |
| draft_model | 9.663 | 43.82% | 326 / 744 | 2.75 |
| eagle3 | 8.716 | 27.57% | 268 / 972 | 2.10 |

### Summarization inference
| Technique | Inference time (s) | Acceptance | Accepted / Draft tokens | Mean acceptance length |
|---|---:|---:|---:|---:|
| base | 15.610 | n/a | n/a | n/a |
| draft_model | 9.245 | 50.59% | 342 / 676 | 3.02 |
| eagle3 | 8.886 | 32.92% | 291 / 884 | 2.32 |

## Acceptance Results (`improve/README.md`)

### Prompt-type/length study (8 cases)
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