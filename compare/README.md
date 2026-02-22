# Base vs Draft Model vs EAGLE3 (Inference Time + Acceptance)

- Target model: `Qwen/Qwen3-8B`
- Draft model (`draft_model` method): `Qwen/Qwen3-0.6B`
- EAGLE model (`eagle3` method): `Tengyunw/qwen3_8b_eagle3`

## Config used

- `TENSOR_PARALLEL_SIZE = 1`
- `DRAFT_TENSOR_PARALLEL_SIZE = 1`
- `GPU_MEMORY_UTILIZATION = 0.88`
- `MAX_MODEL_LEN = 4096`
- `MAX_TOKENS = 512`
- `num_speculative_tokens = 4`
- Sampling params: `temperature=0.0`, `top_p=1.0`

## Normal inference

| Technique | Inference time (s) | Acceptance | Accepted / Draft tokens | Mean acceptance length |
|---|---:|---:|---:|---:|
| base | 14.868 | n/a | n/a | n/a |
| draft_model | 9.663 | 43.82% | 326 / 744 | 2.75 |
| eagle3 | 8.716 | 27.57% | 268 / 972 | 2.10 |

## Summarization inference

| Technique | Inference time (s) | Acceptance | Accepted / Draft tokens | Mean acceptance length |
|---|---:|---:|---:|---:|
| base | 15.610 | n/a | n/a | n/a |
| draft_model | 9.245 | 50.59% | 342 / 676 | 3.02 |
| eagle3 | 8.886 | 32.92% | 291 / 884 | 2.32 |