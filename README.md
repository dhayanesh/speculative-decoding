# Speculative Decoding Runtime Comparison

## Overview

This workspace contains two separate vLLM inference scripts:

- `inference_without_speculative.py`
- `inference_with_speculative.py`

Both scripts include the same built-in summarization prompt and only differ by
speculative decoding configuration.

## Models

- Main model: `Qwen/Qwen3-14B`
- Draft model (speculative script only): `Qwen/Qwen3-0.6B`

## Prompt Task

- Task type: long-form summarization
- Prompt is hardcoded in both scripts
- Input length observed at runtime: `2477` tokens (approximately a 2500-token input task)
- Output tokens requested: `512`

## Install

```bash
python -m pip install vllm torch
```

## Run

```bash
CUDA_VISIBLE_DEVICES=0 python inference_without_speculative.py
CUDA_VISIBLE_DEVICES=0 python inference_with_speculative.py
```

## Current Script Settings

Defined independently in each script:

- `tensor_parallel_size=1`
- `gpu_memory_utilization=0.88`
- `max_model_len=4096`
- `max_tokens=512`

## Runtime Comparison

Hardware used: `NVIDIA A40` (single GPU)

Single-run snapshot:

- without spec latency: `27.777s`
- with spec latency: `14.141s`
- speedup: `1.964x`

5-trial:

- without spec:
  - input tokens: `2477`
  - output tokens: `512`
  - latency mean/std: `27.783s ± 0.006`
  - throughput mean/std: `18.43 ± 0.00 tok/s`

- with spec:
  - input tokens: `2477`
  - output tokens: `512`
  - latency mean/std: `14.131s ± 0.003`
  - throughput mean/std: `36.23 ± 0.01 tok/s`

Comparison:

- latency speedup (mean): `1.966x`
- throughput ratio (mean): `1.966x`
