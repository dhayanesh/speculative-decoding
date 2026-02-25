from time import perf_counter

from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter as MetricCounter

TENSOR_PARALLEL_SIZE = 1
DRAFT_TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.88
MAX_MODEL_LEN = 4096
MAX_TOKENS = 512
NUM_SPECULATIVE_TOKENS = 4

ARTICLE_PARAGRAPH = (
    "Speculative decoding is a generation optimization where a smaller draft model "
    "proposes candidate tokens, and a larger target model verifies them in batches. "
    "This can reduce expensive target-model forward passes when proposals are accepted. "
    "Real performance depends on acceptance rate, scheduler behavior, KV-cache pressure, "
    "and kernel efficiency. In production, observability, failure handling, and rollout "
    "strategy matter as much as raw speed."
)

SUMMARIZATION_PROMPT = (
    "You are an expert technical editor. Summarize the following report for senior "
    "engineering leadership. Return these sections exactly: Executive Summary, Key "
    "Findings, Risks, Tradeoffs, and Recommended Plan.\n\n"
    "REPORT:\n"
    + "\n".join([ARTICLE_PARAGRAPH for _ in range(32)])
    + "\n\nSUMMARY:\n"
)

llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    max_model_len=MAX_MODEL_LEN,
    disable_log_stats=False,
    speculative_config={
        "model": "Qwen/Qwen3-0.6B",
        "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
        "draft_tensor_parallel_size": DRAFT_TENSOR_PARALLEL_SIZE,
        "max_model_len": MAX_MODEL_LEN,
        "method": "draft_model",
    },
)

def _read_spec_decode_counters(llm: LLM) -> dict[str, int]:
    counters = {
        "num_drafts": 0,
        "num_draft_tokens": 0,
        "num_accepted_tokens": 0,
    }
    for metric in llm.llm_engine.get_metrics():
        if not isinstance(metric, MetricCounter):
            continue
        if metric.name == "vllm:spec_decode_num_drafts":
            counters["num_drafts"] += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            counters["num_draft_tokens"] += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            counters["num_accepted_tokens"] += metric.value
    return counters


def _print_acceptance(task_name: str, before: dict[str, int], after: dict[str, int]) -> None:
    num_drafts = after["num_drafts"] - before["num_drafts"]
    num_draft_tokens = after["num_draft_tokens"] - before["num_draft_tokens"]
    num_accepted_tokens = after["num_accepted_tokens"] - before["num_accepted_tokens"]
    if num_draft_tokens <= 0:
        print(f"{task_name} acceptance: n/a (no draft tokens recorded)")
        return
    acceptance_pct = (num_accepted_tokens / num_draft_tokens) * 100
    print(
        f"{task_name} acceptance: {acceptance_pct:.2f}% "
        f"({num_accepted_tokens}/{num_draft_tokens} accepted draft tokens)"
    )
    if num_drafts > 0:
        mean_acceptance_len = 1 + (num_accepted_tokens / num_drafts)
        print(f"{task_name} mean acceptance length: {mean_acceptance_len:.2f}")


warmup_prompts = ["Warmup run"]
warmup_sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=16)
llm.generate(warmup_prompts, warmup_sampling_params)

# Normal inference request
prompts = ["Explain speculative decoding in 3 concise bullet points."]
sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS)
spec_metrics_before = _read_spec_decode_counters(llm)
start = perf_counter()
outputs = llm.generate(prompts, sampling_params)
end = perf_counter()
print(f"Normal inference time: {end - start:.3f} sec")
_print_acceptance("Normal inference", spec_metrics_before, _read_spec_decode_counters(llm))
for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")

# Summarization request (~2500 input tokens)
prompts = [SUMMARIZATION_PROMPT]
sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS)
spec_metrics_before = _read_spec_decode_counters(llm)
start = perf_counter()
outputs = llm.generate(prompts, sampling_params)
end = perf_counter()
print(f"Summarization time: {end - start:.3f} sec")
_print_acceptance("Summarization", spec_metrics_before, _read_spec_decode_counters(llm))
for output in outputs:
    print(f"Prompt: {output.prompt[:120]!r}..., Generated text: {output.outputs[0].text!r}")
