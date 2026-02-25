from time import perf_counter

from vllm import LLM, SamplingParams

TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.88
MAX_MODEL_LEN = 4096
MAX_TOKENS = 512

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
)


warmup_prompts = ["Warmup run"]
warmup_sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=16)
llm.generate(warmup_prompts, warmup_sampling_params)

# Normal inference request
prompts = ["Explain speculative decoding in 3 concise bullet points."]
sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS)
start = perf_counter()
outputs = llm.generate(prompts, sampling_params)
end = perf_counter()
print(f"Normal inference time: {end - start:.3f} sec")
for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")

# Summarization request (~2500 input tokens)
prompts = [SUMMARIZATION_PROMPT]
sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS)
start = perf_counter()
outputs = llm.generate(prompts, sampling_params)
end = perf_counter()
print(f"Summarization time: {end - start:.3f} sec")
for output in outputs:
    print(f"Prompt: {output.prompt[:120]!r}..., Generated text: {output.outputs[0].text!r}")
