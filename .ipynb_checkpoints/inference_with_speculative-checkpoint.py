import time

from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-14B"
DRAFT_MODEL = "Qwen/Qwen3-0.6B"
TENSOR_PARALLEL_SIZE = 1
DRAFT_TENSOR_PARALLEL_SIZE = 1
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

PROMPT = (
    "You are an expert technical editor. Summarize the following report for senior "
    "engineering leadership. Return these sections exactly: Executive Summary, Key "
    "Findings, Risks, Tradeoffs, and Recommended Plan.\n\n"
    "REPORT:\n"
    + "\n".join([ARTICLE_PARAGRAPH for _ in range(32)])
    + "\n\nSUMMARY:\n"
)


def run_inference_with_speculative() -> dict:
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        speculative_config={
            "model": DRAFT_MODEL,
            "method": "draft_model",
            "num_speculative_tokens": 4,
            "draft_tensor_parallel_size": DRAFT_TENSOR_PARALLEL_SIZE,
        },
    )
    input_tokens = len(llm.get_tokenizer().encode(PROMPT))
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    llm.generate(PROMPT, params, use_tqdm=False)  # warmup
    start = time.perf_counter()
    result = llm.generate(PROMPT, params, use_tqdm=False)[0].outputs[0]
    latency = time.perf_counter() - start

    tokens = len(result.token_ids)
    return {
        "mode": "with_speculative_decoding_draft_model_0.6b",
        "input_tokens": input_tokens,
        "latency_s": latency,
        "output_tokens": tokens,
        "tokens_per_second": tokens / latency,
        "text_head": result.text.strip()[:200],
    }


if __name__ == "__main__":
    metrics = run_inference_with_speculative()
    print(f"mode: {metrics['mode']}")
    print(f"input_tokens: {metrics['input_tokens']}")
    print(f"latency_s: {metrics['latency_s']:.3f}")
    print(f"output_tokens: {metrics['output_tokens']}")
    print(f"tokens_per_second: {metrics['tokens_per_second']:.2f}")
    print(f"text_head: {metrics['text_head']}")
