from time import perf_counter

from vllm import LLM, SamplingParams

TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.88
MAX_MODEL_LEN = 4096
MAX_TOKENS = 256

ARTICLE_PARAGRAPH = (
    "An internal platform team rolled out speculative decoding to cut serving costs. "
    "Initial canary runs showed lower latency but unstable tail performance during peak "
    "traffic. Investigation pointed to low draft-token acceptance for complex prompts, "
    "cache pressure from long contexts, and uneven scheduler behavior across replicas. "
    "The team introduced stricter prompt templates, better fallback handling, and "
    "per-route observability before expanding rollout."
)

def _build_summary_prompt(num_paragraphs: int) -> str:
    return (
        "You are an incident review editor. Summarize the report for engineering "
        "leadership. Return these sections exactly: Executive Summary, Root Causes, "
        "Mitigations, Remaining Risks, Next 30 Days.\n\n"
        "REPORT:\n"
        + "\n".join([ARTICLE_PARAGRAPH for _ in range(num_paragraphs)])
        + "\n\nSUMMARY:\n"
    )


TEST_CASES = [
    {
        "id": "short_definition",
        "category": "factual_short",
        "prompt": "What is speculative decoding? Answer in exactly one sentence.",
        "max_tokens": 96,
    },
    {
        "id": "structured_json",
        "category": "structured_output",
        "prompt": (
            "Return valid minified JSON only with keys issue,severity,action.\n"
            "ISSUE: Tail latency spikes during peak traffic.\n"
            "SEVERITY OPTIONS: low,medium,high.\n"
            "ACTION: one concrete mitigation."
        ),
        "max_tokens": 96,
    },
    {
        "id": "code_generation",
        "category": "code",
        "prompt": (
            "Write a Python function clamp(x, lo, hi) with type hints and a docstring. "
            "Return code only."
        ),
        "max_tokens": 192,
    },
    {
        "id": "creative_writing",
        "category": "creative_openended",
        "prompt": (
            "Write a vivid 180-word micro-story about a failing datacenter at midnight."
        ),
        "max_tokens": 256,
    },
    {
        "id": "reasoning_math",
        "category": "reasoning",
        "prompt": (
            "A service processes 240 requests/min. Latency rises 25% and throughput drops "
            "10%. What is the new throughput and what percentage increase in total processing "
            "time per 1000 requests? Show steps."
        ),
        "max_tokens": 224,
    },
    {
        "id": "translation",
        "category": "translation",
        "prompt": (
            "Translate to Spanish, keeping technical terms in English: "
            "'Speculative decoding improves latency when draft token acceptance is high.'"
        ),
        "max_tokens": 96,
    },
    {
        "id": "summary_medium",
        "category": "long_context_summary",
        "prompt": _build_summary_prompt(8),
        "max_tokens": MAX_TOKENS,
    },
    {
        "id": "summary_long",
        "category": "long_context_summary",
        "prompt": _build_summary_prompt(32),
        "max_tokens": MAX_TOKENS,
    },
]

llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    max_model_len=MAX_MODEL_LEN,
)


warmup_prompts = ["Warmup run"]
warmup_sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=16)
llm.generate(warmup_prompts, warmup_sampling_params)

tokenizer = llm.get_tokenizer()
print(
    "RESULT_HEADER|model=base|"
    "case_id|category|prompt_chars|prompt_tokens|output_tokens|time_sec"
)
for case in TEST_CASES:
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=case["max_tokens"]
    )
    start = perf_counter()
    outputs = llm.generate([case["prompt"]], sampling_params)
    end = perf_counter()
    output_tokens = len(outputs[0].outputs[0].token_ids)
    prompt_tokens = len(tokenizer.encode(case["prompt"]))
    print(
        "RESULT|model=base|"
        f"case_id={case['id']}|category={case['category']}|"
        f"prompt_chars={len(case['prompt'])}|prompt_tokens={prompt_tokens}|"
        f"output_tokens={output_tokens}|time_sec={end - start:.3f}"
    )
