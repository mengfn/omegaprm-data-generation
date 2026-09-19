"""Model adapters. Importing the project never imports vLLM or changes credentials."""
import re
from structure import Completion


class VLLMCompleter:
    def __init__(self, model: str, max_tokens: int = 2048, temperature: float = 0.8,
                 top_p: float = 0.95, max_model_len: int = 4096,
                 tensor_parallel_size: int = 1, prompt_mode: str = "chat",
                 revision: str | None = None, prompt_template: str | None = None):
        from vllm import LLM, SamplingParams
        self.sampling_class = SamplingParams
        self.llm = LLM(model=model, enable_prefix_caching=True,
                       max_model_len=max_model_len, tensor_parallel_size=tensor_parallel_size,
                       revision=revision, tokenizer_revision=revision)
        self.tokenizer = self.llm.get_tokenizer()
        self.max_tokens = max_tokens
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.top_p = top_p
        self.prompt_mode = prompt_mode
        self.prompt_template = prompt_template
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("a fast tokenizer with offset mappings is required for lossless token splitting")

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def split_midpoint(self, text):
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        if len(offsets) <= 1:
            return None
        # Snap to an actual Unicode character boundary: some byte-level tokens
        # share offsets, so splitting decoded token arrays can corrupt text.
        candidates = [(abs(i + 1 - len(offsets) / 2), end)
                      for i, (_, end) in enumerate(offsets) if 0 < end < len(text)]
        if not candidates:
            return None
        _, cut = min(candidates)
        return text[:cut], text[cut:]

    def make_prompt(self, problem, prefix):
        if self.prompt_template is not None:
            # Literal placeholders, so LaTeX braces in few-shot examples survive.
            # Split first: placeholders in user text are never expanded recursively.
            template = self.prompt_template
            pieces = re.split(r"(\{\{problem\}\}|\{\{prefix\}\})", template)
            return "".join(problem if p == "{{problem}}" else prefix if p == "{{prefix}}" else p for p in pieces)
        instruction = "Solve the following problem step by step. Put the final answer in \\boxed{...}.\n\n" + problem
        if self.prompt_mode == "plain":
            return "Question:\n" + instruction + "\n\nSolution:\n" + prefix
        messages = [{"role": "user", "content": instruction}]
        # Render a fixed assistant opening, then append the prefix verbatim.
        # Some template continuation helpers strip its trailing whitespace.
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefix

    def generate(self, problem, prefix, n, seed):
        prompt = self.make_prompt(problem, prefix)
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=self.prompt_mode == "plain")
        if len(token_ids) + self.max_tokens > self.max_model_len:
            return [Completion("", "context_length", 0) for _ in range(n)]
        params = self.sampling_class(n=n, seed=seed, temperature=self.temperature,
                                    top_p=self.top_p, max_tokens=self.max_tokens)
        outputs = self.llm.generate([{"prompt_token_ids": token_ids}], params, use_tqdm=False)
        return [Completion(x.text, x.finish_reason or "unknown", self.count_tokens(x.text))
                for request in outputs for x in request.outputs]

