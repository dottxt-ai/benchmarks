"""Benchmark the lm-format-enforcer library."""
from lmformatenforcer import RegexParser, TokenEnforcer
from lmformatenforcer.integrations.transformers import (
    build_token_enforcer_tokenizer_data,
)
from transformers import AutoTokenizer

models = [
    "meta-llama/Llama-2-7b-hf",  # 32,000 tokens vocabulary
    "gpt2",  # 50,257 tokens vocabulary
    "meta-llama/Meta-Llama-3.1-8B-Instruct",  # 128,256 tokens vocabulary
    "google/gemma-2-2b-it",  # 256,128 tokens vocabulary
]

case = [
    (r"\d{3}-\d{2}-\d{4}", "203-22-1234"),
    (
        r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?",
        "https://github.com/outlines-dev/outlines",
    ),
    (
        r"A: [\w \.\*\-=\+,\?/]{10,30}\. The answer is [1-9][0-9]{0,9}\.",
        "A: Some thoughts before answering. The answer is 42.",
    ),
]


class LMFormatEnforcer:
    params = [models, case]
    param_names = ["model", "regex"]
    timeout = 600

    def setup(self, model, _):
        """Set up the benchmark.

        We convert the tokenizer during set up as this only
        needs to be done once for a given model.

        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, clean_up_tokenization_spaces=True
        )
        self.tokenizer_data = build_token_enforcer_tokenizer_data(self.tokenizer)

    def time_lfe(self, _, regex):
        regex_string, regex_example = regex
        regex_example_tokens = self.tokenizer.encode(regex_example)

        parser = RegexParser(regex_string)
        token_enforcer = TokenEnforcer(self.tokenizer_data, parser)

        for i in range(len(regex_example_tokens)):
            _ = token_enforcer.get_allowed_tokens(regex_example_tokens[: i + 1])
