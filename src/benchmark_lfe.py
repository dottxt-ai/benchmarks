"""Benchmark the lm-format-enforcer library."""
from lmformatenforcer import JsonSchemaParser, RegexParser, TokenEnforcer
from lmformatenforcer.integrations.transformers import (
    build_token_enforcer_tokenizer_data,
)
from transformers import AutoTokenizer

from .data import json_cases, models, regex_cases


class LMFormatEnforcerRegex:
    params = [models, regex_cases]
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
        regex_string, regex_example = regex["regex"], regex["example"]
        regex_example_tokens = self.tokenizer.encode(regex_example)

        parser = RegexParser(regex_string)
        token_enforcer = TokenEnforcer(self.tokenizer_data, parser)

        for i in range(len(regex_example_tokens)):
            _ = token_enforcer.get_allowed_tokens(regex_example_tokens[: i + 1])


class LMFormatEnforcerJsonSchema:
    params = [models, json_cases]
    param_names = ["model", "json"]
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

    def time_lfe(self, _, json):
        json_string, json_example = json["schema"], json["example"]
        json_example_tokens = self.tokenizer.encode(json_example)

        parser = JsonSchemaParser(json_string)
        token_enforcer = TokenEnforcer(self.tokenizer_data, parser)

        for i in range(len(json_example_tokens)):
            _ = token_enforcer.get_allowed_tokens(json_example_tokens[: i + 1])
