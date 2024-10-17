"""Benchmark the lm-format-enforcer library."""
from lmformatenforcer import JsonSchemaParser, RegexParser, TokenEnforcer
from lmformatenforcer.integrations.transformers import (
    build_token_enforcer_tokenizer_data,
)
from transformers import AutoTokenizer

from .data import json_cases, models, regex_cases


class LMFormatEnforcerBenchmark:
    def do_setup(self, model, samples):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, clean_up_tokenization_spaces=True
        )
        self.tokenizer_data = build_token_enforcer_tokenizer_data(self.tokenizer)
        self.all_tokenized_samples = [
            self.tokenizer.encode(sample) for sample in samples
        ]

    def _exhaust_samples(self, token_enforcer):
        for sample_tokens in self.all_tokenized_samples:
            for i in range(len(sample_tokens)):
                _ = token_enforcer.get_allowed_tokens(sample_tokens[: i + 1])

    def _get_first_token(self, token_enforcer):
        """Get first token to verify lazy index is fully warmed up"""
        _ = token_enforcer.get_allowed_tokens(self.all_tokenized_samples[0][:1])

    def teardown(self, *args):
        del self.tokenizer_data


class LMFormatEnforcerRegex(LMFormatEnforcerBenchmark):
    params = [models, regex_cases.keys()]
    param_names = ["model", "regex_name"]
    timeout = 1200

    def setup(self, model, regex_name):
        samples = regex_cases[regex_name]["samples"]
        self.do_setup(model, samples)

    def _get_enforcer(self, regex_name):
        pattern = regex_cases[regex_name]["regex"]
        parser = RegexParser(pattern)
        return TokenEnforcer(self.tokenizer_data, parser)

    def time_lfe_total(self, _, regex_name):
        enforcer = self._get_enforcer(regex_name)
        self._exhaust_samples(enforcer)

    def time_lfe_first_token(self, _, regex_name):
        enforcer = self._get_enforcer(regex_name)
        self._get_first_token(enforcer)


class LMFormatEnforcerRegexRunTime(LMFormatEnforcerBenchmark):
    """Class which warms-up enforcer in setup steps"""

    _get_enforcer = LMFormatEnforcerRegex._get_enforcer

    params = [models, regex_cases.keys()]
    param_names = ["model", "regex_name"]
    timeout = 1200

    def setup(self, model, regex_name):
        samples = regex_cases[regex_name]["samples"]
        self.do_setup(model, samples)

        # ensure warmed up so we're only measuring runtime
        self.enforcer = self._get_enforcer(regex_name)
        self._get_first_token(self.enforcer)

    def time_lfe_runtime(self, *args):
        self._exhaust_samples(self.enforcer)


class LMFormatEnforcerJsonSchema(LMFormatEnforcerBenchmark):
    params = [models, json_cases.keys()]
    param_names = ["model", "json_schema_name"]
    timeout = 600

    def setup(self, model, json_schema_name):
        samples = json_cases[json_schema_name]["samples"]
        self.do_setup(model, samples)

    def _get_enforcer(self, json_schema_name):
        schema = json_cases[json_schema_name]["schema"]
        parser = JsonSchemaParser(schema)
        return TokenEnforcer(self.tokenizer_data, parser)

    def time_lfe_total(self, _, json_schema_name):
        enforcer = self._get_enforcer(json_schema_name)
        self._exhaust_samples(enforcer)

    def time_lfe_first_token(self, _, json_schema_name):
        enforcer = self._get_enforcer(json_schema_name)
        self._get_first_token(enforcer)


class LMFormatEnforcerJsonSchemaRunTime(LMFormatEnforcerBenchmark):
    """Class which warms-up enforcer in setup steps"""

    _get_enforcer = LMFormatEnforcerJsonSchema._get_enforcer

    params = [models, json_cases.keys()]
    param_names = ["model", "json_schema_name"]
    timeout = 600

    def setup(self, model, json_schema_name):
        samples = json_cases[json_schema_name]["samples"]
        self.do_setup(model, samples)

        # ensure warmed up so we're only measuring runtime
        self.enforcer = self._get_enforcer(json_schema_name)
        self._get_first_token(self.enforcer)

    def time_lfe_runtime(self, *args):
        self._exhaust_samples(self.enforcer)
