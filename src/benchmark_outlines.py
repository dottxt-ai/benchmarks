"""Benchmark the Outlines library."""
import json

import outlines.caching as caching
import torch
from outlines.fsm.guide import RegexGuide
from outlines.fsm.json_schema import build_regex_from_schema
from outlines.models.transformers import TransformerTokenizer
from transformers import AutoTokenizer

from .data import json_cases, models, regex_cases


class OutlinesBenchmark:
    guide_class = RegexGuide

    def do_setup(self, model, samples):
        """Set up the benchmark.

        We JIT-compile Numba functions and convert the vocabulary
        during set up as this only need to be ever done once.

        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, clean_up_tokenization_spaces=True
        )
        self.tokenizer = TransformerTokenizer(self.tokenizer)
        self.guide_class("a", self.tokenizer)  # JIT-compile and convert the vocabulary

        self.all_tokenized_samples = [
            self.tokenizer.encode(sample)[0][0] for sample in samples
        ]

    def _exhaust_samples(self, guide):
        state = guide.initial_state
        for sample_tokens in self.all_tokenized_samples:
            for token in sample_tokens:
                if isinstance(token, torch.Tensor):
                    token = token.item()
                state = guide.get_next_state(state, token)
                _ = guide.get_next_instruction(state)

    def _get_first_token(self, guide):
        """Get first token to verify lazy index is fully warmed up"""
        state = guide.get_next_state(
            guide.initial_state, self.all_tokenized_samples[0][0]
        )
        _ = guide.get_next_instruction(state)

    def teardown(self, *args):
        caching.clear_cache()


class OutlinesRegex(OutlinesBenchmark):
    params = [models, regex_cases.keys()]
    param_names = ["model", "regex_name"]
    timeout = 1200

    def setup(self, model, regex_name):
        samples = regex_cases[regex_name]["samples"]
        self.do_setup(model, samples)

    def time_outlines_total(self, _, regex_name):
        regex_string = regex_cases[regex_name]["regex"]
        guide = self.guide_class(regex_string, self.tokenizer)
        self._exhaust_samples(guide)

    def time_outlines_first_token(self, _, regex_name):
        regex_string = regex_cases[regex_name]["regex"]
        guide = self.guide_class(regex_string, self.tokenizer)
        self._get_first_token(guide)


class OutlinesRegexRunTime(OutlinesBenchmark):
    """Class which warms-up Guide in setup steps"""

    params = [models, regex_cases.keys()]
    param_names = ["model", "regex_name"]
    timeout = 1200

    def setup(self, model, regex_name):
        samples = regex_cases[regex_name]["samples"]
        self.do_setup(model, samples)

        # ensure warmed up so we're only measuring runtime
        regex_string = regex_cases[regex_name]["regex"]
        self.guide = self.guide_class(regex_string, self.tokenizer)
        self._get_first_token(self.guide)

    def time_outlines_runtime(self, *args):
        self._exhaust_samples(self.guide)


class OutlinesJsonSchema(OutlinesBenchmark):
    json_from_regex_fn = lambda self, schema: build_regex_from_schema(schema)

    params = [models, json_cases.keys()]
    param_names = ["model", "json_schema_name"]
    timeout = 1200

    def setup(self, model, json_schema_name):
        samples = json_cases[json_schema_name]["samples"]
        self.do_setup(model, samples)

    def time_outlines_total(self, _, json_schema_name):
        json_string = json_cases[json_schema_name]["schema"]
        regex_string = self.json_from_regex_fn(json.dumps(json_string))
        guide = self.guide_class(regex_string, self.tokenizer)
        self._exhaust_samples(guide)

    def time_outlines_first_token(self, _, json_schema_name):
        json_string = json_cases[json_schema_name]["schema"]
        regex_string = self.json_from_regex_fn(json.dumps(json_string))
        guide = self.guide_class(regex_string, self.tokenizer)
        self._get_first_token(guide)


class OutlinesJsonSchemaRunTime(OutlinesBenchmark):
    """Class which warms-up Guide in setup steps"""

    json_from_regex_fn = lambda self, schema: build_regex_from_schema(schema)

    params = [models, json_cases.keys()]
    param_names = ["model", "json_schema_name"]
    timeout = 1200

    def setup(self, model, json_schema_name):
        samples = json_cases[json_schema_name]["samples"]
        self.do_setup(model, samples)

        # ensure warmed up so we're only measuring runtime
        json_string = json_cases[json_schema_name]["schema"]
        regex_string = self.json_from_regex_fn(json.dumps(json_string))
        self.guide = self.guide_class(regex_string, self.tokenizer)
        self._get_first_token(self.guide)

    def time_outlines_runtime(self, *args):
        self._exhaust_samples(self.guide)
