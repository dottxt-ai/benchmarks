import json

from outlines_core.fsm.guide import RegexGuide
from outlines_core.fsm.json_schema import build_regex_from_schema
from outlines_core.models.transformers import TransformerTokenizer
from transformers import AutoTokenizer

from .data import json_cases, models, regex_cases


class OutlinesCoreRegex:
    params = [models, regex_cases]
    param_names = ["model", "regex"]
    timeout = 600

    def setup(self, model, _):
        """Set up the benchmark.

        We JIT-compile Numba functions and convert the vocabulary
        during set up as this only need to be ever done once.

        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, clean_up_tokenization_spaces=True
        )
        self.tokenizer = TransformerTokenizer(self.tokenizer)

    def time_outlines_core(self, _, regex):
        """Measure generation time with Outlines.

        Outlines' generation time is split between compiling an index for each
        regular expression, and walking this index while generating tokens.

        """
        regex_string, regex_example = regex["regex"], regex["example"]
        regex_example_tokens = self.tokenizer.encode(regex_example)[0][0]
        guide = RegexGuide(regex_string, self.tokenizer)

        state = 0
        for token in regex_example_tokens:
            _ = guide.get_next_instruction(state)
            state = guide.get_next_state(state, token)


class OutlinesCoreJsonSchema:
    params = [models, json_cases]
    param_names = ["model", "json"]
    timeout = 600

    def setup(self, model, _):
        """Set up the benchmark.

        We JIT-compile Numba functions and convert the vocabulary
        during set up as this only need to be ever done once.

        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, clean_up_tokenization_spaces=True
        )
        self.tokenizer = TransformerTokenizer(self.tokenizer)

    def time_outlines_core(self, _, json_case):
        """Measure generation time with Outlines.

        Outlines' generation time is split between compiling an index for each
        regular expression, and walking this index while generating tokens.

        """
        json_string, json_example = json_case["schema"], json_case["example"]
        json_example_tokens = self.tokenizer.encode(json_example)[0][0]

        regex_string = build_regex_from_schema(json.dumps(json_string))
        guide = RegexGuide(regex_string, self.tokenizer)

        state = 0
        for token in json_example_tokens:
            _ = guide.get_next_instruction(state)
            state = guide.get_next_state(state, token)
