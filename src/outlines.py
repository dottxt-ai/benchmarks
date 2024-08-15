"""Benchmark the Outlines library."""
import json

from transformers import AutoTokenizer

from outlines.fsm.guide import RegexGuide
from outlines.fsm.json_schema import build_regex_from_schema
from outlines.models.transformers import TransformerTokenizer

models = [
    "meta-llama/Llama-2-7b-hf",  # 32,000 tokens vocabulary
    "gpt2",  # 50,257 tokens vocabulary
    "meta-llama/Meta-Llama-3.1-8B-Instruct",  # 128,256 tokens vocabulary
    "google/gemma-2-2b-it",  # 256,128 tokens vocabulary
]

regex_case = [
    (r"\d{3}-\d{2}-\d{4}", "203-22-1234"),
    (
        r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?",
        "https://github.com/outlines-dev/outlines",
    ),
    (
        r"A: [\w \.\*\-=\+,\?/]{10,50}\. The answer is [1-9][0-9]{0,9}\.",
        "A: Some thoughts before answering. The answer is 42.",
    ),
]


class OutlinesRegex:
    params = [models, regex_case]
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
        RegexGuide("a", self.tokenizer)  # JIT-compile and convert the vocabulary

    def time_outlines(self, _, regex):
        """Measure generation time with Outlines.

        Outlines' generation time is split between compiling an index for each
        regular expression, and walking this index while generating tokens.

        """
        regex_string, regex_example = regex
        regex_example_tokens = self.tokenizer.encode(regex_example)[0][0]
        guide = RegexGuide(regex_string, self.tokenizer)

        state = 0
        for token in regex_example_tokens:
            _ = guide.get_next_instruction(state)
            state = guide.get_next_state(state, token)


json_case = [
    (
        {
            "$defs": {
                "Armor": {
                    "enum": ["leather", "chainmail", "plate"],
                    "title": "Armor",
                    "type": "string",
                }
            },
            "properties": {
                "name": {"maxLength": 10, "title": "Name", "type": "string"},
                "age": {"title": "Age", "type": "integer"},
                "armor": {"$ref": "#/$defs/Armor"},
                "strength": {"title": "Strength", "type": "integer"},
            },
            "required": ["name", "age", "armor", "strength"],
            "title": "Character",
            "type": "object",
        },
        """{'name': 'Super Warrior', 'age': 26,  'armor': 'leather', 'armor': 10}""",
    )
]


class OutlinesJsonSchema:
    params = [models, json_case]
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
        RegexGuide("a", self.tokenizer)  # JIT-compile and convert the vocabulary

    def time_outlines(self, _, json_case):
        """Measure generation time with Outlines.

        Outlines' generation time is split between compiling an index for each
        regular expression, and walking this index while generating tokens.

        """
        json_string, json_example = json_case
        json_example_tokens = self.tokenizer.encode(json_example)[0][0]

        regex_string = build_regex_from_schema(json.dumps(json_string))
        guide = RegexGuide(regex_string, self.tokenizer)

        state = 0
        for token in json_example_tokens:
            _ = guide.get_next_instruction(state)
            state = guide.get_next_state(state, token)
