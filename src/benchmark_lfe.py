"""Benchmark the lm-format-enforcer library."""
from lmformatenforcer import JsonSchemaParser, RegexParser, TokenEnforcer
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
    (
        "(0|[1-9][0-9]*)|true|false|([a-zA-Z_][a-zA-Z_0-9]*)",
        "AVeryLongStringtoTest1234",
    ),
    (r"\+[1-9]\d{1,14}", "1234567891234"),
]


class LMFormatEnforcerRegex:
    params = [models, regex_case]
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
    ),
    (
        {
            "$schema": "http://json-schema.org/draft-04/schema#",
            "title": "Schema for a recording",
            "type": "object",
            "definitions": {
                "artist": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "number"},
                        "name": {"type": "string"},
                        "functions": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["id", "name", "functions"],
                }
            },
            "properties": {
                "id": {"type": "number"},
                "work": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "number"},
                        "name": {"type": "string"},
                        "composer": {"$ref": "#/definitions/artist"},
                    },
                },
                "recording_artists": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/artist"},
                },
            },
            "required": ["id", "work", "recording_artists"],
        },
        """{'id': 999, 'work': {'id': 1, 'name': 'Strasbourg Saint-Denis', 'composer': 'Roy Hargrove'}, 'recording_artists': [{'id': 2, 'name': 'Roy Hargrove', 'functions': ['Trumpet', 'Singing']}]}""",
    ),
]


class LMFormatEnforcerJsonSchema:
    params = [models, json_case]
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
        json_string, json_example = json
        json_example_tokens = self.tokenizer.encode(json_example)

        parser = JsonSchemaParser(json_string)
        token_enforcer = TokenEnforcer(self.tokenizer_data, parser)

        for i in range(len(json_example_tokens)):
            _ = token_enforcer.get_allowed_tokens(json_example_tokens[: i + 1])
