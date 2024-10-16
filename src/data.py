import json
from pathlib import Path

SAMPLES_PATH = Path(__file__).parent / "samples"


models = [
    "NousResearch/Nous-Hermes-llama-2-7b",  # 32,000 tokens vocabulary
    "gpt2",  # 50,257 tokens vocabulary
    "NousResearch/Hermes-3-Llama-3.1-8B",  # 128,256 tokens vocabulary
    "unsloth/gemma-2-2b-it-bnb-4bit",  # 256,128 tokens vocabulary
]


regex_cases = {
    "Phone Number": {
        "regex": r"\d{3}-\d{3}-\d{4}",
        "samples": json.load(open(SAMPLES_PATH / "phone_number.json")),
    },
    "URL": {
        "regex": r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?",
        "samples": json.load(open(SAMPLES_PATH / "url.json")),
    },
    "GSM8K": {
        "regex": r"A: [\w \.\*\-=\+,\?/]{10,50}\. The answer is [1-9][0-9]{0,9}\.",
        "samples": json.load(open(SAMPLES_PATH / "gsm8k.json")),
        # gsm8k.json attribution: https://huggingface.co/datasets/thesven/gsm8k-reasoning
    },
    "Complex string": {
        "regex": r"(0|[1-9][0-9]*)|true|false|([a-zA-Z_][a-zA-Z_0-9]*)",
        "samples": json.load(open(SAMPLES_PATH / "complex_str.json")),
    },
    "Long integer": {
        "regex": r"\+[1-9]\d{1,14}",
        "samples": json.load(open(SAMPLES_PATH / "long_integer.json")),
    },
}


json_cases = {
    "RPG character": {
        "schema": {
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
        "samples": list(
            map(json.dumps, json.load(open(SAMPLES_PATH / "rpg_characters.json")))
        ),
    },
    "Simple nested schema": {
        "schema": {
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
        "samples": list(
            map(json.dumps, json.load(open(SAMPLES_PATH / "recording_schema.json")))
        ),
    },
}
