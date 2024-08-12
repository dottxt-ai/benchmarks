"""Benchmark the Outlines library."""
from transformers import AutoTokenizer

from outlines.fsm.guide import RegexGuide
from outlines.models.transformers import TransformerTokenizer

models = [
    "meta-llama/Llama-2-7b-hf",  # 32,000 tokens vocabulary
    "gpt2",  # 50,257 tokens vocabulary
    "meta-llama/Meta-Llama-3.1-8B-Instruct",  # 128,256 tokens vocabulary
    "google/gemma-2-2b-it",  # 256,128 tokens vocabulary
]

case = (r"\d{3}-\d{2}-\d{4}", "203-22-1234")


class Outlines:
    params = [models, case]
    param_names = ["model", "regex"]
    timeout = 600

    def setup(self, model, _):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, clean_up_tokenization_spaces=True
        )
        self.tokenizer = TransformerTokenizer(self.tokenizer)
        # The purpose of the following line is to force the JIT-compilation
        # of Numba functions, and the type conversion.
        RegexGuide("a", self.tokenizer)

    def time_outlines(self, _, regex):
        regex_string, regex_example = regex
        regex_example_tokens = self.tokenizer.encode(regex_example)[0][0]
        guide = RegexGuide(regex_string, self.tokenizer)

        state = 0
        for token in regex_example_tokens:
            _ = guide.get_next_instruction(state)
            state = guide.get_next_state(state, token)
