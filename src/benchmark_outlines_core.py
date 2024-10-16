from outlines.caching import cache
from outlines_core.fsm.guide import RegexGuide, create_states_mapping
from outlines_core.fsm.json_schema import build_regex_from_schema

from .benchmark_outlines import (
    OutlinesJsonSchema,
    OutlinesJsonSchemaRunTime,
    OutlinesRegex,
    OutlinesRegexRunTime,
)


@cache()
def cached_create_states_mapping(regex_string, tokenizer, *args, **kwargs):
    return create_states_mapping(regex_string, tokenizer, *args, **kwargs)


class CachedOutlinesCoreRegexGuide(RegexGuide):
    """
    Guide to generate text in the language of a regular expression.
    CoreRegexGuide with outlines cache
    """

    @classmethod
    def from_regex(
        cls,
        regex_string: str,
        tokenizer,
        **kwargs,
    ):
        return RegexGuide.from_regex(
            regex_string,
            tokenizer,
            _create_states_mapping=cached_create_states_mapping,
            **kwargs,
        )


class OutlinesCoreRegex(OutlinesRegex):
    guide_class = CachedOutlinesCoreRegexGuide.from_regex


class OutlinesCoreRegexRunTime(OutlinesRegexRunTime):
    guide_class = CachedOutlinesCoreRegexGuide.from_regex


class OutlinesCoreJsonSchema(OutlinesJsonSchema):
    guide_class = CachedOutlinesCoreRegexGuide.from_regex
    json_from_regex_fn = lambda self, schema: build_regex_from_schema(schema)


class OutlinesCoreJsonSchemaRunTime(OutlinesJsonSchemaRunTime):
    guide_class = CachedOutlinesCoreRegexGuide.from_regex
    json_from_regex_fn = lambda self, schema: build_regex_from_schema(schema)
