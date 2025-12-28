"""TRM Inference utilities."""

from .span_decoder import (
    SpanDecoder,
    DecodedSpan,
    ToolArguments,
    decode_spans,
    extract_unified_fields,
)

__all__ = [
    "SpanDecoder",
    "DecodedSpan",
    "ToolArguments",
    "decode_spans",
    "extract_unified_fields",
]
