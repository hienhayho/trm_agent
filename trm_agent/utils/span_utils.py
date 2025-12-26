"""Utility functions for span extraction."""

from dataclasses import dataclass


@dataclass
class SpanPosition:
    """Represents a span position in token indices."""

    start: int  # Token start position (inclusive)
    end: int  # Token end position (inclusive)

    @property
    def is_valid(self) -> bool:
        """Check if span is valid (found in text)."""
        return self.start >= 0 and self.end >= self.start


# Special value for "not found" spans
NO_SPAN = SpanPosition(start=-1, end=-1)


def find_span_in_text(
    text: str,
    value: str,
    case_sensitive: bool = False,
) -> tuple[int, int] | None:
    """Find character-level span of value in text.

    Args:
        text: The text to search in
        value: The value to find
        case_sensitive: Whether to do case-sensitive search

    Returns:
        Tuple of (char_start, char_end) or None if not found.
        char_end is exclusive (like Python slicing).
    """
    if not value or not text:
        return None

    search_text = text if case_sensitive else text.lower()
    search_value = value if case_sensitive else value.lower()

    # Strip whitespace from value for matching
    search_value = search_value.strip()
    if not search_value:
        return None

    start = search_text.find(search_value)
    if start == -1:
        return None

    return (start, start + len(search_value))


def char_span_to_token_span(
    char_start: int,
    char_end: int,
    token_offsets: list[tuple[int, int]],
) -> SpanPosition:
    """Convert character-level span to token-level span.

    Args:
        char_start: Character start position (inclusive)
        char_end: Character end position (exclusive)
        token_offsets: List of (char_start, char_end) for each token

    Returns:
        SpanPosition with token indices (both inclusive)
    """
    if not token_offsets:
        return NO_SPAN

    token_start = -1
    token_end = -1

    for i, (tok_start, tok_end) in enumerate(token_offsets):
        # Skip invalid offsets (e.g., special tokens)
        if tok_start < 0 or tok_end < 0:
            continue

        # Find first token that contains or starts after char_start
        if token_start == -1 and tok_end > char_start:
            token_start = i

        # Find last token that contains char_end
        if tok_start < char_end:
            token_end = i

    if token_start == -1 or token_end == -1:
        return NO_SPAN

    return SpanPosition(start=token_start, end=token_end)


def find_value_token_span(
    value: str,
    full_text: str,
    token_offsets: list[tuple[int, int]],
    case_sensitive: bool = False,
) -> SpanPosition:
    """Find token span for a value in tokenized text.

    Combines find_span_in_text and char_span_to_token_span.

    Args:
        value: The value to find
        full_text: The full text that was tokenized
        token_offsets: Token offset mapping from tokenizer
        case_sensitive: Whether to do case-sensitive search

    Returns:
        SpanPosition with token indices, or NO_SPAN if not found
    """
    # Convert to string if not already
    if not isinstance(value, str):
        value = str(value)

    if not value or not value.strip():
        return NO_SPAN

    char_span = find_span_in_text(full_text, value, case_sensitive)
    if char_span is None:
        return NO_SPAN

    return char_span_to_token_span(char_span[0], char_span[1], token_offsets)
