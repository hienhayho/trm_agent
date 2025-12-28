"""Utility functions for span extraction."""

import re
import unicodedata
from dataclasses import dataclass, field


# Role priority weights (higher = more preferred)
# user > assistant > tool_response > system
ROLE_WEIGHTS = {
    0: 0.1,   # system
    1: 1.0,   # user (highest priority)
    2: 0.7,   # assistant
    3: 0.5,   # tool_call
    4: 0.3,   # tool_response
}


@dataclass
class SpanPosition:
    """Represents a span position in token indices."""

    start: int  # Token start position (inclusive)
    end: int  # Token end position (inclusive)

    @property
    def is_valid(self) -> bool:
        """Check if span is valid (found in text)."""
        return self.start >= 0 and self.end >= self.start


@dataclass
class WeightedSpan:
    """Represents a span with an associated weight/priority."""

    start: int  # Token start position (inclusive)
    end: int  # Token end position (inclusive)
    weight: float = 1.0  # Priority weight (higher = more preferred)
    char_start: int = -1  # Character start position (for debugging)
    char_end: int = -1  # Character end position (for debugging)

    @property
    def is_valid(self) -> bool:
        """Check if span is valid."""
        return self.start >= 0 and self.end >= self.start

    def to_span_position(self) -> "SpanPosition":
        """Convert to SpanPosition."""
        return SpanPosition(start=self.start, end=self.end)


# Special value for "not found" spans
NO_SPAN = SpanPosition(start=-1, end=-1)


def find_span_in_text(
    text: str,
    value: str,
    case_sensitive: bool = False,
) -> tuple[int, int] | None:
    """Find character-level span of value in text (first occurrence).

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


def find_all_spans_in_text(
    text: str,
    value: str,
    case_sensitive: bool = False,
) -> list[tuple[int, int]]:
    """Find ALL character-level spans of value in text.

    Args:
        text: The text to search in
        value: The value to find
        case_sensitive: Whether to do case-sensitive search

    Returns:
        List of (char_start, char_end) tuples for all occurrences.
        char_end is exclusive (like Python slicing).
    """
    if not value or not text:
        return []

    search_text = text if case_sensitive else text.lower()
    search_value = value if case_sensitive else value.lower()

    # Strip whitespace from value for matching
    search_value = search_value.strip()
    if not search_value:
        return []

    spans = []
    start = 0
    while True:
        pos = search_text.find(search_value, start)
        if pos == -1:
            break
        spans.append((pos, pos + len(search_value)))
        start = pos + 1  # Move past this occurrence

    return spans


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
    """Find token span for a value in tokenized text (first occurrence).

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


def find_all_value_token_spans(
    value: str,
    full_text: str,
    token_offsets: list[tuple[int, int]],
    role_ids: list[int] | None = None,
    case_sensitive: bool = False,
    position_weight: float = 0.3,
    role_weight: float = 0.7,
) -> list[WeightedSpan]:
    """Find ALL token spans for a value with priority weights.

    Weight is computed as:
        weight = position_weight * (char_pos / text_len) + role_weight * role_priority

    Priority: last occurrence + user role > assistant > tool_response > system

    Args:
        value: The value to find
        full_text: The full text that was tokenized
        token_offsets: Token offset mapping from tokenizer
        role_ids: Role ID for each token (optional, for role-based weighting)
        case_sensitive: Whether to do case-sensitive search
        position_weight: Weight for position-based priority (later = higher)
        role_weight: Weight for role-based priority

    Returns:
        List of WeightedSpan objects, sorted by weight (highest first)
    """
    # Convert to string if not already
    if not isinstance(value, str):
        value = str(value)

    if not value or not value.strip():
        return []

    # Find all character-level spans
    char_spans = find_all_spans_in_text(full_text, value, case_sensitive)
    if not char_spans:
        return []

    text_len = len(full_text)
    weighted_spans = []

    for char_start, char_end in char_spans:
        # Convert to token span
        token_span = char_span_to_token_span(char_start, char_end, token_offsets)
        if not token_span.is_valid:
            continue

        # Compute position weight (later = higher, normalized to 0-1)
        pos_score = char_start / text_len if text_len > 0 else 0.0

        # Compute role weight
        role_score = 0.5  # Default if no role_ids
        if role_ids is not None and token_span.start < len(role_ids):
            role_id = role_ids[token_span.start]
            role_score = ROLE_WEIGHTS.get(role_id, 0.5)

        # Combined weight
        weight = position_weight * pos_score + role_weight * role_score

        weighted_spans.append(WeightedSpan(
            start=token_span.start,
            end=token_span.end,
            weight=weight,
            char_start=char_start,
            char_end=char_end,
        ))

    # Sort by weight (highest first)
    weighted_spans.sort(key=lambda s: s.weight, reverse=True)

    return weighted_spans


def find_best_value_token_span(
    value: str,
    full_text: str,
    token_offsets: list[tuple[int, int]],
    role_ids: list[int] | None = None,
    case_sensitive: bool = False,
) -> SpanPosition:
    """Find the BEST token span for a value (highest priority).

    Uses weighted scoring: last occurrence + user role preferred.

    Args:
        value: The value to find
        full_text: The full text that was tokenized
        token_offsets: Token offset mapping from tokenizer
        role_ids: Role ID for each token (optional)
        case_sensitive: Whether to do case-sensitive search

    Returns:
        SpanPosition with token indices, or NO_SPAN if not found
    """
    weighted_spans = find_all_value_token_spans(
        value, full_text, token_offsets, role_ids, case_sensitive
    )

    if not weighted_spans:
        return NO_SPAN

    # Return the highest weighted span
    return weighted_spans[0].to_span_position()


# =============================================================================
# Fuzzy Matching Utilities
# =============================================================================


def normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching.

    - Convert to lowercase
    - Normalize unicode (NFKC)
    - Remove extra whitespace
    - Strip leading/trailing whitespace
    """
    if not text:
        return ""

    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Normalize whitespace (multiple spaces -> single space)
    text = re.sub(r"\s+", " ", text)

    # Strip
    text = text.strip()

    return text


def normalize_phone(phone: str) -> str:
    """Normalize phone number for matching.

    Removes common separators: spaces, dashes, dots, parentheses.
    """
    if not phone:
        return ""

    # Remove common separators
    phone = re.sub(r"[\s\-\.\(\)]+", "", phone)

    return phone


def normalize_for_matching(value: str, field_type: str | None = None) -> str:
    """Normalize a value for fuzzy matching based on field type.

    Args:
        value: The value to normalize
        field_type: Optional field type hint ('phone', 'address', etc.)

    Returns:
        Normalized string
    """
    if not value:
        return ""

    # Apply field-specific normalization
    if field_type == "phone":
        return normalize_phone(value)

    # Default: general text normalization
    return normalize_text(value)


def find_fuzzy_spans_in_text(
    text: str,
    value: str,
    field_type: str | None = None,
) -> list[tuple[int, int]]:
    """Find spans using fuzzy matching with normalization.

    Tries multiple matching strategies:
    1. Exact match (case-insensitive)
    2. Normalized match (whitespace, unicode)
    3. Phone-style match (for phone numbers)
    4. Partial match (value as substring after normalization)

    Args:
        text: The text to search in
        value: The value to find
        field_type: Optional field type for specialized normalization

    Returns:
        List of (char_start, char_end) tuples for all matches.
    """
    if not value or not text:
        return []

    value = value.strip()
    if not value:
        return []

    # Strategy 1: Exact match (case-insensitive)
    spans = find_all_spans_in_text(text, value, case_sensitive=False)
    if spans:
        return spans

    # Strategy 2: Normalized match
    norm_value = normalize_text(value)
    if norm_value != value.lower():
        spans = find_all_spans_in_text(text, norm_value, case_sensitive=False)
        if spans:
            return spans

    # Strategy 3: Phone-style match (remove separators)
    if field_type == "phone" or re.match(r"^[\d\s\-\.\(\)]+$", value):
        phone_value = normalize_phone(value)
        if phone_value and len(phone_value) >= 4:  # Min phone length
            # Search for phone in text with normalized matching
            norm_text = normalize_phone(text)
            # Find position in normalized text
            pos = norm_text.find(phone_value)
            if pos != -1:
                # Map back to original text position (approximate)
                # Count non-digit chars before this position in original
                spans = _find_phone_in_original(text, phone_value)
                if spans:
                    return spans

    # Strategy 4: Try matching words individually for multi-word values
    words = norm_value.split()
    if len(words) > 1:
        # Try to find the longest word
        longest_word = max(words, key=len)
        if len(longest_word) >= 3:  # Min word length
            spans = find_all_spans_in_text(text, longest_word, case_sensitive=False)
            if spans:
                return spans

    return []


def _find_phone_in_original(text: str, normalized_phone: str) -> list[tuple[int, int]]:
    """Find a normalized phone number in original text.

    Scans through text to find sequences of digits that match.
    """
    spans = []
    text_len = len(text)
    phone_len = len(normalized_phone)

    i = 0
    while i < text_len:
        # Find start of digit sequence
        while i < text_len and not text[i].isdigit():
            i += 1

        if i >= text_len:
            break

        start = i
        digits = []
        j = i

        # Collect digits (skipping separators)
        while j < text_len and len(digits) < phone_len:
            if text[j].isdigit():
                digits.append(text[j])
            elif text[j] not in " -.()+":
                break  # Non-separator, non-digit ends the sequence
            j += 1

        # Check if we found the phone
        if "".join(digits) == normalized_phone:
            spans.append((start, j))
            i = j
        else:
            i += 1

    return spans


def find_all_value_token_spans_fuzzy(
    value: str,
    full_text: str,
    token_offsets: list[tuple[int, int]],
    role_ids: list[int] | None = None,
    field_type: str | None = None,
    position_weight: float = 0.3,
    role_weight: float = 0.7,
) -> list[WeightedSpan]:
    """Find ALL token spans using fuzzy matching with fallback.

    First tries exact matching, then falls back to fuzzy matching.

    Args:
        value: The value to find
        full_text: The full text that was tokenized
        token_offsets: Token offset mapping from tokenizer
        role_ids: Role ID for each token (optional)
        field_type: Optional field type for specialized normalization
        position_weight: Weight for position-based priority
        role_weight: Weight for role-based priority

    Returns:
        List of WeightedSpan objects, sorted by weight (highest first)
    """
    # Convert to string if not already
    if not isinstance(value, str):
        value = str(value)

    if not value or not value.strip():
        return []

    # Try exact matching first
    weighted_spans = find_all_value_token_spans(
        value, full_text, token_offsets, role_ids, case_sensitive=False,
        position_weight=position_weight, role_weight=role_weight
    )

    if weighted_spans:
        return weighted_spans

    # Fall back to fuzzy matching
    char_spans = find_fuzzy_spans_in_text(full_text, value, field_type)
    if not char_spans:
        return []

    text_len = len(full_text)
    weighted_spans = []

    for char_start, char_end in char_spans:
        token_span = char_span_to_token_span(char_start, char_end, token_offsets)
        if not token_span.is_valid:
            continue

        # Compute weights (same as exact matching but slightly lower)
        pos_score = char_start / text_len if text_len > 0 else 0.0

        role_score = 0.5
        if role_ids is not None and token_span.start < len(role_ids):
            role_id = role_ids[token_span.start]
            role_score = ROLE_WEIGHTS.get(role_id, 0.5)

        # Slightly lower weight for fuzzy matches
        weight = (position_weight * pos_score + role_weight * role_score) * 0.9

        weighted_spans.append(WeightedSpan(
            start=token_span.start,
            end=token_span.end,
            weight=weight,
            char_start=char_start,
            char_end=char_end,
        ))

    weighted_spans.sort(key=lambda s: s.weight, reverse=True)
    return weighted_spans
