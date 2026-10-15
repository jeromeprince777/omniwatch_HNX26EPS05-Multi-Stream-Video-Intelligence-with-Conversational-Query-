"""Unit tests for Location Phrase Normalization (docs/03_CLARIFY_ONCE_MEMORY_SPEC.md).

Requirement: Unit-test with >= 15 cases. No fuzzy auto-merge.
"""

import pytest
from omniwatch.kb.normalize import normalize_phrase


@pytest.mark.parametrize(
    ("input_phrase", "expected"),
    [
        # Case 1: Simple lowercasing
        ("Main Gate", "main gate"),
        # Case 2: Leading article 'the'
        ("the main gate", "main gate"),
        # Case 3: Leading article 'a'
        ("a parking lot", "parking lot"),
        # Case 4: Leading article 'an'
        ("an entrance", "entrance"),
        # Case 5: Trailing 'camera'
        ("main gate camera", "main gate"),
        # Case 6: Trailing 'cam'
        ("lobby cam", "lobby"),
        # Case 7: Trailing 'zone'
        ("loading zone", "loading"),
        # Case 8: Trailing 'area'
        ("reception area", "reception"),
        # Case 9: Combined leading article and trailing camera
        ("the main gate camera", "main gate"),
        # Case 10: Hyphenated words
        ("secret-bunker", "secret bunker"),
        # Case 11: Hyphenated with trailing cam
        ("secret-bunker-cam", "secret bunker"),
        # Case 12: Extra whitespace and punctuation
        ("  North   Entrance !? ", "north entrance"),
        # Case 13: Singularize simple plural 'gates'
        ("main gates", "main gate"),
        # Case 14: Singularize 'lobbies'
        ("east lobbies", "east lobby"),
        # Case 15: Singularize 'doors'
        ("exit doors", "exit door"),
        # Case 16: Empty string
        ("", ""),
        # Case 17: No fuzzy merge ('front gate' is NOT 'main gate')
        ("front gate", "front gate"),
        # Case 18: Word ending in 'ss' should not be truncated (e.g. grass, express)
        ("express lane", "express lane"),
    ],
)
def test_normalize_phrase(input_phrase: str, expected: str):
    assert normalize_phrase(input_phrase) == expected
