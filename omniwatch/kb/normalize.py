"""Location Phrase Normalization for Clarify-Once Memory.

Responsibility:
- Normalizes location referent phrases: lowercase, strip punctuation, collapse whitespace.
- Drops leading articles ('the', 'a', 'an') and trailing suffixes ('camera', 'cam', 'area', 'zone').
- Singularizes simple plural forms.
- Avoids fuzzy matching (e.g. 'front gate' is NOT silently mapped to 'main gate').

Exposed Interfaces:
- def normalize_phrase(phrase: str) -> str
"""

from __future__ import annotations

import re


def normalize_phrase(phrase: str) -> str:
    """Normalizes location referent phrase per docs/03_CLARIFY_ONCE_MEMORY_SPEC.md.

    - lowercase, strip punctuation, collapse whitespace
    - drop leading articles ('the', 'a', 'an')
    - drop trailing suffixes ('camera', 'cam', 'area', 'zone')
    - singularize simple plurals
    - deterministic: no fuzzy merging
    """
    if not phrase:
        return ""
    text = phrase.lower().strip()
    # Replace hyphens and punctuation with space
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Drop leading articles repeatedly
    changed = True
    while changed:
        changed = False
        for article in ("the ", "a ", "an "):
            if text.startswith(article):
                text = text[len(article) :].strip()
                changed = True

    # Drop trailing area/camera words repeatedly
    changed = True
    while changed:
        changed = False
        for suffix in (" camera", " cam", " area", " zone"):
            if text.endswith(suffix):
                text = text[: -len(suffix)].strip()
                changed = True

    # Singularize simple plurals
    words = text.split()
    singular_words = []
    for w in words:
        if len(w) > 3:
            if w.endswith("ies"):
                w = w[:-3] + "y"
            elif w.endswith("es") and w[:-2].endswith(("ch", "sh", "ss", "x", "z")):
                w = w[:-2]
            elif w.endswith("s") and not w.endswith("ss"):
                w = w[:-1]
        singular_words.append(w)
    text = " ".join(singular_words)

    return text

