"""Programmatic neutrality and anti-recommendation guardrails.

Enforces Alpha Auditor's central principle:
'Alpha Auditor conducts the research, but the user interprets the results.'
"""

from __future__ import annotations

import re

PROHIBITED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(you should buy|you should sell|we recommend buying|we recommend selling|strong buy|strong sell)\b", re.I), "explicit trade recommendation"),
    (re.compile(r"\b(is a good strategy|is a bad strategy|is a great strategy|is a terrible strategy)\b", re.I), "subjective strategy quality verdict"),
    (re.compile(r"\b(strategy is profitable|strategy is unprofitable|proven to be profitable|guaranteed to profit)\b", re.I), "overall profitability verdict"),
    (re.compile(r"\b(recommended for investment|avoid this strategy|safe investment|sure bet)\b", re.I), "advisory language"),
    (re.compile(r"\b(likely to succeed in the future|will outperform|will beat the market)\b", re.I), "predictive performance claim"),
]


def check_neutrality(text: str) -> tuple[bool, list[str]]:
    """Return (is_compliant, list_of_violations) for a text string."""
    violations: list[str] = []
    for pattern, reason in PROHIBITED_PATTERNS:
        if pattern.search(text):
            violations.append(reason)
    return len(violations) == 0, violations


def sanitize_research_notes(notes: list[str]) -> list[str]:
    """Ensure all research notes and methodological statements remain strictly objective."""
    cleaned: list[str] = []
    for note in notes:
        ok, _ = check_neutrality(note)
        if ok:
            cleaned.append(note)
        else:
            s = note
            for pattern, _ in PROHIBITED_PATTERNS:
                s = pattern.sub("[neutralized evaluative phrase]", s)
            cleaned.append(s)
    return cleaned
