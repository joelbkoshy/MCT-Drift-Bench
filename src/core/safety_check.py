"""
Safety Check Module — Deterministic Crisis Detection Layer.

This module implements a rule-based, deterministic crisis keyword scanner
that operates BEFORE any LLM processing. It does not use generative
components — all detection is pattern-matching based.

If high-risk content is detected, the LLM is bypassed entirely and the
system escalates to clinician notification + emergency resource display.
"""

import re
from dataclasses import dataclass


@dataclass
class SafetyResult:
    status: str          # "HIGH_RISK", "SAFE"
    matched_keywords: list
    action: str          # "escalate", "proceed"
    message: str


# Crisis keyword patterns — phrases indicating suicidal ideation,
# self-harm, or harm to others. Uses regex for flexible matching.
CRISIS_PATTERNS = [
    # Suicidal ideation — direct
    r"\bkill(ing)?\s+(myself|me)\b",
    r"\bend(ing)?\s+(my\s+life|it\s+all|everything)\b",
    r"\bsuicid\w*\b",
    r"\bdon'?t\s+want\s+to\s+(be\s+(alive|here)|live|exist)\b",
    r"\bdo\s+not\s+want\s+to\s+(be\s+(alive|here)|live|exist)\b",
    r"\bwant\s+to\s+die\b",
    r"\bwish\s+I\s+(was|were)\s+dead\b",
    r"\bbetter\s+off\s+(without\s+me|dead)\b",
    r"\bno\s+reason\s+to\s+(keep\s+going|live|continue)\b",
    r"\bnot\s+worth\s+living\b",

    # Suicidal ideation — planning
    r"\bhave\s+a\s+plan\b",
    r"\bcollecting\s+(pills|medication)\b",
    r"\bwriting\s+letters?\s+(to\s+my\s+family|goodbye)\b",
    r"\bresearching\s+methods?\b",

    # Self-harm
    r"\bhurt\s+myself\b",
    r"\bself[\s-]?harm\w*\b",
    r"\bcutting\b",
    r"\bburn(ed|ing)?\s+(my|myself)\b",
    r"\bharm(ing)?\s+(myself|me)\b",

    # Indirect crisis signals
    r"\beveryone\s+would\s+be\s+better\s+off\s+without\s+me\b",
    r"\bjust\s+a\s+burden\b",
    r"\bcan'?t?\s+go\s+on\b",
    r"\bending\s+it\b",
    r"\bmaking\s+the\s+outside\s+match\b",
]

# Emergency resources displayed to patient on crisis detection
EMERGENCY_RESOURCES = {
    "crisis_hotline": "988 Suicide & Crisis Lifeline: Call or text 988",
    "crisis_text": "Crisis Text Line: Text HOME to 741741",
    "emergency": "Emergency Services: Call 911",
    "international": "International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/",
}


def check_safety(text: str) -> SafetyResult:
    """
    Scan input text for crisis keywords using deterministic pattern matching.
    
    This function runs BEFORE any LLM processing. If any crisis pattern is
    detected, the result instructs the pipeline to bypass LLM generation
    entirely and escalate to the clinician.
    
    Args:
        text: Patient narrative input text.
        
    Returns:
        SafetyResult with status, matched keywords, action, and message.
    """
    text_lower = text.lower()
    matched = []

    for pattern in CRISIS_PATTERNS:
        matches = re.findall(pattern, text_lower)
        if matches:
            matched.append(pattern)

    if matched:
        return SafetyResult(
            status="HIGH_RISK",
            matched_keywords=matched,
            action="escalate",
            message=(
                "Crisis indicators detected in patient input. "
                "LLM bypassed. Clinician notification triggered. "
                "Emergency resources displayed to patient."
            ),
        )

    return SafetyResult(
        status="SAFE",
        matched_keywords=[],
        action="proceed",
        message="No crisis indicators detected. Proceeding to CAS analysis.",
    )


def get_emergency_response() -> str:
    """
    Return the pre-formatted emergency response to display to the patient
    when crisis content is detected. This is a static, clinician-approved
    message — never LLM-generated.
    """
    return (
        "I want you to know that what you're feeling matters, and help is available right now.\n\n"
        "Please reach out to one of these resources immediately:\n"
        f"• {EMERGENCY_RESOURCES['crisis_hotline']}\n"
        f"• {EMERGENCY_RESOURCES['crisis_text']}\n"
        f"• {EMERGENCY_RESOURCES['emergency']}\n\n"
        "Your clinician has been notified and will follow up with you.\n"
        "You do not have to go through this alone."
    )
