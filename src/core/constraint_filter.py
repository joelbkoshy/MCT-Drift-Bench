"""
Constraint Filter Module — Post-Generation MCT Compliance Checker.

After the LLM generates a draft response, this module checks whether
the output violates any MCT principles. If violations are detected,
the response is flagged for regeneration or clinician review.

This is a deterministic, rule-based filter — no LLM involvement.
"""

import re
from dataclasses import dataclass


@dataclass
class ConstraintResult:
    is_compliant: bool
    violations: list        # list of violation descriptions
    violation_count: int
    action: str             # "approve", "flag_for_review", "reject"


# Patterns that indicate MCT violations in LLM output
VIOLATION_PATTERNS = {
    "why_question": {
        "pattern": r"\bwhy\s+(do|did|are|is|have|has|would|does|don'?t|can'?t)\s+you\b",
        "description": "Contains 'why' question — promotes rumination",
    },
    "content_exploration": {
        "pattern": r"\b(tell me more about|can you describe|what happened|explain what|share more|elaborate on)\b",
        "description": "Explores depressive content — violates process-over-content principle",
    },
    "cognitive_restructuring": {
        "pattern": r"\b(evidence for|evidence against|is that really true|thought record|cognitive distortion|reframe|alternative thought|another way to think)\b",
        "description": "Uses CBT cognitive restructuring — MCT does not challenge thought content",
    },
    "rumination_validation": {
        "pattern": r"\b(it'?s? (understandable|natural|normal) (that you|to) (keep thinking|dwell|ruminate|go over|worry))\b",
        "description": "Validates rumination process — MCT discourages extended engagement",
    },
    "reassurance": {
        "pattern": r"\b(things will (get better|work out|be okay|improve)|i'?m sure|don'?t worry|it'?ll be fine)\b",
        "description": "Provides reassurance about content — reinforces threat monitoring",
    },
    "emotional_elaboration": {
        "pattern": r"\b(how does that make you feel|what emotions|describe your feelings|sit with that feeling)\b",
        "description": "Encourages emotional elaboration — MCT focuses on process not emotional content",
    },
}

# Positive indicators — MCT-consistent elements we want to see
MCT_POSITIVE_INDICATORS = {
    "process_labeling": r"\b(rumination|threat monitoring|avoidance|thinking pattern|thinking process|CAS|cognitive attentional)\b",
    "detached_mindfulness": r"\b(notice|observe|watch|let it pass|without engaging|detached|mindful|clouds|stream|float)\b",
    "att_reference": r"\b(attention training|shift.{0,10}attention|attentional control|attention exercise|ATT)\b",
    "postponement": r"\b(postpone|delay|worry period|designated time|set aside)\b",
    "metacognitive_question": r"\b(is this (thinking|worry|thought).{0,20}(helpful|useful|productive)|what happens if you|can you let)\b",
}


def check_constraints(llm_output: str) -> ConstraintResult:
    """
    Check whether an LLM-generated response complies with MCT principles.
    
    Args:
        llm_output: The draft response text from the LLM.
        
    Returns:
        ConstraintResult with compliance status, violations, and action.
    """
    output_lower = llm_output.lower()
    violations = []

    for name, rule in VIOLATION_PATTERNS.items():
        if re.search(rule["pattern"], output_lower):
            violations.append(f"[{name}] {rule['description']}")

    violation_count = len(violations)

    if violation_count == 0:
        action = "approve"
        is_compliant = True
    elif violation_count <= 1:
        action = "flag_for_review"
        is_compliant = False
    else:
        action = "reject"
        is_compliant = False

    return ConstraintResult(
        is_compliant=is_compliant,
        violations=violations,
        violation_count=violation_count,
        action=action,
    )


def score_mct_indicators(llm_output: str) -> dict:
    """
    Count positive MCT indicators present in the LLM output.
    Used as part of the fidelity scoring system.
    
    Args:
        llm_output: The draft response text from the LLM.
        
    Returns:
        Dict mapping indicator name to boolean (present or not).
    """
    output_lower = llm_output.lower()
    indicators = {}
    for name, pattern in MCT_POSITIVE_INDICATORS.items():
        indicators[name] = bool(re.search(pattern, output_lower))
    return indicators
