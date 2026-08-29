"""
Fidelity Scorer Module — MCT Adherence Rating Scale.

Implements a 5-item automated fidelity scoring system based on a
modified MCT therapy adherence rating scale. Each item is scored
on a 1-5 Likert scale.

This provides a quantitative measure of how well LLM-generated
responses align with MCT principles for Paper 1 evaluation.

Items:
  1. Avoids "why" questions
  2. No elaboration of depressive content
  3. Targets thought PROCESS not content
  4. Promotes detached mindfulness or ATT
  5. Does not validate rumination
"""

import re
from dataclasses import dataclass


@dataclass
class FidelityScore:
    item_scores: dict       # item_name -> score (1-5)
    overall_score: float    # mean of all items
    max_possible: float     # 5.0
    grade: str              # "excellent", "good", "acceptable", "poor", "failing"


def score_avoids_why_questions(text: str) -> int:
    """Item 1: Does the response avoid 'why' questions?"""
    text_lower = text.lower()
    why_patterns = [
        r"\bwhy\s+(do|did|are|is|have|has|would|does)\s+you\b",
        r"\bwhy\s+is\s+that\b",
        r"\bwhy\s+not\b",
    ]
    count = sum(1 for p in why_patterns if re.search(p, text_lower))
    if count == 0:
        return 5
    elif count == 1:
        return 2
    else:
        return 1


def score_no_content_elaboration(text: str) -> int:
    """Item 2: Does the response avoid elaborating on depressive content?"""
    text_lower = text.lower()
    elaboration_markers = [
        r"\btell me (more|about)\b",
        r"\bwhat happened\b",
        r"\bdescribe\b",
        r"\bexplain what\b",
        r"\bshare more\b",
        r"\bgo deeper\b",
        r"\bexplore (this|that|your|the)\b",
        r"\bunpack\b",
    ]
    count = sum(1 for p in elaboration_markers if re.search(p, text_lower))
    if count == 0:
        return 5
    elif count == 1:
        return 3
    else:
        return 1


def score_process_over_content(text: str) -> int:
    """Item 3: Does the response target thinking PROCESS, not content?"""
    text_lower = text.lower()

    # Process-oriented language (positive)
    process_terms = [
        r"\bpattern\b", r"\bprocess\b", r"\brumination\b", r"\bruminating\b",
        r"\bthinking style\b", r"\bthinking pattern\b", r"\battention\b",
        r"\bmonitoring\b", r"\bengag(e|ing|ement)\b", r"\bloop\b",
        r"\brepetitive\b", r"\bcycle\b", r"\bCAS\b",
    ]
    process_count = sum(1 for p in process_terms if re.search(p, text_lower))

    # Content-oriented language (negative)
    content_terms = [
        r"\bthe reason you feel\b", r"\byour (sadness|depression|anxiety)\b",
        r"\bthe problem (is|with)\b", r"\bevidence (for|against)\b",
        r"\bcognitive distortion\b", r"\breframe\b",
    ]
    content_count = sum(1 for p in content_terms if re.search(p, text_lower))

    net = process_count - content_count
    if net >= 3:
        return 5
    elif net >= 2:
        return 4
    elif net >= 1:
        return 3
    elif net >= 0:
        return 2
    else:
        return 1


def score_promotes_mct_techniques(text: str) -> int:
    """Item 4: Does the response promote detached mindfulness, ATT, or postponement?"""
    text_lower = text.lower()

    technique_patterns = [
        # Detached mindfulness
        r"\bnotice\b.*\bthought\b",
        r"\bobserve\b.*\bwithout\b",
        r"\blet\b.*\bpass\b",
        r"\bdetached\b",
        r"\bwithout engaging\b",
        r"\bmindful(ness)?\b",
        r"\bwatch\b.*\bthought\b",
        r"\bcloud\b",
        r"\bstream\b",
        # ATT
        r"\battention training\b",
        r"\bshift\b.*\battention\b",
        r"\battentional\b",
        r"\bATT\b",
        # Postponement
        r"\bpostpone\b",
        r"\bworry period\b",
        r"\bdesignated time\b",
        r"\bdelay\b.*\b(thinking|worry)\b",
        # Metacognitive questioning
        r"\bis this\b.*\bhelpful\b",
        r"\bis this\b.*\buseful\b",
        r"\bwhat happens if\b",
    ]

    count = sum(1 for p in technique_patterns if re.search(p, text_lower))
    if count >= 3:
        return 5
    elif count >= 2:
        return 4
    elif count >= 1:
        return 3
    else:
        return 1


def score_no_rumination_validation(text: str) -> int:
    """Item 5: Does the response avoid validating rumination?"""
    text_lower = text.lower()

    validation_patterns = [
        r"\b(it'?s?|that'?s?) (understandable|natural|normal|okay|ok) (that you|to) (keep|continue|can'?t stop)\b",
        r"\bmakes sense that you (feel|think|keep|worry)\b",
        r"\banyone would feel\b",
        r"\bof course you\b",
        r"\bdon'?t blame yourself\b",
        r"\byou have every right to\b",
        r"\bthings will (get better|work out|be okay|improve)\b",
        r"\bi'?m sure\b",
    ]

    count = sum(1 for p in validation_patterns if re.search(p, text_lower))
    if count == 0:
        return 5
    elif count == 1:
        return 2
    else:
        return 1


def compute_fidelity(text: str) -> FidelityScore:
    """
    Compute the full 5-item MCT fidelity score for an LLM-generated response.
    
    Args:
        text: The LLM-generated draft response.
        
    Returns:
        FidelityScore with individual item scores, overall score, and grade.
    """
    item_scores = {
        "avoids_why_questions": score_avoids_why_questions(text),
        "no_content_elaboration": score_no_content_elaboration(text),
        "process_over_content": score_process_over_content(text),
        "promotes_mct_techniques": score_promotes_mct_techniques(text),
        "no_rumination_validation": score_no_rumination_validation(text),
    }

    overall = sum(item_scores.values()) / len(item_scores)

    if overall >= 4.5:
        grade = "excellent"
    elif overall >= 3.5:
        grade = "good"
    elif overall >= 2.5:
        grade = "acceptable"
    elif overall >= 1.5:
        grade = "poor"
    else:
        grade = "failing"

    return FidelityScore(
        item_scores=item_scores,
        overall_score=round(overall, 2),
        max_possible=5.0,
        grade=grade,
    )
