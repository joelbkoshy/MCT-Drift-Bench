"""
CAS Detector Module — Cognitive Attentional Syndrome Classification.

Classifies patient narrative input into CAS marker categories based on
the metacognitive model (Wells, 2009; Winter et al., 2019):
  - Rumination: repetitive negative thinking about causes/meanings
  - Threat Monitoring: heightened attention scanning for danger
  - Avoidance: maladaptive coping through withdrawal/suppression
  - Neutral: no CAS markers detected

Uses rule-based keyword/phrase matching for deterministic, explainable
classification. No LLM involvement in classification.
"""

from dataclasses import dataclass


@dataclass
class CASResult:
    primary_category: str     # "rumination", "threat_monitoring", "avoidance", "neutral"
    scores: dict              # category -> match count
    matched_phrases: dict     # category -> list of matched phrases
    confidence: str           # "high", "medium", "low"


# CAS marker phrase dictionaries — each phrase is a linguistic indicator
# of the corresponding CAS process.
CAS_MARKERS = {
    "rumination": [
        # ── classic rumination phrasing ──
        "i keep thinking",
        "i can't stop thinking",
        "i cannot stop thinking",
        "can't get it out of my head",
        "can't get it out of my mind",
        "over and over",
        "going over it",
        "replaying",
        "replay",
        "dwelling",
        "stuck in a loop",
        "keep asking myself",
        "keep going back",
        "keep analyzing",
        "constantly analyze",
        "why do i feel",
        "why am i",
        "why can't i",
        "why cannot i",
        "what went wrong",
        "what is wrong with me",
        "what did i do wrong",
        "keep comparing",
        "circling back",
        "same thoughts",
        "same problems",
        "over and over again",
        "keep trying to understand",
        "keep wondering",
        "i go through these",
        "again and again",
        "spend hours thinking",
        "spend most of my day thinking",
        "keeps returning to",
        "mind keeps going back",
        "thoughts keep circling",
        "going over",
        "fixated on",
        "stuck in thought",
        "self-examination",
        "self-scrutiny",
        "mental loop",
        "second-guessing",
        "keep returning",
        # ── real-world / colloquial rumination ──
        "overthink",
        "over think",
        "obsessing",
        "obsess over",
        "ruminating",
        "ruminate",
        "can't move on",
        "cant move on",
        "can't let go",
        "cant let go",
        "keep blaming myself",
        "keep beating myself",
        "haunts me",
        "haunting me",
        "always wondering",
        "think about it constantly",
        "think about it all the time",
        "i always think about",
        "stuck in my head",
        "thoughts won't stop",
        "thoughts wont stop",
        "can't shake",
        "cant shake",
        "never stop thinking",
        "keeps coming back",
        "thought loop",
        "broken record",
        "what's wrong with me",
        "whats wrong with me",
        "kicking myself",
        "beating myself up",
        "my mind won't stop",
        "my mind wont stop",
        "i can't help but think",
        "i cant help but think",
        "my brain won't shut",
        "my brain wont shut",
        "my mind won't shut",
        "my mind wont shut",
        "keeps eating at me",
        "eating me up",
        "gnawing at me",
        "nagging thought",
        "constantly thinking",
        "always think about",
        "can't stop wondering",
        "cant stop wondering",
    ],
    "threat_monitoring": [
        # ── classic threat monitoring ──
        "what if",
        "something bad",
        "i'm worried that",
        "i am worried that",
        "watching for signs",
        "scanning for",
        "on edge",
        "waiting for the next bad",
        "assume it is bad",
        "expect the worst",
        "disaster",
        "danger",
        "something wrong",
        "constantly monitoring",
        "stay vigilant",
        "let my guard down",
        "warning signs",
        "afraid of finding",
        "sign of disapproval",
        "keep watching",
        "something feels wrong",
        "around the corner",
        "falls apart",
        "terrible will happen",
        "braced for",
        "anticipation of",
        "high alert",
        "hyperaware",
        # ── real-world / colloquial threat monitoring ──
        "anxiety",
        "anxious",
        "paranoid",
        "paranoia",
        "panic attack",
        "panicking",
        "dreading",
        "i dread",
        "constant fear",
        "always afraid",
        "always scared",
        "fear of",
        "scared that",
        "worried about",
        "terrified",
        "can't relax",
        "cant relax",
        "waiting for something bad",
        "feel unsafe",
        "feeling unsafe",
        "nervous wreck",
        "on guard",
        "hypervigilant",
    ],
    "avoidance": [
        # ── classic avoidance ──
        "i just can't",
        "i just cannot",
        "i give up",
        "there's no point",
        "there is no point",
        "do not see the point",
        "don't see the point",
        "stopped going",
        "stopped doing",
        "stopped answering",
        "have been avoiding",
        "cannot face",
        "can't face",
        "push away",
        "try not to think",
        "go through the motions",
        "easier to just",
        "given up",
        "nothing feels worth",
        "stay in bed",
        "calling in sick",
        "turned down",
        "suppress",
        "push them away",
        "not worth the effort",
        "withdrawn from",
        "retreated",
        "disengagement",
        "engineered my life",
        # ── real-world / colloquial avoidance ──
        "isolating",
        "isolated myself",
        "isolate myself",
        "avoiding",
        "i avoid",
        "shut myself",
        "shut everyone",
        "shut people out",
        "shutting down",
        "numb",
        "numbness",
        "don't want to go",
        "dont want to go",
        "don't want to do",
        "dont want to do",
        "don't want to see",
        "dont want to see",
        "don't want to talk",
        "dont want to talk",
        "don't want to leave",
        "dont want to leave",
        "can't be bothered",
        "cant be bothered",
        "don't care anymore",
        "dont care anymore",
        "hiding from",
        "hiding away",
        "running away",
        "stopped trying",
        "stopped caring",
        "lost interest",
        "lost motivation",
        "no motivation",
        "no energy",
        "apathy",
        "apathetic",
        "withdrawn",
        "detached",
        "pushing people away",
        "pushing everyone away",
        "escape from",
        "escaping",
        "gave up",
        "what's the point",
        "whats the point",
        "nothing matters",
    ],
}


def detect_cas(text: str) -> CASResult:
    """
    Classify patient text into CAS marker categories using phrase matching.
    
    Args:
        text: Patient narrative input text.
        
    Returns:
        CASResult with primary category, scores, matched phrases, and confidence.
    """
    text_lower = text.lower()
    scores = {}
    matched_phrases = {}

    for category, phrases in CAS_MARKERS.items():
        matches = [p for p in phrases if p in text_lower]
        scores[category] = len(matches)
        matched_phrases[category] = matches

    # Determine primary category
    max_score = max(scores.values())

    if max_score == 0:
        primary = "neutral"
        confidence = "high"
    else:
        # Find category with highest match count
        primary = max(scores, key=scores.get)

        # Determine confidence based on margin between top two
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            margin = sorted_scores[0] - sorted_scores[1]
        else:
            margin = sorted_scores[0]

        if margin >= 2:
            confidence = "high"
        elif margin >= 1:
            confidence = "medium"
        else:
            confidence = "low"

    return CASResult(
        primary_category=primary,
        scores=scores,
        matched_phrases=matched_phrases,
        confidence=confidence,
    )
