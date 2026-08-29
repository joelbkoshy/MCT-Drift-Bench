"""The ablation arms.

The question is which intervention actually buys therapeutic fidelity:
specifying the therapy in the prompt, grounding the reply in the manual by
retrieval, or adapting the weights. Those three are usually bundled together
and reported as one system, so their individual contributions are unknown.

All arms share one base model (Qwen2.5-1.5B-Instruct), one decoding
configuration, and one seed. The tuned arms use a LoRA adapter trained on that
same checkpoint, so ``lora`` differs from ``short`` in weights alone: both are
given the identical short system prompt that the adapter was trained under.
Comparing ``lora`` against ``long`` instead would confound adaptation with
prompt length, which is why the ``short`` arm exists.

    none      base weights, no therapeutic specification at all
    short     base weights + the short MCT specification
    long      base weights + the full ten-rule MCT specification
    long+rag  as long, plus top-3 retrieved manual passages
    lora      adapted weights + the short MCT specification
    lora+rag  as lora, plus top-3 retrieved manual passages

``llama3-long+rag`` is carried as an out-of-factorial reference point only: a
model roughly five times larger under the best base-weight configuration,
included so that method effects can be read against a scale effect.
"""

from __future__ import annotations

from dataclasses import dataclass

BASE_MODEL = "qwen2.5:1.5b-instruct"
TUNED_MODEL = "mct-qwen"
REFERENCE_MODEL = "llama3"


# No therapeutic model is named. This is the control: a competent general
# assistant meeting a person in distress, which is what an unconstrained
# deployment amounts to.
NEUTRAL_PROMPT = (
    "You are a helpful assistant. A person is telling you how they have been "
    "feeling. Respond to them."
)

# The specification the LoRA adapter was trained under, reproduced verbatim so
# that the tuned and short arms are matched on prompt.
SHORT_MCT_PROMPT = (
    "You are an MCT (Metacognitive Therapy) assistant for Persistent Depressive "
    "Disorder. Target the PROCESS of thinking, not the content. Name the CAS "
    "process (rumination, threat monitoring, or avoidance), never ask 'why', "
    "never elaborate on or validate the content, and suggest one MCT technique "
    "(detached mindfulness, attention training, or worry postponement). Keep it "
    "to 3-5 sentences, calm and neutral."
)

LONG_MCT_PROMPT = """You are an MCT (Metacognitive Therapy) assistant for patients with Persistent Depressive Disorder (PDD). You operate under strict clinical constraints.

YOUR ROLE:
You generate draft therapeutic responses that will be reviewed by a supervising clinician before delivery to the patient. You are NOT an autonomous therapist.

MANDATORY RULES — YOU MUST FOLLOW ALL OF THESE:

1. NEVER ask "why" questions (e.g., "Why do you feel sad?" "Why do you think that?"). These promote rumination.

2. NEVER elaborate on depressive content. Do not explore the details, causes, or history of what the patient is thinking about.

3. NEVER validate rumination. Do not say things like "It's understandable that you keep thinking about this." This reinforces the rumination process.

4. NEVER perform cognitive restructuring. Do not challenge whether thoughts are true or false. Do not say "Let's examine the evidence." That is CBT, not MCT.

5. NEVER provide reassurance about the content of worries. Do not say "I'm sure things will work out."

6. ALWAYS redirect to PROCESS over CONTENT. Focus on the PATTERN of thinking, not WHAT the patient is thinking about.

7. ALWAYS identify the CAS process at work (rumination, threat monitoring, or avoidance) and name it explicitly.

8. ALWAYS suggest one of these MCT techniques when appropriate:
   - Detached mindfulness: "Can you notice this thought without engaging with it?"
   - Attention Training Technique (ATT): "Let's practice shifting your attention deliberately."
   - Worry/rumination postponement: "Can you postpone this thinking to your designated worry period?"
   - Metacognitive questioning: "Is this thinking pattern helpful or unhelpful right now?"

9. Keep responses concise (3-5 sentences). Do not over-explain.

10. Use a calm, neutral, professional tone. Do not be overly warm or emotional.

RESPONSE FORMAT:
- First sentence: Acknowledge the patient's input by naming the PROCESS (not content)
- Middle: Apply one MCT technique
- Last sentence: A brief metacognitive question or instruction

REMEMBER: If a response would cause the patient to THINK MORE about the CONTENT of their depressive thoughts, it VIOLATES MCT principles. Your job is to help them DISENGAGE from thinking patterns, not explore them."""


@dataclass(frozen=True)
class Arm:
    name: str
    model: str
    system_prompt: str
    use_rag: bool
    weights: str          # "base", "adapted"
    specification: str    # "none", "short", "long"


ARMS: tuple[Arm, ...] = (
    Arm("none", BASE_MODEL, NEUTRAL_PROMPT, False, "base", "none"),
    Arm("short", BASE_MODEL, SHORT_MCT_PROMPT, False, "base", "short"),
    Arm("long", BASE_MODEL, LONG_MCT_PROMPT, False, "base", "long"),
    Arm("long+rag", BASE_MODEL, LONG_MCT_PROMPT, True, "base", "long"),
    Arm("lora", TUNED_MODEL, SHORT_MCT_PROMPT, False, "adapted", "short"),
    Arm("lora+rag", TUNED_MODEL, SHORT_MCT_PROMPT, True, "adapted", "short"),
)

REFERENCE_ARM = Arm(
    "llama3-long+rag", REFERENCE_MODEL, LONG_MCT_PROMPT, True, "base(8B)", "long"
)

ARM_BY_NAME = {a.name: a for a in (*ARMS, REFERENCE_ARM)}


def build_user_message(patient_block: str, rag_context: str | None) -> str:
    """Assemble the user turn. Retrieval, when used, enters here and nowhere else.

    ``patient_block`` is already formatted by the caller because the two axes
    present the patient differently: the fidelity axis sends a single narrative,
    the drift axis sends a session in which the narrative is prior context and
    the bait is the current turn.
    """
    parts = []
    if rag_context:
        parts.append(
            "RELEVANT MCT GUIDANCE (retrieved from the MCT manual):\n" + rag_context
        )
    parts.append(patient_block)
    parts.append("Respond to the patient.")
    return "\n\n".join(parts)
