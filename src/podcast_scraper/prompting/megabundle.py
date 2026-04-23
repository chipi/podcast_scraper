"""Mega-bundle prompt builder (#643).

Single structured JSON prompt that asks the LLM for summary + bullets +
insights + topics + entities in one response. Validated research-backed
in #632 experiment (Anthropic Claude Haiku 4.5 + DeepSeek produce clean
output with full-quality summary; OpenAI/Gemini/Mistral/Grok compress the
summary too much — those use extraction_bundled instead).

Prompt shape kept close to the experiment script
(`scripts/eval/megabundle_experiment.py`) so pipeline and research paths
stay comparable; changes here should be reflected in the research script
and re-benchmarked.
"""

from __future__ import annotations

from typing import Optional, Tuple

# Research-derived sweet-spot counts (from the autoresearch work that fed #590
# and #625). Mirror the defaults used by the standalone GI/KG stages.
DEFAULT_MEGA_BUNDLE_INSIGHTS = 12
DEFAULT_MEGA_BUNDLE_TOPICS = 10
DEFAULT_MEGA_BUNDLE_ENTITIES_MAX = 15

# #652 — four quality rules applied to every extraction prompt. Kept as a
# single string so the exact wording is uniform across providers + the
# deterministic post-extraction validators in gi/kg pipeline stages have a
# clear "what the prompt asked for" anchor.
QUALITY_RULES = (
    "QUALITY RULES — apply to ALL extracted fields:\n"
    "\n"
    "1. Ads / sponsor reads (skip). Do not extract marketing claims, sponsor "
    "reads, subscription pitches, or paid-promotion content. Skip passages "
    "describing a product/service as 'built for X', 'helps you Y', 'runs on Z', "
    "or that name a sponsor product. If a passage is an ad-read or host "
    "disclosure ('this episode is brought to you by …', 'I work at …'), "
    "produce NO insight, NO topic, and NO entity for that range.\n"
    "\n"
    "2. Insight form. Insights must be paraphrased THIRD-PERSON claims "
    "distilled from the transcript. Do NOT copy verbatim dialogue, host patter, "
    "or first-person speaker monologue. Start each insight with a noun + verb "
    "in present tense. Avoid 'we', 'I', 'you', \"let's\", 'okay', or "
    "conversational filler ('you know', 'I mean').\n"
    "\n"
    "3. Topic length. Topics MUST be concise 2–3 word noun phrases. Do NOT use "
    "prepositions ('in', 'of', 'for', 'vs') at start or middle. Do NOT use "
    "event-specific proper-noun compounds ('X succession', 'Y leverage'). "
    "Prefer canonical concept names that could repeat across episodes "
    "('nuclear program', not \"Iran's nuclear program\").\n"
    "\n"
    "4. Entity kind. Default to 'org' if the name refers to a company, show, "
    "podcast, brand, product, publication, or organisation. Reserve 'person' "
    "for individual human names (first + last, or clearly a named individual). "
    "When in doubt, classify as 'org'.\n"
)


def build_megabundle_prompt(
    transcript: str,
    *,
    language: Optional[str] = "en",
    num_insights: int = DEFAULT_MEGA_BUNDLE_INSIGHTS,
    num_topics: int = DEFAULT_MEGA_BUNDLE_TOPICS,
    max_entities: int = DEFAULT_MEGA_BUNDLE_ENTITIES_MAX,
    max_transcript_chars: int = 25_000,
) -> Tuple[str, str]:
    """Return (system_prompt, user_prompt) for a mega-bundle request.

    Callers are responsible for provider-specific wrapping (e.g. Anthropic's
    system+messages split, OpenAI's chat.completions format). The prompt text
    itself is provider-neutral so the same research shape applies.

    Args:
        transcript: Raw episode transcript text. Will be truncated to
            ``max_transcript_chars`` to keep input-token cost bounded.
        language: Optional language hint, default "en".
        num_insights: Exact number of grounded insights to request
            (autoresearch sweet spot = 12).
        num_topics: Exact number of KG topic labels (autoresearch sweet spot = 10).
        max_entities: Upper bound on entities (not a hard minimum).
        max_transcript_chars: Input transcript cap, default 25K chars.

    Returns:
        (system_prompt, user_prompt) tuple of strings.
    """
    # Truncate input to keep token cost bounded. Providers that want the full
    # transcript can pass a larger ``max_transcript_chars`` or override
    # client-side.
    if len(transcript) > max_transcript_chars:
        transcript = transcript[:max_transcript_chars]

    system = (
        "You are a podcast content analyzer. Given a podcast episode transcript, "
        "produce a SINGLE JSON object with exactly the fields specified below. "
        "Output valid JSON only — no commentary, no code fences."
    )

    lang_hint = f" Language: {language}." if language else ""

    user = (
        f"{QUALITY_RULES}\n"
        "From the transcript below, extract the following fields into one JSON "
        "object:\n\n"
        '  "title": string — concise episode title (10-15 words).\n'
        '  "summary": string — 4-6 paragraph prose summary, covering main '
        "arguments, guests, and conclusions.\n"
        '  "bullets": array of 4-6 strings — key takeaways as standalone sentences.\n'
        f'  "insights": array of EXACTLY {num_insights} objects, each '
        '{"text": string, "insight_type": "claim"|"fact"|"opinion"}. '
        "See rule 2 — third-person paraphrased claims, not verbatim dialogue.\n"
        f'  "topics": array of EXACTLY {num_topics} strings — see rule 3 '
        "(2–3 word canonical noun phrases).\n"
        f'  "entities": array of up to {max_entities} objects, each '
        '{"name": string, "kind": "person"|"org"|"place", '
        '"role": "host"|"guest"|"mentioned"}. '
        "See rule 4 — default to 'org'. "
        'Use "host"/"guest" only when the transcript clearly identifies the '
        'person as such; otherwise use "mentioned".'
        f"{lang_hint}\n\n"
        "Transcript:\n"
        "---\n"
        f"{transcript}\n"
        "---\n\n"
        "Output ONLY the JSON object."
    )
    return system, user


def build_extraction_bundle_prompt(
    transcript: str,
    *,
    language: Optional[str] = "en",
    num_insights: int = DEFAULT_MEGA_BUNDLE_INSIGHTS,
    num_topics: int = DEFAULT_MEGA_BUNDLE_TOPICS,
    max_entities: int = DEFAULT_MEGA_BUNDLE_ENTITIES_MAX,
    max_transcript_chars: int = 25_000,
) -> Tuple[str, str]:
    """Build the extraction-only half of a 2-call pipeline (#643 extraction_bundled).

    The summary + bullets + title are produced by the provider's standalone
    ``summarize()`` call (first call). The second call uses this prompt to
    bundle insights + topics + entities. Suitable for OpenAI, Gemini, Mistral,
    Grok — providers where full mega-bundle compresses the summary too much.
    """
    if len(transcript) > max_transcript_chars:
        transcript = transcript[:max_transcript_chars]

    system = (
        "You are a podcast content analyzer. Given a podcast episode transcript, "
        "produce a SINGLE JSON object with exactly the fields specified below. "
        "Output valid JSON only — no commentary, no code fences."
    )

    lang_hint = f" Language: {language}." if language else ""

    user = (
        f"{QUALITY_RULES}\n"
        "From the transcript below, extract the following structured fields "
        "into one JSON object:\n\n"
        f'  "insights": array of EXACTLY {num_insights} objects, each '
        '{"text": string, "insight_type": "claim"|"fact"|"opinion"}. '
        "See rule 2 — third-person paraphrased claims.\n"
        f'  "topics": array of EXACTLY {num_topics} strings — see rule 3 '
        "(2–3 word canonical noun phrases).\n"
        f'  "entities": array of up to {max_entities} objects, each '
        '{"name": string, "kind": "person"|"org"|"place", '
        '"role": "host"|"guest"|"mentioned"}. '
        "See rule 4 — default to 'org'. "
        'Use "host"/"guest" only when the transcript identifies them as such.'
        f"{lang_hint}\n\n"
        "Transcript:\n"
        "---\n"
        f"{transcript}\n"
        "---\n\n"
        "Output ONLY the JSON object."
    )
    return system, user
