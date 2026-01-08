"""Prompts for summarization models.

This module contains all prompt templates and patterns used for
episode summarization.
"""

# Reduce phase prompts
REDUCE_PROMPT = (
    "You are summarizing a full podcast episode using multiple partial summaries as input.\n"
    "Your task is to produce a clear, accurate, and cohesive final summary.\n"
    "\n"
    "Follow these principles:\n"
    "\n"
    "CONTENT SELECTION\n"
    "- Focus on the most important ideas, insights, arguments, explanations, "
    "turning points, and decisions.\n"
    "- Prioritize what the speakers were trying to explain, what changed over "
    "time, what challenges existed, what solutions were discussed, and what "
    "strategies or lessons emerged.\n"
    "- Emphasize cause-and-effect, reasoning, motivations, and big takeaways.\n"
    "- Include only information that contributes to understanding the episode's key themes.\n"
    "\n"
    "CONTENT FILTERING\n"
    "- Ignore sponsorships, promotions, discount codes, calls to action, and "
    "housekeeping notes.\n"
    "- Exclude generic intros/outros, chit-chat, fillers, and unrelated "
    "anecdotes unless essential to a key idea.\n"
    "- Do NOT include direct quotes, speaker names, banter, or dialogue-style "
    "formatting.\n"
    "- Do NOT attribute statements to specific individuals unless critical to meaning.\n"
    "\n"
    "STYLE\n"
    "- Write in a neutral, professional, narrative voice with well-structured "
    "paragraphs.\n"
    "- Use smooth transitions; avoid repetition, generic reflections, and "
    "unsupported conclusions.\n"
    "- Do NOT reference the structure of the source "
    "(e.g., 'in this section,' 'in these chunks,' 'later they said').\n"
    "\n"
    "OUTPUT LENGTH & QUALITY\n"
    "- Aim for a comprehensive, medium-length summary (2–4 paragraphs).\n"
    "- Ensure the summary reads as a complete, standalone explanation of the "
    "episode's core points."
)

REDUCE_PROMPT_SHORT = (
    "Summarize the following podcast episode into 2–4 paragraphs. "
    "Focus on the most important ideas, decisions, and lessons. "
    "Ignore sponsorships, promotions, intros, outros, and small talk."
)

# Default prompt for summarization
# Important: Explicitly instruct model to avoid hallucinations and stick to source content
DEFAULT_SUMMARY_PROMPT = (
    "Summarize the following podcast episode transcript accurately. "
    "Focus on the main topics, key insights, and important discussions. "
    "Only include information that is explicitly stated in the transcript. "
    "Do not add, infer, or invent any information not present in the original text:"
)

# Instruction leak protection patterns
# These patterns detect when model instructions leak into the summary output
INSTRUCTION_LEAK_PATTERNS = [
    r"your task is to",
    r"aim for a comprehensive",
    r"write in a neutral",
    r"do not reference the structure",
    r"output length & quality",
    r"follow these principles",
    r"content selection",
    r"content filtering",
    r"style",
    r"summarize the following",
]
