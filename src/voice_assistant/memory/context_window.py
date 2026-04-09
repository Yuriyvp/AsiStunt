"""Token budget management for context window.

Total: 8192 tokens
- Persona card: 800
- Mood/state: 50
- Rolling summary: 600
- System overhead: 100
- Dialogue tail: ~6050 (fills remaining)
- Generation headroom: 500
"""

TOTAL_CONTEXT = 8192
PERSONA_BUDGET = 800
MOOD_BUDGET = 50
SUMMARY_BUDGET = 600
OVERHEAD_BUDGET = 100
GENERATION_HEADROOM = 500
FIXED_BUDGET = PERSONA_BUDGET + MOOD_BUDGET + SUMMARY_BUDGET + OVERHEAD_BUDGET + GENERATION_HEADROOM
DIALOGUE_BUDGET = TOTAL_CONTEXT - FIXED_BUDGET  # ~6050

# Trigger rolling summary when dialogue tail exceeds this % of budget
SUMMARY_TRIGGER_RATIO = 0.80
SUMMARY_TRIGGER_TOKENS = int(DIALOGUE_BUDGET * SUMMARY_TRIGGER_RATIO)

# Min turns to keep in dialogue (never evict these)
MIN_KEEP_TURNS = 4

# Max tokens for a single user message before truncation
SINGLE_TURN_MAX = 2000
SINGLE_TURN_KEEP_HEAD = 500
SINGLE_TURN_KEEP_TAIL = 500
