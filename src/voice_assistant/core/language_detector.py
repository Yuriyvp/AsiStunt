"""Multi-strategy language detection for supported language set.

Parakeet TDT v3 transcribes 25+ languages but doesn't expose the detected
language.  This module identifies the language from the transcribed text
using a chain of strategies, from most reliable to least:

  1. Cyrillic script detection  → ru  (100% reliable)
  2. South Slavic diacritics    → hr  (č/ć/š/ž/đ are definitive)
  3. langdetect filtered by supported languages + aliases
  4. Common-word hints          → en vs hr fallback

Each layer is cheap (<1ms total) and requires no extra models.
"""
import logging

logger = logging.getLogger(__name__)

# langdetect often confuses closely related languages.
# Map common misdetections to the correct supported code.
LANG_ALIASES: dict[str, str] = {
    "sr": "hr",   # Serbian  → Croatian
    "bs": "hr",   # Bosnian  → Croatian
}

# Characters unique to South Slavic Latin orthography.
_SOUTH_SLAVIC_DIACRITICS = frozenset("čćšžđČĆŠŽĐ")

# Common words that almost never appear in the other languages.
# Used as a last-resort fallback when langdetect fails entirely
# (e.g. returns "Somali" for "Hey, hi, how are you today?").
_WORD_HINTS: dict[str, frozenset[str]] = {
    "en": frozenset({
        "the", "is", "are", "was", "were", "have", "has", "been", "will",
        "can", "could", "would", "should", "what", "how", "who", "where",
        "when", "why", "this", "that", "these", "those", "hello", "hi",
        "today", "you", "your", "my", "with", "for", "but", "not",
        "they", "them", "about", "just", "from", "here", "there",
        "going", "doing", "know", "think", "want", "need", "very",
    }),
    "hr": frozenset({
        "je", "su", "sam", "si", "bio", "bila", "biti", "imam", "idem",
        "kako", "gdje", "tko", "kad", "danas", "dobar", "dan", "hvala",
        "molim", "ovo", "smo", "ste", "ali", "ili", "oko", "kod",
        "nakon", "prije", "tijekom", "jako", "dobro", "ima", "nema",
        "nisam", "nije", "nisu", "sve", "ona", "oni", "one", "taj",
        "onda", "zato", "malo", "puno", "samo", "vec",
    }),
}

# Minimum text length for detection (shorter text → keep current language).
MIN_CHARS = 12


def detect_language(
    text: str,
    supported_languages: set[str] | list[str],
) -> str | None:
    """Detect the language of *text* within *supported_languages*.

    Returns the ISO 639-1 code if confident, or ``None`` when uncertain
    (caller should keep the current language unchanged).
    """
    stripped = text.strip()
    if len(stripped) < MIN_CHARS:
        return None

    supported = set(supported_languages)
    alpha = [c for c in stripped if c.isalpha()]
    if not alpha:
        return None

    # --- Strategy 1: Cyrillic script → Russian --------------------------
    cyrillic = sum(1 for c in alpha if "\u0400" <= c <= "\u04ff")
    if cyrillic / len(alpha) > 0.3:
        for lang in ("ru", "uk", "bg"):
            if lang in supported:
                logger.debug("lang detect: Cyrillic script → %s", lang)
                return lang
        return None

    # --- Strategy 2: South Slavic diacritics → Croatian -----------------
    if any(c in _SOUTH_SLAVIC_DIACRITICS for c in stripped):
        for lang in ("hr", "bs", "sr"):
            if lang in supported:
                logger.debug("lang detect: diacritics → %s", lang)
                return lang

    # --- Strategy 3: langdetect (filtered by supported + aliases) -------
    try:
        from langdetect import detect_langs, DetectorFactory
        DetectorFactory.seed = 0
        langs = detect_langs(stripped)
        for det in langs:
            code = LANG_ALIASES.get(det.lang, det.lang)
            if code in supported and det.prob >= 0.4:
                logger.debug("lang detect: langdetect → %s (p=%.2f)", code, det.prob)
                return code
    except Exception:
        pass

    # --- Strategy 4: word-hint fallback (need ≥2 matching words) --------
    words = set(stripped.lower().split())
    scores: dict[str, int] = {}
    for lang, hints in _WORD_HINTS.items():
        if lang in supported:
            score = len(words & hints)
            if score >= 2:
                scores[lang] = score
    if scores:
        best = max(scores, key=scores.get)
        logger.debug("lang detect: word hints → %s (score=%d)", best, scores[best])
        return best

    return None
