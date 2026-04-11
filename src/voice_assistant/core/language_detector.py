"""Multi-strategy language detection for supported language set.

Parakeet TDT v3 transcribes 25+ languages but doesn't expose the detected
language.  This module identifies the language from the transcribed text
using a chain of strategies, from most reliable to least:

  0. Single-word exact match    → definitive language markers
  1. Cyrillic script detection  → ru  (100% reliable)
  2. South Slavic diacritics    → hr  (č/ć/š/ž/đ are definitive)
  3. langdetect filtered by supported languages + aliases
  4. Common-word hints          → en vs hr fallback (≥1 match for short, ≥2 for long)

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

# Words that unambiguously identify a language even as a single word.
# Used for very short utterances ("Yeah", "Да", "Da").
_DEFINITIVE_WORDS: dict[str, str] = {
    # English
    "yeah": "en", "yes": "en", "no": "en", "okay": "en", "ok": "en",
    "sure": "en", "right": "en", "well": "en", "thanks": "en",
    "thank": "en", "please": "en", "hello": "en", "hi": "en",
    "hey": "en", "bye": "en", "sorry": "en", "really": "en",
    "exactly": "en", "absolutely": "en", "interesting": "en",
    "nice": "en", "cool": "en", "great": "en", "good": "en",
    "actually": "en", "maybe": "en", "probably": "en",
    "what": "en", "why": "en", "how": "en", "where": "en", "when": "en",
    "true": "en", "false": "en", "agree": "en",
    # Russian
    "да": "ru", "нет": "ru", "конечно": "ru", "хорошо": "ru",
    "ладно": "ru", "привет": "ru", "пока": "ru", "спасибо": "ru",
    "пожалуйста": "ru", "здравствуйте": "ru", "извините": "ru",
    "правильно": "ru", "точно": "ru", "именно": "ru", "понятно": "ru",
    "интересно": "ru", "отлично": "ru", "замечательно": "ru",
    # Croatian
    "da": "hr", "ne": "hr", "hvala": "hr", "molim": "hr",
    "bok": "hr", "dobar": "hr", "dobro": "hr", "naravno": "hr",
    "upravo": "hr", "tocno": "hr", "zanimljivo": "hr",
}

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
        "yeah", "yes", "okay", "sure", "right", "well", "really",
        "actually", "maybe", "probably", "exactly", "interesting",
        "nice", "cool", "great", "good", "it", "so",
    }),
    "hr": frozenset({
        "je", "su", "sam", "si", "bio", "bila", "biti", "imam", "idem",
        "kako", "gdje", "tko", "kad", "danas", "dobar", "dan", "hvala",
        "molim", "ovo", "smo", "ste", "ali", "ili", "oko", "kod",
        "nakon", "prije", "tijekom", "jako", "dobro", "ima", "nema",
        "nisam", "nije", "nisu", "sve", "ona", "oni", "one", "taj",
        "onda", "zato", "malo", "puno", "samo", "vec",
    }),
    "ru": frozenset({
        "это", "как", "что", "где", "когда", "почему", "зачем",
        "очень", "тоже", "ещё", "уже", "потому", "сейчас", "здесь",
        "там", "тут", "мне", "мой", "наш", "ваш", "его", "она",
        "они", "для", "при", "без", "все", "вот", "так",
    }),
}

# Minimum text length for full detection (strategies 3+4).
# Strategies 0-2 (definitive word, Cyrillic, diacritics) work at any length.
MIN_CHARS_FULL = 12


def detect_language(
    text: str,
    supported_languages: set[str] | list[str],
) -> str | None:
    """Detect the language of *text* within *supported_languages*.

    Returns the ISO 639-1 code if confident, or ``None`` when uncertain
    (caller should keep the current language unchanged).
    """
    stripped = text.strip()
    if not stripped:
        logger.info("LANG DETECT: empty text → None")
        return None

    supported = set(supported_languages)
    alpha = [c for c in stripped if c.isalpha()]
    if not alpha:
        logger.info("LANG DETECT: no alpha chars in '%s' → None", stripped[:40])
        return None

    words_lower = set(stripped.lower().split())
    # Strip trailing punctuation from words for matching
    words_clean = set()
    for w in words_lower:
        clean = w.rstrip(".,!?;:\"'")
        if clean:
            words_clean.add(clean)

    # --- Strategy 1: Cyrillic script → Russian --------------------------
    # Checked first — Cyrillic is 100% reliable and trumps everything.
    cyrillic = sum(1 for c in alpha if "\u0400" <= c <= "\u04ff")
    if cyrillic / len(alpha) > 0.3:
        for lang in ("ru", "uk", "bg"):
            if lang in supported:
                logger.info("LANG DETECT: Cyrillic %.0f%% → %s (text='%s')",
                            100 * cyrillic / len(alpha), lang, stripped[:60])
                return lang
        logger.info("LANG DETECT: Cyrillic detected but no supported Cyrillic lang")
        return None

    # --- Strategy 2: South Slavic diacritics → Croatian -----------------
    if any(c in _SOUTH_SLAVIC_DIACRITICS for c in stripped):
        for lang in ("hr", "bs", "sr"):
            if lang in supported:
                logger.info("LANG DETECT: diacritics → %s (text='%s')",
                            lang, stripped[:60])
                return lang

    # --- Strategy 0: Definitive single-word match (works at any length) ---
    # For very short text (1-2 words), a single definitive word is enough.
    # Runs after Cyrillic/diacritics so script-level signals win for mixed text.
    for w in words_clean:
        lang = _DEFINITIVE_WORDS.get(w)
        if lang and lang in supported:
            logger.info("LANG DETECT: definitive word '%s' → %s (text='%s')",
                        w, lang, stripped[:60])
            return lang

    # --- Short text: strategies 0-2 exhausted, stop here ----------------
    if len(stripped) < MIN_CHARS_FULL:
        logger.info("LANG DETECT: short text (%d chars) no definitive match → None (text='%s')",
                    len(stripped), stripped[:60])
        return None

    # --- Strategy 3: langdetect (filtered by supported + aliases) -------
    try:
        from langdetect import detect_langs, DetectorFactory
        DetectorFactory.seed = 0
        langs = detect_langs(stripped)
        for det in langs:
            code = LANG_ALIASES.get(det.lang, det.lang)
            if code in supported and det.prob >= 0.4:
                logger.info("LANG DETECT: langdetect → %s (p=%.2f) (text='%s')",
                            code, det.prob, stripped[:60])
                return code
        logger.info("LANG DETECT: langdetect no match in supported (results=%s)", langs[:3])
    except Exception as e:
        logger.warning("LANG DETECT: langdetect error: %s", e)

    # --- Strategy 4: word-hint fallback ---------------------------------
    scores: dict[str, int] = {}
    for lang, hints in _WORD_HINTS.items():
        if lang in supported:
            score = len(words_clean & hints)
            if score >= 2:
                scores[lang] = score
    if scores:
        best = max(scores, key=scores.get)
        logger.info("LANG DETECT: word hints → %s (score=%d) (text='%s')",
                    best, scores[best], stripped[:60])
        return best

    logger.info("LANG DETECT: no match → None (text='%s')", stripped[:60])
    return None
