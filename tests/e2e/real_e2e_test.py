"""Real end-to-end test — runs the full voice assistant pipeline with real models.

Feeds pre-recorded WAV files through: VAD → ASR → LLM → TTS → Playlist.
Only mic input and speaker output are faked (WAV files in, audio captured out).
All other components are REAL: Silero VAD, Parakeet ASR, llama.cpp LLM, OmniVoice TTS.

Logs everything with timestamps. Produces analysis report at the end.

Usage:
    cd /home/winers/voice-assistant
    .venv/bin/python tests/e2e/real_e2e_test.py 2>&1 | tee /tmp/va_e2e.log
"""
import asyncio
import json
import logging
import os
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "samples")


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def get_rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return 0


def get_gpu_mb() -> float:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024 / 1024
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# WAV loader
# ---------------------------------------------------------------------------

def load_wav_16k(path: str) -> np.ndarray:
    """Load WAV file and return float32 array at 16kHz mono."""
    import wave
    with wave.open(path, "rb") as wf:
        assert wf.getsampwidth() == 2, f"Expected 16-bit WAV, got {wf.getsampwidth()*8}-bit"
        assert wf.getnchannels() == 1, f"Expected mono, got {wf.getnchannels()} channels"
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    # Resample to 16kHz if needed
    if sr != 16000:
        ratio = 16000 / sr
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    return audio


# ---------------------------------------------------------------------------
# Fake audio input — feeds WAV data chunk by chunk
# ---------------------------------------------------------------------------

class FakeAudioInput:
    """Replaces AudioInput — feeds pre-loaded audio in 480-sample chunks.

    Supports queueing multiple utterances with silence gaps between them.
    The orchestrator calls read_chunk() in its VAD loop.
    """
    CHUNK_SIZE = 480   # 30ms at 16kHz
    SILENCE_GAP_S = 1.0  # silence between utterances

    def __init__(self):
        self._queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self._current_audio: np.ndarray | None = None
        self._position: int = 0
        self._feeding = False
        self.ring = _FakeCaptureRing()
        self._pending_utterances: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._feeder_task: asyncio.Task | None = None

    async def start(self):
        self._feeder_task = asyncio.create_task(self._feeder_loop())

    async def stop(self):
        if self._feeder_task:
            self._feeder_task.cancel()
            try:
                await self._feeder_task
            except asyncio.CancelledError:
                pass

    def queue_utterance(self, audio: np.ndarray):
        """Queue an utterance to be fed into the pipeline."""
        self._pending_utterances.put_nowait(audio)

    async def _feeder_loop(self):
        """Background task that feeds utterances into the chunk queue."""
        try:
            while True:
                # Wait for next utterance
                audio = await self._pending_utterances.get()

                # Feed the audio as chunks
                pos = 0
                while pos < len(audio):
                    chunk = audio[pos:pos + self.CHUNK_SIZE]
                    if len(chunk) < self.CHUNK_SIZE:
                        chunk = np.pad(chunk, (0, self.CHUNK_SIZE - len(chunk)))
                    self._queue.put_nowait(chunk)
                    self.ring.write(chunk)
                    pos += self.CHUNK_SIZE
                    # Pace at roughly real-time (30ms per chunk)
                    await asyncio.sleep(0.005)  # faster than real-time but gives VAD time

                # Silence gap after utterance
                silence_chunks = int(self.SILENCE_GAP_S * 16000 / self.CHUNK_SIZE)
                silence = np.zeros(self.CHUNK_SIZE, dtype=np.float32)
                for _ in range(silence_chunks):
                    self._queue.put_nowait(silence)
                    self.ring.write(silence)
                    await asyncio.sleep(0.005)

        except asyncio.CancelledError:
            raise

    async def read_chunk(self) -> np.ndarray:
        """Called by orchestrator's VAD loop."""
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            # No audio queued — return silence
            await asyncio.sleep(0.03)  # simulate real-time 30ms
            silence = np.zeros(self.CHUNK_SIZE, dtype=np.float32)
            self.ring.write(silence)
            return silence


class _FakeCaptureRing:
    """Minimal CaptureRing replacement for test — stores audio for ASR drain."""
    def __init__(self, capacity: int = 16000 * 30):
        self._buffer = np.zeros(capacity, dtype=np.float32)
        self._write_pos = 0
        self._capacity = capacity

    def write(self, chunk: np.ndarray):
        n = len(chunk)
        end = self._write_pos + n
        if end <= self._capacity:
            self._buffer[self._write_pos:end] = chunk
        else:
            first = self._capacity - self._write_pos
            self._buffer[self._write_pos:] = chunk[:first]
            self._buffer[:n - first] = chunk[first:]
        self._write_pos = end % self._capacity

    def read_last(self, num_samples: int) -> np.ndarray:
        if num_samples > self._capacity:
            num_samples = self._capacity
        start = (self._write_pos - num_samples) % self._capacity
        if start < self._write_pos:
            return self._buffer[start:self._write_pos].copy()
        else:
            return np.concatenate([
                self._buffer[start:],
                self._buffer[:self._write_pos],
            ])


# ---------------------------------------------------------------------------
# Fake playback — captures TTS audio instead of playing it
# ---------------------------------------------------------------------------

class FakePlayback:
    """Replaces PlaybackManager — reads from playlist and captures audio to disk."""
    def __init__(self, playlist):
        self._playlist = playlist
        self.is_active = False
        self._fading = False
        self.captured_audio: list[np.ndarray] = []

    async def start(self):
        self.is_active = True

    async def stop(self):
        self.is_active = False

    def fade_out(self, duration_ms: int = 15):
        """Simulate fade-out for barge-in."""
        self._fading = True
        self.is_active = False


# ---------------------------------------------------------------------------
# Test result tracking
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    turn_idx: int
    scenario: str
    sample_id: str
    expected_lang: str
    expected_text: str  # what we said (ground truth)
    asr_text: str = ""  # what ASR recognized
    detected_lang: str = ""
    assistant_response: str = ""
    asr_time_s: float = 0
    llm_ttft_s: float = 0
    llm_total_s: float = 0
    tts_total_s: float = 0
    turn_total_s: float = 0
    rss_mb: float = 0
    gpu_mb: float = 0
    state_transitions: list = field(default_factory=list)
    errors: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

SCENARIOS = [
    # --- Scenario 1: Basic conversation (en) ---
    {
        "name": "basic_english_conversation",
        "description": "5-turn English conversation — tests ASR accuracy and LLM coherence",
        "turns": [
            {"sample": "en_01_greeting",  "lang": "en", "wait_done": True},
            {"sample": "en_02_question",  "lang": "en", "wait_done": True},
            {"sample": "en_04_followup",  "lang": "en", "wait_done": True},
            {"sample": "en_05_topic_switch", "lang": "en", "wait_done": True},
            {"sample": "en_09_short_sure", "lang": "en", "wait_done": True},
        ],
    },
    # --- Scenario 2: Language switching ---
    {
        "name": "language_switching",
        "description": "Switch between en → ru → hr → en — tests language detection and TTS switching",
        "turns": [
            {"sample": "en_01_greeting",  "lang": "en", "wait_done": True},
            {"sample": "ru_01_greeting",  "lang": "ru", "wait_done": True},
            {"sample": "hr_01_greeting",  "lang": "hr", "wait_done": True},
            {"sample": "en_02_question",  "lang": "en", "wait_done": True},
        ],
    },
    # --- Scenario 3: Rapid short inputs ---
    {
        "name": "rapid_short_inputs",
        "description": "Quick short utterances — tests VAD sensitivity with minimal speech",
        "turns": [
            {"sample": "en_06_short_yes",  "lang": "en", "wait_done": True},
            {"sample": "en_07_short_ok",   "lang": "en", "wait_done": True},
            {"sample": "en_08_short_what", "lang": "en", "wait_done": True},
        ],
    },
    # --- Scenario 4: Russian conversation ---
    {
        "name": "russian_conversation",
        "description": "Multi-turn Russian — tests non-English ASR and LLM response language",
        "turns": [
            {"sample": "ru_01_greeting",  "lang": "ru", "wait_done": True},
            {"sample": "ru_02_question",  "lang": "ru", "wait_done": True},
            {"sample": "ru_03_complex",   "lang": "ru", "wait_done": True},
        ],
    },
    # --- Scenario 5: Barge-in test ---
    {
        "name": "barge_in",
        "description": "User interrupts during assistant speech — tests barge-in pipeline",
        "turns": [
            {"sample": "en_03_complex",       "lang": "en", "wait_done": True},  # long question → long answer
            {"sample": "barge_01_interrupt",   "lang": "en", "wait_done": False, "barge_in": True,
             "inject_delay_s": 3.0},  # inject after 3s of assistant speaking
        ],
    },
    # --- Scenario 6: Long complex utterances ---
    {
        "name": "complex_utterances",
        "description": "Long complex inputs in multiple languages — stress ASR and LLM context",
        "turns": [
            {"sample": "en_03_complex",  "lang": "en", "wait_done": True},
            {"sample": "ru_05_long",     "lang": "ru", "wait_done": True},
            {"sample": "hr_03_complex",  "lang": "hr", "wait_done": True},
        ],
    },
]


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

async def run_e2e_test():
    # --- Logging setup ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    logger = logging.getLogger("e2e_test")

    logger.info("=" * 80)
    logger.info("REAL E2E TEST START")
    logger.info("=" * 80)

    t0 = time.monotonic()
    all_results: list[TurnResult] = []

    def mem_log(label):
        rss = get_rss_mb()
        gpu = get_gpu_mb()
        elapsed = time.monotonic() - t0
        logger.info("[%.1fs] MEM: RSS=%.0f MB  GPU=%.0f MB  (%s)", elapsed, rss, gpu, label)
        return rss, gpu

    mem_log("before_imports")

    # --- Load SOUL and settings ---
    from voice_assistant.core.soul_loader import load_soul
    from voice_assistant.core.settings_loader import load_settings
    soul = load_soul("soul/default.soul.yaml")
    settings = load_settings("config/settings.yaml")
    mem_log("after_config")

    # --- Initialize components via ProcessManager ---
    from voice_assistant.core.vram_guard import VRAMGuard
    from voice_assistant.core.ipc import StdoutEmitter
    from voice_assistant.process.manager import ProcessManager

    vram = VRAMGuard()
    devnull = open(os.devnull, "w")
    emitter = StdoutEmitter(output_stream=devnull)
    await emitter.start()

    pm = ProcessManager(soul, vram, emitter, settings=settings)
    logger.info("Starting all real models...")
    mode = await pm.startup()
    logger.info("Pipeline mode: %s", mode.value)
    mem_log("after_model_startup")

    if not (pm._llm_process and pm._tts and pm._asr and pm._vad):
        logger.error("NOT ALL COMPONENTS LOADED — aborting")
        await pm.shutdown()
        return

    # --- Build orchestrator with fake audio I/O ---
    from voice_assistant.core.orchestrator import Orchestrator
    from voice_assistant.core.audio_output import Playlist, FillerCache
    from voice_assistant.adapters.llamacpp_llm import LlamaCppLLM
    from voice_assistant.core.state_machine import PipelineMode, PipelineState

    fake_audio = FakeAudioInput()
    playlist = Playlist()
    fake_playback = FakePlayback(playlist)
    filler_cache = FillerCache()
    llm = LlamaCppLLM(f"http://127.0.0.1:{pm._llm_process.port}")

    supported = [vl.id for vl in settings.voice_languages] or ["en"]
    persona = soul.persona_card()
    system_prompt = (
        f"{persona}\n\n"
        "You are a voice assistant. Your responses will be spoken aloud via text-to-speech. "
        "Always respond in plain conversational text. Never use markdown, asterisks, "
        "bullet points, numbered lists, headers, or any formatting symbols. "
        "Keep responses concise and natural for spoken conversation."
    )
    orch = Orchestrator(
        audio_input=fake_audio,
        vad=pm._vad,
        asr=pm._asr,
        llm=llm,
        tts=pm._tts,
        playlist=playlist,
        playback=fake_playback,
        filler_cache=filler_cache,
        system_prompt=system_prompt,
        default_language=settings.default_language,
        supported_languages=supported,
    )

    # Track state transitions
    state_log = []
    original_on_change = orch._on_state_change

    def tracking_on_change(old, new):
        state_log.append((time.monotonic() - t0, old.value, new.value))
        original_on_change(old, new)

    orch._state._on_change = tracking_on_change

    # Start orchestrator (this starts the VAD loop)
    await fake_audio.start()
    await fake_playback.start()
    orch._state.set_mode(PipelineMode.FULL)
    if orch._state.state != PipelineState.IDLE:
        orch._state.transition(PipelineState.IDLE)
    orch._vad_task = asyncio.create_task(orch._vad_loop())
    logger.info("Orchestrator started with fake audio I/O")
    mem_log("after_orchestrator_start")

    # --- Load all WAV samples ---
    logger.info("Loading WAV samples from %s", SAMPLE_DIR)
    wav_cache = {}
    for fname in sorted(os.listdir(SAMPLE_DIR)):
        if fname.endswith(".wav"):
            sample_id = fname.replace(".wav", "")
            audio = load_wav_16k(os.path.join(SAMPLE_DIR, fname))
            wav_cache[sample_id] = audio
            logger.info("  Loaded %s: %.1fs (%d samples)", sample_id, len(audio) / 16000, len(audio))
    logger.info("Loaded %d samples", len(wav_cache))

    # --- Run scenarios ---
    global_turn = 0
    for scenario in SCENARIOS:
        logger.info("")
        logger.info("=" * 80)
        logger.info("SCENARIO: %s", scenario["name"])
        logger.info("  %s", scenario["description"])
        logger.info("=" * 80)

        # Clear dialogue for each scenario
        orch._dialogue.clear()
        orch._pending_speech = None
        orch._generation_cancelled = False
        state_log.clear()

        for turn_def in scenario["turns"]:
            global_turn += 1
            sample_id = turn_def["sample"]
            expected_lang = turn_def["lang"]
            audio = wav_cache[sample_id]
            is_barge_in = turn_def.get("barge_in", False)

            result = TurnResult(
                turn_idx=global_turn,
                scenario=scenario["name"],
                sample_id=sample_id,
                expected_lang=expected_lang,
                expected_text=_get_sample_text(sample_id),
            )

            logger.info("")
            logger.info("─── TURN %d [%s] %s ─── (%.1fs audio) '%s'",
                        global_turn, expected_lang, sample_id,
                        len(audio) / 16000, result.expected_text[:60])

            turn_start = time.monotonic()
            state_before = len(state_log)

            if is_barge_in:
                # Wait for specified delay, then inject barge-in audio
                delay = turn_def.get("inject_delay_s", 2.0)
                logger.info("  BARGE-IN: waiting %.1fs before injecting interrupt audio", delay)
                await asyncio.sleep(delay)

            # Feed audio into the fake audio input
            fake_audio.queue_utterance(audio)

            # Wait for the turn to complete
            if turn_def.get("wait_done", True):
                # Wait for state to cycle: should go IDLE → LISTENING → PROCESSING → SPEAKING → IDLE
                # Timeout: 120s max per turn
                try:
                    await _wait_for_idle(orch, timeout=120.0)
                except asyncio.TimeoutError:
                    logger.error("  TURN %d TIMED OUT after 120s (state=%s)",
                                 global_turn, orch.state.value)
                    result.errors.append("TIMEOUT")
            else:
                # For barge-in: just wait a bit for the barge-in to process
                await asyncio.sleep(5.0)

            turn_elapsed = time.monotonic() - turn_start
            result.turn_total_s = turn_elapsed
            result.rss_mb, result.gpu_mb = mem_log(f"turn_{global_turn}")
            result.state_transitions = list(state_log[state_before:])

            # Extract ASR result and assistant response from dialogue
            for turn in orch._dialogue:
                if turn.role == "user" and turn.source == "voice":
                    result.asr_text = turn.content
                    result.detected_lang = orch._current_language

            if orch._dialogue and orch._dialogue[-1].role == "assistant":
                result.assistant_response = orch._dialogue[-1].content
                if orch._dialogue[-1].interrupted:
                    result.errors.append("INTERRUPTED")

            logger.info("  TURN %d DONE: %.1fs | ASR='%s' | lang=%s | response=%d chars",
                        global_turn, turn_elapsed,
                        result.asr_text[:60], result.detected_lang,
                        len(result.assistant_response))

            all_results.append(result)

            # Brief pause between turns
            await asyncio.sleep(1.0)

    mem_log("after_all_scenarios")

    # --- Shutdown ---
    logger.info("Shutting down...")
    if orch._vad_task:
        orch._vad_task.cancel()
        try:
            await orch._vad_task
        except asyncio.CancelledError:
            pass
    if orch._processing_task:
        orch._processing_task.cancel()
        try:
            await orch._processing_task
        except asyncio.CancelledError:
            pass
    await fake_audio.stop()
    await fake_playback.stop()
    await llm.shutdown()
    await pm.shutdown()
    vram.shutdown()
    await emitter.stop()
    devnull.close()

    total_time = time.monotonic() - t0
    mem_log("after_shutdown")

    # --- Analysis report ---
    _print_report(logger, all_results, total_time)


async def _wait_for_idle(orch, timeout: float = 120.0):
    """Wait until orchestrator returns to IDLE after processing a turn."""
    from voice_assistant.core.state_machine import PipelineState
    deadline = time.monotonic() + timeout
    saw_processing = False

    while time.monotonic() < deadline:
        state = orch.state
        if state == PipelineState.PROCESSING or state == PipelineState.SPEAKING:
            saw_processing = True
        elif state == PipelineState.IDLE and saw_processing:
            # Give a moment for any pending speech processing
            await asyncio.sleep(0.5)
            if orch.state == PipelineState.IDLE:
                return
        await asyncio.sleep(0.1)

    raise asyncio.TimeoutError()


def _get_sample_text(sample_id: str) -> str:
    """Get the original text for a sample (for ASR accuracy comparison)."""
    for s in _ALL_SAMPLES:
        if s["id"] == sample_id:
            return s["text"]
    return ""


# Import sample definitions from generator
_ALL_SAMPLES = [
    {"id": "en_01_greeting",     "text": "Hello, how are you today?"},
    {"id": "en_02_question",     "text": "Tell me something interesting about electric cars."},
    {"id": "en_03_complex",      "text": "I've been thinking about consciousness lately. Do you think machines can truly understand, or is it just pattern matching?"},
    {"id": "en_04_followup",     "text": "That's fascinating. Can you elaborate on the philosophical implications?"},
    {"id": "en_05_topic_switch", "text": "Let's talk about something different. What do you know about deep sea creatures?"},
    {"id": "en_06_short_yes",    "text": "Yes."},
    {"id": "en_07_short_ok",     "text": "OK."},
    {"id": "en_08_short_what",   "text": "What?"},
    {"id": "en_09_short_sure",   "text": "Sure, that sounds great!"},
    {"id": "en_10_stop",         "text": "Stop. I want to ask you something else."},
    {"id": "ru_01_greeting",     "text": "Привет, как у тебя дела сегодня?"},
    {"id": "ru_02_question",     "text": "Расскажи мне что-нибудь интересное про космос."},
    {"id": "ru_03_complex",      "text": "Что ты думаешь о будущем технологий и искусственного интеллекта?"},
    {"id": "ru_04_short",        "text": "Да, конечно."},
    {"id": "ru_05_long",         "text": "А теперь давай поговорим о классической русской литературе. Кто твой любимый автор и какое произведение нравится больше всего?"},
    {"id": "hr_01_greeting",     "text": "Bok, kako si danas?"},
    {"id": "hr_02_question",     "text": "Reci mi nešto zanimljivo o Hrvatskoj."},
    {"id": "hr_03_complex",      "text": "Što misliš o umjetnoj inteligenciji i njenom utjecaju na društvo?"},
    {"id": "barge_01_interrupt",  "text": "Wait, wait, stop. I have a question."},
    {"id": "barge_02_redirect",   "text": "Actually, never mind. Tell me about the weather instead."},
]


def _word_error_rate(reference: str, hypothesis: str) -> float:
    """Simple word error rate (Levenshtein on words)."""
    ref = reference.lower().strip().split()
    hyp = hypothesis.lower().strip().split()
    if not ref:
        return 0.0 if not hyp else 1.0

    # Dynamic programming
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(ref)][len(hyp)] / len(ref)


def _print_report(logger, results: list[TurnResult], total_time: float):
    """Print comprehensive analysis report."""
    logger.info("")
    logger.info("=" * 100)
    logger.info("E2E TEST REPORT")
    logger.info("=" * 100)
    logger.info("Total time: %.1fs | Turns: %d", total_time, len(results))
    logger.info("")

    # --- Per-turn results ---
    logger.info("%-4s %-25s %-4s %-4s %6s %6s %6s %-30s %-30s",
                "#", "Sample", "Lang", "Det", "Turn", "RSS", "GPU",
                "Expected (first 30)", "ASR (first 30)")
    logger.info("-" * 150)

    total_wer = 0
    wer_count = 0
    lang_correct = 0
    lang_total = 0
    errors = []

    for r in results:
        wer = _word_error_rate(r.expected_text, r.asr_text)
        lang_match = "✓" if r.detected_lang == r.expected_lang else f"✗→{r.detected_lang}"

        if r.asr_text:
            total_wer += wer
            wer_count += 1
        if r.detected_lang:
            lang_total += 1
            if r.detected_lang == r.expected_lang:
                lang_correct += 1

        err_str = " ".join(r.errors) if r.errors else ""

        logger.info(
            "%-4d %-25s %-4s %-4s %5.1fs %5.0fM %5.0fM %-30s %-30s  WER=%.0f%% %s",
            r.turn_idx, r.sample_id, r.expected_lang, lang_match,
            r.turn_total_s, r.rss_mb, r.gpu_mb,
            r.expected_text[:30], r.asr_text[:30],
            wer * 100, err_str,
        )

        if r.errors:
            errors.extend(r.errors)

    # --- Summary stats ---
    logger.info("")
    logger.info("=" * 100)
    logger.info("SUMMARY")
    logger.info("=" * 100)

    avg_wer = (total_wer / wer_count * 100) if wer_count else 0
    lang_acc = (lang_correct / lang_total * 100) if lang_total else 0
    avg_turn = sum(r.turn_total_s for r in results) / len(results) if results else 0
    peak_rss = max(r.rss_mb for r in results) if results else 0
    peak_gpu = max(r.gpu_mb for r in results) if results else 0
    initial_rss = results[0].rss_mb if results else 0
    rss_growth = peak_rss - initial_rss

    logger.info("  ASR Word Error Rate (avg): %.1f%%", avg_wer)
    logger.info("  Language Detection:        %d/%d correct (%.0f%%)", lang_correct, lang_total, lang_acc)
    logger.info("  Avg Turn Time:             %.1fs", avg_turn)
    logger.info("  Peak RSS:                  %.0f MB", peak_rss)
    logger.info("  Peak GPU:                  %.0f MB", peak_gpu)
    logger.info("  RSS Growth:                %.0f MB (%.0f → %.0f)", rss_growth, initial_rss, peak_rss)
    logger.info("  Errors:                    %d (%s)", len(errors), ", ".join(errors) if errors else "none")
    logger.info("")

    # --- Conversation quality (show dialogue) ---
    logger.info("=" * 100)
    logger.info("CONVERSATION LOG (ASR text → LLM response)")
    logger.info("=" * 100)
    for r in results:
        logger.info("")
        logger.info("[%s] Turn %d — %s", r.scenario, r.turn_idx, r.sample_id)
        logger.info("  EXPECTED: %s", r.expected_text)
        logger.info("  ASR:      %s", r.asr_text if r.asr_text else "(empty)")
        logger.info("  LLM:      %s", r.assistant_response[:200] if r.assistant_response else "(no response)")
        if r.errors:
            logger.info("  ERRORS:   %s", ", ".join(r.errors))

    # --- Memory timeline ---
    logger.info("")
    logger.info("MEMORY TIMELINE:")
    for r in results:
        bar = "█" * int(r.rss_mb / 100)
        logger.info("  Turn %-3d %6.0f MB RSS  %6.0f MB GPU  %s",
                     r.turn_idx, r.rss_mb, r.gpu_mb, bar)

    logger.info("")
    logger.info("=" * 100)
    logger.info("E2E TEST COMPLETE — %.1fs total", total_time)
    logger.info("=" * 100)


if __name__ == "__main__":
    try:
        asyncio.run(run_e2e_test())
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
