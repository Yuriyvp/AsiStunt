"""Pipeline orchestrator — wires all components together.

Owns the state machine, coordinates: AudioInput → VAD → ASR → LLM → Chunker → TTS → Playlist.
Handles barge-in (150ms gate), text input, filler playback.

This is a single asyncio task — no threads except sounddevice callbacks and playback.
"""
import asyncio
import logging
import time
from dataclasses import dataclass

import numpy as np

from voice_assistant.core.state_machine import StateMachine, PipelineState, PipelineMode
from voice_assistant.core.audio_input import AudioInput
from voice_assistant.core.audio_output import Playlist, PlaybackManager, AudioChunk, FillerCache
from voice_assistant.core.sentence_chunker import SentenceChunker
from voice_assistant.core.language_detector import detect_language
from voice_assistant.core.mood_signal_parser import MoodSignalParser
from voice_assistant.core.mood import MoodState
from voice_assistant.models.silero_vad import SileroVAD
from voice_assistant.ports.asr import ASRPort
from voice_assistant.ports.llm import LLMPort
from voice_assistant.ports.tts import TTSPort

logger = logging.getLogger(__name__)

BARGE_IN_GATE_MS = 150
LISTENING_TIMEOUT_S = 30
TTS_CHUNK_TIMEOUT_S = 20
TTS_WORKER_TIMEOUT_S = 60

# Global monotonic start for absolute timestamps in logs
_T0 = time.monotonic()


@dataclass
class Turn:
    role: str            # "user" or "assistant"
    content: str
    source: str          # "voice" or "text"
    timestamp: float
    mood: str | None = None
    interrupted: bool = False
    partial: str | None = None


class Orchestrator:
    """Central pipeline orchestrator.

    Lifecycle:
        orch = Orchestrator(...)
        await orch.start()
        # runs until stopped
        await orch.stop()
    """

    def __init__(
        self,
        audio_input: AudioInput,
        vad: SileroVAD,
        asr: ASRPort,
        llm: LLMPort,
        tts: TTSPort,
        playlist: Playlist,
        playback: PlaybackManager,
        filler_cache: FillerCache,
        system_prompt: str = "",
        default_language: str = "en",
        supported_languages: list[str] | None = None,
    ):
        self._audio = audio_input
        self._vad = vad
        self._asr = asr
        self._llm = llm
        self._tts = tts
        self._playlist = playlist
        self._playback = playback
        self._fillers = filler_cache

        self._system_prompt = system_prompt or "You are a helpful voice assistant. Respond in plain conversational text. Never use markdown, bullet points, numbered lists, headers, or any formatting symbols."

        self._state = StateMachine(on_change=self._on_state_change)
        self._mood = MoodState()
        self._dialogue: list[Turn] = []
        self._current_language: str = default_language
        self._supported_languages: list[str] = supported_languages or [default_language]

        # Callback for chunk-level events (wired to IPC in main.py)
        self._on_chunk_synthesized: callable | None = None

        # Task handles for cancellation
        self._vad_task: asyncio.Task | None = None
        self._processing_task: asyncio.Task | None = None
        self._barge_in_speech_start: float | None = None
        self._generation_cancelled: bool = False
        self._pending_speech: np.ndarray | None = None

    @property
    def state(self) -> PipelineState:
        return self._state.state

    @property
    def dialogue(self) -> list[Turn]:
        return self._dialogue

    @property
    def mood(self) -> MoodState:
        return self._mood

    @staticmethod
    def _ts() -> str:
        """Absolute timestamp since orchestrator module load, for log correlation."""
        return f"[T+{time.monotonic() - _T0:.3f}s]"

    def _on_state_change(self, old: PipelineState, new: PipelineState) -> None:
        """Emit state change signal for IPC/UI."""
        logger.info("%s STATE: %s → %s (playlist_empty=%s)",
                    self._ts(), old.value, new.value, self._playlist.is_empty)

    async def start(self) -> None:
        """Start the orchestrator — begins VAD loop."""
        await self._audio.start()
        await self._playback.start()
        self._state.set_mode(PipelineMode.FULL)
        # State machine starts at IDLE — force it if needed
        if self._state.state != PipelineState.IDLE:
            self._state.transition(PipelineState.IDLE)
        self._vad_task = asyncio.create_task(self._vad_loop())
        logger.info("Orchestrator started — mic active, VAD loop running")

    async def stop(self) -> None:
        if self._vad_task:
            self._vad_task.cancel()
            try:
                await self._vad_task
            except asyncio.CancelledError:
                pass
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        await self._audio.stop()
        await self._playback.stop()

    async def handle_text_input(self, text: str) -> None:
        """Handle text input from Tauri UI.

        Typing doesn't trigger state changes — only submission.
        Detects language from typed text and updates current language.
        """
        logger.info("Text input [state=%s]: '%s'", self._state.state.value, text[:80])
        current = self._state.state

        if current == PipelineState.LISTENING:
            # Cancel listening, use text instead
            self._vad.reset()

        if current == PipelineState.PROCESSING:
            # Cancel current work
            self._generation_cancelled = True
            await self._llm.cancel()
            if self._processing_task:
                self._processing_task.cancel()
                try:
                    await self._processing_task
                except (asyncio.CancelledError, Exception):
                    pass
                self._processing_task = None

        if current == PipelineState.SPEAKING:
            # Barge-in
            await self._execute_barge_in()
            # Wait briefly for barge-in to settle
            await asyncio.sleep(0.05)

        # Detect language from typed text (same logic as ASR)
        detected = self._detect_text_language(text)
        if detected and detected != self._current_language:
            self._current_language = detected
            self._tts.set_language(detected)
            logger.info("Text input language switched to '%s'", detected)

        # Process text input
        self._state.transition(PipelineState.PROCESSING)
        self._processing_task = asyncio.create_task(
            self._process_turn(text, source="text")
        )

    def _detect_text_language(self, text: str) -> str | None:
        """Detect language of typed text using multi-strategy detection."""
        return detect_language(text, self._supported_languages)

    def _reset_audio_state(self, reason: str) -> None:
        """Centralized audio state cleanup — single place for all buffer resets.

        Called from barge-in AND stale playback stop. Resets:
        - Playback (fade + clear playlist)
        - VAD stored segments (without resetting detection — user may be speaking)
        - AudioInput chunk queue (stale chunks from sounddevice callback)
        - Orchestrator flags (_pending_speech, _barge_in_speech_start, _generation_cancelled)
        """
        # 1. Stop output
        self._playback.fade_out(15)
        dropped = self._playlist.drop_future()
        self._playlist.clear()

        # 2. Clear stale VAD segments (keep detection state — user may be speaking)
        stale = self._vad.clear_segments()

        # 3. Flush queued audio chunks from mic callback
        flushed = self._audio.flush_queue()

        # 4. Clear orchestrator flags
        self._pending_speech = None
        self._barge_in_speech_start = None
        self._generation_cancelled = True

        logger.info("%s AUDIO RESET (%s): dropped=%d playlist_chunks, "
                    "stale_vad=%d segments, flushed=%d audio_chunks",
                    self._ts(), reason, len(dropped), stale, flushed)

    def _stop_stale_playback(self) -> None:
        """Stop any audio still playing from a previous turn.

        After _process_turn transitions to IDLE, the playlist may still have
        audio playing in the sounddevice callback thread.  When a new turn
        starts we must stop it so old and new audio don't overlap.
        """
        if not self._playlist.is_empty:
            logger.info("%s STALE PLAYBACK — clearing", self._ts())
            self._reset_audio_state("stale-playback")

    async def _vad_loop(self) -> None:
        """Continuous VAD processing loop."""
        while True:
            try:
                chunk = await self._audio.read_chunk()
                event = self._vad.process_chunk(chunk)

                if event is None:
                    # Check for barge-in during SPEAKING or PROCESSING
                    if (self._state.state == PipelineState.SPEAKING
                            and self._vad.is_speech
                            and self._barge_in_speech_start is not None):
                        elapsed = (time.monotonic() - self._barge_in_speech_start) * 1000
                        if elapsed > BARGE_IN_GATE_MS:
                            logger.info("%s BARGE-IN gate passed (%.0fms) in state %s",
                                        self._ts(), elapsed, self._state.state.value)
                            await self._execute_barge_in()
                    continue

                if event.type == "speech_start":
                    logger.info("%s VAD speech_start — state=%s playlist_empty=%s",
                                self._ts(), self._state.state.value, self._playlist.is_empty)
                    if self._state.state == PipelineState.IDLE:
                        # Stop any leftover playback from previous turn before listening
                        self._stop_stale_playback()
                        self._state.transition(PipelineState.LISTENING)
                    elif self._state.state == PipelineState.SPEAKING:
                        # User speaks while assistant is talking → barge-in gate
                        self._barge_in_speech_start = time.monotonic()
                        logger.info("%s Barge-in gate started (state=SPEAKING)", self._ts())
                    elif self._state.state == PipelineState.PROCESSING:
                        # User continues speaking while first segment is being processed.
                        # Don't barge-in — let speech accumulate and queue at speech_end.
                        logger.info("%s Speech during PROCESSING — will queue at speech_end",
                                    self._ts())

                elif event.type == "speech_end":
                    logger.info("%s VAD speech_end (%.0fms) — state=%s",
                                self._ts(), event.duration_ms, self._state.state.value)
                    self._barge_in_speech_start = None

                    if self._state.state == PipelineState.LISTENING:
                        # Drain speech audio from VAD buffer and transcribe
                        speech_audio = self._vad.drain_speech_samples()
                        if speech_audio is not None:
                            logger.info("%s Drained %.1fs speech audio → starting ASR",
                                        self._ts(), len(speech_audio) / 16000)
                            self._state.transition(PipelineState.PROCESSING)
                            self._processing_task = asyncio.create_task(
                                self._process_voice_turn(speech_audio)
                            )
                        else:
                            logger.info("%s No speech audio drained — returning to IDLE", self._ts())
                            self._state.transition(PipelineState.IDLE)

                    elif self._state.state == PipelineState.PROCESSING:
                        # User continued speaking while first segment is being processed.
                        # Queue the speech — it will be processed after current turn completes.
                        speech_audio = self._vad.drain_speech_samples()
                        if speech_audio is not None:
                            if self._pending_speech is not None:
                                self._pending_speech = np.concatenate([self._pending_speech, speech_audio])
                            else:
                                self._pending_speech = speech_audio
                            logger.info("%s Speech ended during PROCESSING — queued %d samples (%.1fs), total pending=%.1fs",
                                        self._ts(), len(speech_audio), len(speech_audio) / 16000,
                                        len(self._pending_speech) / 16000)
                        else:
                            logger.info("%s Speech ended during PROCESSING but no audio drained", self._ts())

                    elif self._state.state == PipelineState.INTERRUPTED:
                        # After barge-in, user finished speaking
                        speech_audio = self._vad.drain_speech_samples()
                        if speech_audio is not None:
                            logger.info("%s Post-barge-in speech drained (%.1fs) → PROCESSING",
                                        self._ts(), len(speech_audio) / 16000)
                            self._state.transition(PipelineState.PROCESSING)
                            self._processing_task = asyncio.create_task(
                                self._process_voice_turn(speech_audio)
                            )
                        else:
                            logger.info("%s Post-barge-in: no speech audio → IDLE", self._ts())
                            self._state.transition(PipelineState.IDLE)

                    elif self._state.state == PipelineState.SPEAKING:
                        # Speech during SPEAKING — either short noise (sub-gate barge-in)
                        # or continuation that started during PROCESSING. Queue as pending
                        # so it's processed after the current turn finishes.
                        speech_audio = self._vad.drain_speech_samples()
                        if speech_audio is not None:
                            if self._pending_speech is not None:
                                self._pending_speech = np.concatenate([self._pending_speech, speech_audio])
                            else:
                                self._pending_speech = speech_audio
                            logger.info("%s Speech ended during SPEAKING (%.0fms) — queued as pending (%.1fs)",
                                        self._ts(), event.duration_ms, len(self._pending_speech) / 16000)
                        else:
                            logger.info("%s Speech ended during SPEAKING (%.0fms) — no audio drained",
                                        self._ts(), event.duration_ms)

                    elif self._state.state == PipelineState.IDLE:
                        # Spurious speech_end in IDLE — drain to keep buffer clean
                        drained = self._vad.drain_speech_samples()
                        if drained is not None:
                            logger.info("%s Spurious speech_end in IDLE — drained %d samples",
                                        self._ts(), len(drained))

                # Listening timeout
                if (self._state.state == PipelineState.LISTENING
                        and self._state.time_in_state > LISTENING_TIMEOUT_S):
                    logger.info("%s Listening timeout (%ds)", self._ts(), LISTENING_TIMEOUT_S)
                    self._state.transition(PipelineState.IDLE)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error in VAD loop")
                await asyncio.sleep(0.1)

    async def _execute_barge_in(self) -> None:
        """Execute barge-in: stop audio, cancel pipeline, reset all buffers."""
        prev_state = self._state.state
        logger.info("%s ═══ BARGE-IN START (from %s) ═══", self._ts(), prev_state.value)

        # 1. Capture played text BEFORE clearing playlist
        played = self._playlist.text_played()

        # 2. Cancel in-flight LLM/TTS task
        had_task = self._processing_task is not None
        if self._processing_task:
            self._processing_task.cancel()
            self._processing_task = None
        asyncio.ensure_future(self._llm.cancel())
        logger.info("%s  Processing task cancelled=%s, LLM cancel fired", self._ts(), had_task)

        # 3. Centralized audio state reset (playlist, VAD segments, audio queue, flags)
        self._reset_audio_state("barge-in")

        # 4. Update last assistant turn as interrupted
        if self._dialogue and self._dialogue[-1].role == "assistant":
            self._dialogue[-1].interrupted = True
            self._dialogue[-1].partial = played

        self._state.transition(PipelineState.INTERRUPTED)
        logger.info("%s ═══ BARGE-IN DONE → INTERRUPTED ═══", self._ts())

    async def _process_voice_turn(self, audio: np.ndarray) -> None:
        """Process a voice turn: ASR → LLM → TTS pipeline.

        After processing, checks for pending speech that arrived during
        this turn (VAD split the user's utterance at a natural pause).
        """
        try:
            logger.info("%s ─── VOICE TURN START (%.1fs audio) ───", self._ts(), len(audio) / 16000)
            asr_start = time.monotonic()
            result = await self._asr.transcribe(audio)
            asr_elapsed = time.monotonic() - asr_start
            text = result["text"]
            lang = result.get("language")
            logger.info("%s ASR done in %.2fs (%.1fs audio): lang=%s text='%s'",
                        self._ts(), asr_elapsed, len(audio) / 16000, lang, text[:120])
            if not text.strip():
                logger.info("%s ASR returned empty text", self._ts())
                # Empty transcription — but still check for pending speech
                if self._pending_speech is not None:
                    pending = self._pending_speech
                    self._pending_speech = None
                    logger.info("%s Empty ASR but pending speech exists — processing %d samples (%.1fs)",
                                self._ts(), len(pending), len(pending) / 16000)
                    await self._process_voice_turn(pending)
                else:
                    self._state.transition(PipelineState.IDLE)
                return

            if lang and lang != self._current_language:
                self._current_language = lang
                self._tts.set_language(lang)
                logger.info("%s Language switched to '%s'", self._ts(), lang)
            await self._process_turn(text, source="voice")

            # Process speech that arrived while this turn was running
            if self._pending_speech is not None:
                pending = self._pending_speech
                self._pending_speech = None
                logger.info("%s ─── PENDING SPEECH found (%d samples, %.1fs) — starting new voice turn ───",
                            self._ts(), len(pending), len(pending) / 16000)
                self._state.transition(PipelineState.PROCESSING)
                await self._process_voice_turn(pending)

        except asyncio.CancelledError:
            logger.info("%s Voice turn CANCELLED", self._ts())
            raise
        except Exception:
            logger.exception("%s Error in voice turn processing", self._ts())
            self._pending_speech = None  # clear on error to avoid stale audio
            self._state.transition(PipelineState.IDLE)

    async def _process_turn(self, user_text: str, source: str) -> None:
        """Process a turn through the full pipeline: context → LLM → chunk → TTS → play.

        TTS runs in a separate consumer task fed via asyncio.Queue, so LLM
        streaming is never blocked waiting for synthesis.  Chunks are consumed
        sequentially by the TTS worker to guarantee playlist ordering.
        """
        tts_queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()
        tts_worker: asyncio.Task | None = None
        turn_language = self._current_language  # snapshot language for this turn
        turn_start = time.monotonic()
        self._generation_cancelled = False

        logger.info("%s ═══ PROCESS TURN START (source=%s, lang=%s) ═══ text='%s'",
                    self._ts(), source, turn_language, user_text[:120])

        try:
            # Record user turn
            user_turn = Turn(
                role="user", content=user_text, source=source,
                timestamp=time.time(),
            )
            self._dialogue.append(user_turn)
            logger.info("%s Dialogue now has %d turns", self._ts(), len(self._dialogue))

            # Filler (conditional)
            self._fillers.record_turn()
            filler = self._fillers.get_filler()
            if filler:
                self._playlist.append(filler)
                logger.info("%s Filler queued", self._ts())

            # Build messages for LLM
            lang_names = {"en": "English", "hr": "Croatian", "ru": "Russian",
                          "de": "German", "fr": "French", "es": "Spanish",
                          "it": "Italian", "pt": "Portuguese"}
            lang_name = lang_names.get(turn_language, turn_language)
            system_with_lang = (
                f"{self._system_prompt}\n\n"
                f"IMPORTANT: The user is currently speaking {lang_name}. "
                f"You MUST respond in {lang_name}. "
                f"NEVER use markdown, asterisks, bullet points, numbered lists, "
                f"bold, italic, headers, or any formatting. Plain text only — "
                f"this is a spoken conversation."
            )
            messages = [
                {"role": "system", "content": system_with_lang},
            ]
            for turn in self._dialogue[-10:]:  # last 10 turns
                messages.append({"role": turn.role, "content": turn.content})
            logger.info("%s LLM context: %d messages", self._ts(), len(messages))

            # Start TTS worker — consumes chunks from queue, synthesizes in order
            tts_chunk_idx = 0

            async def _tts_consumer():
                nonlocal tts_chunk_idx
                while True:
                    item = await tts_queue.get()
                    if item is None:
                        logger.info("%s TTS worker received sentinel — stopping", self._ts())
                        break  # sentinel — no more chunks
                    if self._generation_cancelled:
                        logger.info("%s TTS worker skipping chunk (generation cancelled)", self._ts())
                        continue
                    tts_chunk_idx += 1
                    chunk_text, chunk_lang = item
                    logger.info("%s TTS worker chunk %d START (%d chars): '%.50s'",
                                self._ts(), tts_chunk_idx, len(chunk_text), chunk_text)
                    await self._synthesize_and_queue(chunk_text, chunk_lang)
                    logger.info("%s TTS worker chunk %d DONE", self._ts(), tts_chunk_idx)

            tts_worker = asyncio.create_task(_tts_consumer())

            # Stream LLM → parse mood → chunk → enqueue for TTS
            llm_start = time.monotonic()
            first_token_at = None
            token_count = 0
            chunk_count = 0
            mood_parser = MoodSignalParser(
                on_mood=lambda tone, intensity: self._mood.update(tone, intensity)
            )
            chunker = SentenceChunker()
            full_response = []

            # Reasoning is off by default for voice — fast responses matter.
            # Orchestrator can enable it for complex queries in the future.
            use_thinking = False
            logger.info("%s LLM streaming started (thinking=%s)", self._ts(), use_thinking)
            async for token in self._llm.stream(messages, thinking=use_thinking):
                if self._generation_cancelled:
                    logger.info("%s LLM token ignored — generation cancelled (token #%d)", self._ts(), token_count + 1)
                    break
                token_count += 1
                if first_token_at is None:
                    first_token_at = time.monotonic()
                    logger.info("%s LLM FIRST TOKEN (TTFT=%.3fs)",
                                self._ts(), first_token_at - llm_start)

                clean = mood_parser.feed(token)
                if clean:
                    full_response.append(clean)
                    chunk_text = chunker.feed(clean)
                    if chunk_text:
                        chunk_count += 1
                        tts_queue.put_nowait((chunk_text, turn_language))
                        logger.info("%s Chunk %d queued (%d chars): '%.60s'",
                                    self._ts(), chunk_count, len(chunk_text), chunk_text)
                        if self._state.state == PipelineState.PROCESSING:
                            self._state.transition(PipelineState.SPEAKING)

            # Flush remaining text
            remaining = mood_parser.finalize()
            if remaining:
                full_response.append(remaining)
                chunk_text = chunker.feed(remaining)
                if chunk_text:
                    chunk_count += 1
                    tts_queue.put_nowait((chunk_text, turn_language))
                    logger.info("%s Chunk %d queued (finalize, %d chars): '%.60s'",
                                self._ts(), chunk_count, len(chunk_text), chunk_text)

            final_chunk = chunker.flush()
            if final_chunk:
                chunk_count += 1
                tts_queue.put_nowait((final_chunk, turn_language))
                logger.info("%s Chunk %d queued (flush, %d chars): '%.60s'",
                            self._ts(), chunk_count, len(final_chunk), final_chunk)

            llm_elapsed = time.monotonic() - llm_start
            speed = token_count / llm_elapsed if llm_elapsed > 0 else 0
            logger.info("%s LLM DONE: %.2fs, %d tokens (%.1f t/s), %d chunks",
                        self._ts(), llm_elapsed, token_count, speed, chunk_count)

            # Signal TTS worker to finish and wait with timeout
            tts_queue.put_nowait(None)
            tts_wait_start = time.monotonic()
            try:
                await asyncio.wait_for(tts_worker, timeout=TTS_WORKER_TIMEOUT_S)
            except asyncio.TimeoutError:
                logger.error("%s TTS worker timed out after %ds, cancelling", self._ts(), TTS_WORKER_TIMEOUT_S)
                tts_worker.cancel()
            tts_wait = time.monotonic() - tts_wait_start
            logger.info("%s TTS worker finished (waited %.2fs)", self._ts(), tts_wait)

            # Record assistant turn
            response_text = "".join(full_response)
            assistant_turn = Turn(
                role="assistant", content=response_text, source="voice",
                timestamp=time.time(), mood=self._mood.mood,
            )
            self._dialogue.append(assistant_turn)
            self._mood.decay()

            # Don't block on playback — transition to IDLE so user can
            # start a new turn while audio is still playing.  Barge-in
            # handles interruption if the user speaks during playback.
            total = time.monotonic() - turn_start
            logger.info("%s ═══ TURN DONE: total=%.2fs LLM=%.2fs TTS_wait=%.2fs chunks=%d tokens=%d playlist_empty=%s ═══",
                        self._ts(), total, llm_elapsed, tts_wait, chunk_count, token_count,
                        self._playlist.is_empty)
            self._playlist.free_played()
            self._state.transition(PipelineState.IDLE)

        except asyncio.CancelledError:
            logger.info("%s PROCESS TURN CANCELLED", self._ts())
            if tts_worker and not tts_worker.done():
                tts_worker.cancel()
            raise
        except Exception:
            logger.exception("%s Error in turn processing", self._ts())
            if tts_worker and not tts_worker.done():
                tts_worker.cancel()
            self._state.transition(PipelineState.IDLE)

    async def _synthesize_and_queue(self, text: str, lang: str) -> None:
        """Synthesize a text chunk and add to playlist."""
        if self._generation_cancelled:
            logger.info("%s TTS skipped (cancelled): '%.40s'", self._ts(), text)
            return
        voice_params = self._mood.get_voice_params()
        voice_params["language"] = lang
        tags = voice_params.get("tags", [])
        if tags:
            logger.info("%s TTS mood=%s tags=%s for: '%.50s'",
                        self._ts(), self._mood.mood, tags, text)
        t0 = time.monotonic()
        try:
            audio = await asyncio.wait_for(
                self._tts.synthesize(text, voice_params),
                timeout=TTS_CHUNK_TIMEOUT_S,
            )
            elapsed = time.monotonic() - t0
            duration = len(audio) / 24000
            logger.info("%s TTS [%s] %.2fs synth → %.2fs audio (RTF=%.2f) mood=%s: '%.50s'",
                        self._ts(), lang, elapsed, duration,
                        elapsed / duration if duration else 0, self._mood.mood, text)
            if self._generation_cancelled:
                logger.info("%s TTS result discarded (cancelled during synth): '%.40s'", self._ts(), text)
                return
            self._playlist.append(AudioChunk(audio=audio, text=text, source="tts"))
            if self._on_chunk_synthesized:
                self._on_chunk_synthesized(text)
        except asyncio.TimeoutError:
            logger.error("%s TTS chunk timed out after %ds: '%.40s'", self._ts(), TTS_CHUNK_TIMEOUT_S, text)
        except Exception:
            logger.exception("%s TTS synthesis failed for: '%.40s'", self._ts(), text)
