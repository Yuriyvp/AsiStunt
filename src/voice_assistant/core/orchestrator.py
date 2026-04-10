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

    @property
    def state(self) -> PipelineState:
        return self._state.state

    @property
    def dialogue(self) -> list[Turn]:
        return self._dialogue

    @property
    def mood(self) -> MoodState:
        return self._mood

    def _on_state_change(self, old: PipelineState, new: PipelineState) -> None:
        """Emit state change signal for IPC/UI."""
        # Will be wired to IPC emitter in Stage 13
        logger.info("State change: %s → %s", old.value, new.value)

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

    async def _vad_loop(self) -> None:
        """Continuous VAD processing loop."""
        while True:
            try:
                chunk = await self._audio.read_chunk()
                event = self._vad.process_chunk(chunk)

                if event is None:
                    # Check for barge-in during SPEAKING or PROCESSING
                    if (self._state.state in (PipelineState.SPEAKING, PipelineState.PROCESSING)
                            and self._vad.is_speech
                            and self._barge_in_speech_start is not None):
                        elapsed = (time.monotonic() - self._barge_in_speech_start) * 1000
                        if elapsed > BARGE_IN_GATE_MS:
                            await self._execute_barge_in()
                    continue

                if event.type == "speech_start":
                    if self._state.state == PipelineState.IDLE:
                        self._state.transition(PipelineState.LISTENING)
                    elif self._state.state in (PipelineState.SPEAKING, PipelineState.PROCESSING):
                        # User speaks while assistant is generating or speaking → barge-in gate
                        self._barge_in_speech_start = time.monotonic()

                elif event.type == "speech_end":
                    self._barge_in_speech_start = None

                    if self._state.state == PipelineState.LISTENING:
                        # Drain speech audio from VAD buffer and transcribe
                        speech_audio = self._vad.drain_speech_samples()
                        if speech_audio is not None:
                            self._state.transition(PipelineState.PROCESSING)
                            self._processing_task = asyncio.create_task(
                                self._process_voice_turn(speech_audio)
                            )
                        else:
                            self._state.transition(PipelineState.IDLE)

                    elif self._state.state == PipelineState.INTERRUPTED:
                        # After barge-in, user finished speaking
                        speech_audio = self._vad.drain_speech_samples()
                        if speech_audio is not None:
                            self._state.transition(PipelineState.PROCESSING)
                            self._processing_task = asyncio.create_task(
                                self._process_voice_turn(speech_audio)
                            )
                        else:
                            self._state.transition(PipelineState.IDLE)

                # Listening timeout
                if (self._state.state == PipelineState.LISTENING
                        and self._state.time_in_state > LISTENING_TIMEOUT_S):
                    logger.info("Listening timeout (%ds)", LISTENING_TIMEOUT_S)
                    self._state.transition(PipelineState.IDLE)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error in VAD loop")
                await asyncio.sleep(0.1)

    async def _execute_barge_in(self) -> None:
        """Execute barge-in: stop audio immediately, cancel pipeline, clean up."""
        logger.info("Barge-in executing")

        # 1. Stop audio output immediately (non-blocking, runs in callback thread)
        self._playback.fade_out(15)
        played = self._playlist.text_played()
        self._playlist.drop_future()

        # 2. Cancel in-flight LLM/TTS task first (don't await LLM cancel — fire and forget)
        if self._processing_task:
            self._processing_task.cancel()
            self._processing_task = None

        # 3. Clear playlist and cancel LLM connection concurrently
        self._playlist.clear()
        # LLM cancel is fast (closes HTTP socket) but don't block barge-in on it
        asyncio.ensure_future(self._llm.cancel())

        # 4. Update last assistant turn as interrupted
        if self._dialogue and self._dialogue[-1].role == "assistant":
            self._dialogue[-1].interrupted = True
            self._dialogue[-1].partial = played

        self._barge_in_speech_start = None
        self._generation_cancelled = True
        self._state.transition(PipelineState.INTERRUPTED)

    async def _process_voice_turn(self, audio: np.ndarray) -> None:
        """Process a voice turn: ASR → LLM → TTS pipeline."""
        try:
            asr_start = time.monotonic()
            result = await self._asr.transcribe(audio)
            asr_elapsed = time.monotonic() - asr_start
            text = result["text"]
            lang = result.get("language")
            logger.info("ASR %.2fs (%.1fs audio): lang=%s text='%s'",
                        asr_elapsed, len(audio) / 16000, lang, text[:80])
            if not text.strip():
                self._state.transition(PipelineState.IDLE)
                return

            if lang and lang != self._current_language:
                self._current_language = lang
                self._tts.set_language(lang)
                logger.info("Language switched to '%s'", lang)
            await self._process_turn(text, source="voice")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error in voice turn processing")
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
        try:
            # Record user turn
            user_turn = Turn(
                role="user", content=user_text, source=source,
                timestamp=time.time(),
            )
            self._dialogue.append(user_turn)

            # Filler (conditional)
            self._fillers.record_turn()
            filler = self._fillers.get_filler()
            if filler:
                self._playlist.append(filler)

            # Build messages for LLM
            lang_names = {"en": "English", "hr": "Croatian", "ru": "Russian",
                          "de": "German", "fr": "French", "es": "Spanish",
                          "it": "Italian", "pt": "Portuguese"}
            lang_name = lang_names.get(turn_language, turn_language)
            system_with_lang = (
                f"{self._system_prompt}\n\n"
                f"IMPORTANT: The user is currently speaking {lang_name}. "
                f"You MUST respond in {lang_name}."
            )
            messages = [
                {"role": "system", "content": system_with_lang},
            ]
            for turn in self._dialogue[-10:]:  # last 10 turns
                messages.append({"role": turn.role, "content": turn.content})

            # Start TTS worker — consumes chunks from queue, synthesizes in order
            tts_chunk_idx = 0

            async def _tts_consumer():
                nonlocal tts_chunk_idx
                while True:
                    item = await tts_queue.get()
                    if item is None:
                        break  # sentinel — no more chunks
                    tts_chunk_idx += 1
                    chunk_text, chunk_lang = item
                    logger.info("[%.3fs] TTS worker starting chunk %d (%d chars)",
                                time.monotonic() - turn_start, tts_chunk_idx, len(chunk_text))
                    await self._synthesize_and_queue(chunk_text, chunk_lang)

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
            logger.info("[%.3fs] LLM streaming started (thinking=%s)", time.monotonic() - turn_start, use_thinking)
            async for token in self._llm.stream(messages, thinking=use_thinking):
                token_count += 1
                if first_token_at is None:
                    first_token_at = time.monotonic()
                    logger.info("[%.3fs] LLM first token (TTFT=%.3fs)",
                                first_token_at - turn_start,
                                first_token_at - llm_start)

                clean = mood_parser.feed(token)
                if clean:
                    full_response.append(clean)
                    chunk_text = chunker.feed(clean)
                    if chunk_text:
                        chunk_count += 1
                        tts_queue.put_nowait((chunk_text, turn_language))
                        logger.info("[%.3fs] Chunk %d queued (%d chars): %.50s",
                                    time.monotonic() - turn_start,
                                    chunk_count, len(chunk_text), chunk_text)
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
                    logger.info("[%.3fs] Chunk %d queued (finalize, %d chars): %.50s",
                                time.monotonic() - turn_start,
                                chunk_count, len(chunk_text), chunk_text)

            final_chunk = chunker.flush()
            if final_chunk:
                chunk_count += 1
                tts_queue.put_nowait((final_chunk, turn_language))
                logger.info("[%.3fs] Chunk %d queued (flush, %d chars): %.50s",
                            time.monotonic() - turn_start,
                            chunk_count, len(final_chunk), final_chunk)

            llm_elapsed = time.monotonic() - llm_start
            logger.info("[%.3fs] LLM done: %.2fs, %d tokens, %d chunks",
                        time.monotonic() - turn_start,
                        llm_elapsed, token_count, chunk_count)

            # Signal TTS worker to finish and wait with timeout
            tts_queue.put_nowait(None)
            try:
                await asyncio.wait_for(tts_worker, timeout=TTS_WORKER_TIMEOUT_S)
            except asyncio.TimeoutError:
                logger.error("TTS worker timed out after %ds, cancelling", TTS_WORKER_TIMEOUT_S)
                tts_worker.cancel()

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
            tts_elapsed = time.monotonic() - turn_start - llm_elapsed
            total = time.monotonic() - turn_start
            logger.info("[%.3fs] TURN DONE: total=%.2fs LLM=%.2fs TTS_wait=%.2fs chunks=%d tokens=%d",
                        time.monotonic() - turn_start,
                        total, llm_elapsed, tts_elapsed, chunk_count, token_count)
            self._state.transition(PipelineState.IDLE)

        except asyncio.CancelledError:
            if tts_worker and not tts_worker.done():
                tts_worker.cancel()
            raise
        except Exception:
            logger.exception("Error in turn processing")
            if tts_worker and not tts_worker.done():
                tts_worker.cancel()
            self._state.transition(PipelineState.IDLE)

    async def _synthesize_and_queue(self, text: str, lang: str) -> None:
        """Synthesize a text chunk and add to playlist."""
        voice_params = self._mood.get_voice_params()
        voice_params["language"] = lang
        t0 = time.monotonic()
        try:
            audio = await asyncio.wait_for(
                self._tts.synthesize(text, voice_params),
                timeout=TTS_CHUNK_TIMEOUT_S,
            )
            elapsed = time.monotonic() - t0
            duration = len(audio) / 24000
            logger.info("TTS [%s] %.2fs synth → %.2fs audio (RTF=%.2f): %.40s",
                        lang, elapsed, duration, elapsed / duration if duration else 0, text)
            self._playlist.append(AudioChunk(audio=audio, text=text, source="tts"))
            if self._on_chunk_synthesized:
                self._on_chunk_synthesized(text)
        except asyncio.TimeoutError:
            logger.error("TTS chunk timed out after %ds: %s", TTS_CHUNK_TIMEOUT_S, text[:40])
        except Exception:
            logger.exception("TTS synthesis failed for: %s", text[:40])
