"""Pipeline orchestrator — wires all components together.

Owns the state machine, coordinates: AudioInput → VAD → ASR → LLM → Chunker → TTS → Playlist.
Handles barge-in (300ms gate), text input, filler playback.

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
from voice_assistant.core.mood_signal_parser import MoodSignalParser
from voice_assistant.core.mood import MoodState
from voice_assistant.models.silero_vad import SileroVAD
from voice_assistant.ports.asr import ASRPort
from voice_assistant.ports.llm import LLMPort
from voice_assistant.ports.tts import TTSPort

logger = logging.getLogger(__name__)

BARGE_IN_GATE_MS = 300
LISTENING_TIMEOUT_S = 30


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
    ):
        self._audio = audio_input
        self._vad = vad
        self._asr = asr
        self._llm = llm
        self._tts = tts
        self._playlist = playlist
        self._playback = playback
        self._fillers = filler_cache

        self._state = StateMachine(on_change=self._on_state_change)
        self._mood = MoodState()
        self._dialogue: list[Turn] = []
        self._current_language: str = "en"

        # Task handles for cancellation
        self._vad_task: asyncio.Task | None = None
        self._processing_task: asyncio.Task | None = None
        self._barge_in_speech_start: float | None = None

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
        self._state.transition(PipelineState.IDLE)
        self._vad_task = asyncio.create_task(self._vad_loop())
        logger.info("Orchestrator started")

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
        """
        current = self._state.state

        if current == PipelineState.LISTENING:
            # Cancel listening, use text instead
            self._vad.reset()

        if current == PipelineState.PROCESSING:
            # Cancel current work
            await self._llm.cancel()
            if self._processing_task:
                self._processing_task.cancel()

        if current == PipelineState.SPEAKING:
            # Barge-in
            await self._execute_barge_in()

        # Process text input
        self._state.transition(PipelineState.PROCESSING)
        self._processing_task = asyncio.create_task(
            self._process_turn(text, source="text")
        )

    async def _vad_loop(self) -> None:
        """Continuous VAD processing loop."""
        while True:
            try:
                chunk = await self._audio.read_chunk()
                event = self._vad.process_chunk(chunk)

                if event is None:
                    # Check for barge-in during SPEAKING
                    if (self._state.state == PipelineState.SPEAKING
                            and self._vad.is_speech
                            and self._barge_in_speech_start is not None):
                        elapsed = (time.monotonic() - self._barge_in_speech_start) * 1000
                        if elapsed > BARGE_IN_GATE_MS:
                            await self._execute_barge_in()
                    continue

                if event.type == "speech_start":
                    if self._state.state == PipelineState.IDLE:
                        self._state.transition(PipelineState.LISTENING)
                    elif self._state.state == PipelineState.SPEAKING:
                        # Start barge-in gate timer
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
        """Execute barge-in: fade out, stash partial, drop future, cancel pipeline."""
        logger.info("Barge-in executing")
        self._playback.fade_out(30)

        played = self._playlist.text_played()
        self._playlist.drop_future()
        self._playlist.clear()

        # Cancel in-flight LLM/TTS
        await self._llm.cancel()
        if self._processing_task:
            self._processing_task.cancel()
            self._processing_task = None

        # Update last assistant turn as interrupted
        if self._dialogue and self._dialogue[-1].role == "assistant":
            self._dialogue[-1].interrupted = True
            self._dialogue[-1].partial = played

        self._barge_in_speech_start = None
        self._state.transition(PipelineState.INTERRUPTED)

    async def _process_voice_turn(self, audio: np.ndarray) -> None:
        """Process a voice turn: ASR → LLM → TTS pipeline."""
        try:
            result = await self._asr.transcribe(audio)
            text = result["text"]
            if not text.strip():
                self._state.transition(PipelineState.IDLE)
                return

            self._current_language = result.get("language", self._current_language)
            await self._process_turn(text, source="voice")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error in voice turn processing")
            self._state.transition(PipelineState.IDLE)

    async def _process_turn(self, user_text: str, source: str) -> None:
        """Process a turn through the full pipeline: context → LLM → chunk → TTS → play."""
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
            # Simplified — ContextBuilder (Stage 11) will replace this
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
            ]
            for turn in self._dialogue[-10:]:  # last 10 turns
                messages.append({"role": turn.role, "content": turn.content})

            # Stream LLM → parse mood → chunk → TTS → playlist
            mood_parser = MoodSignalParser(
                on_mood=lambda tone, intensity: self._mood.update(tone, intensity)
            )
            chunker = SentenceChunker()
            full_response = []

            async for token in self._llm.stream(messages):
                clean = mood_parser.feed(token)
                if clean:
                    full_response.append(clean)
                    chunk_text = chunker.feed(clean)
                    if chunk_text:
                        await self._synthesize_and_queue(chunk_text)
                        if self._state.state == PipelineState.PROCESSING:
                            self._state.transition(PipelineState.SPEAKING)

            # Flush remaining
            remaining = mood_parser.finalize()
            if remaining:
                full_response.append(remaining)
                chunk_text = chunker.feed(remaining)
                if chunk_text:
                    await self._synthesize_and_queue(chunk_text)

            final_chunk = chunker.flush()
            if final_chunk:
                await self._synthesize_and_queue(final_chunk)

            # Record assistant turn
            response_text = "".join(full_response)
            assistant_turn = Turn(
                role="assistant", content=response_text, source="voice",
                timestamp=time.time(), mood=self._mood.mood,
            )
            self._dialogue.append(assistant_turn)
            self._mood.decay()

            # Wait for playback to finish
            await self._playlist.wait_until_done()
            self._state.transition(PipelineState.IDLE)

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error in turn processing")
            self._state.transition(PipelineState.IDLE)

    async def _synthesize_and_queue(self, text: str) -> None:
        """Synthesize a text chunk and add to playlist."""
        voice_params = self._mood.get_voice_params()
        try:
            audio = await self._tts.synthesize(text, voice_params)
            self._playlist.append(AudioChunk(audio=audio, text=text, source="tts"))
        except Exception:
            logger.exception("TTS synthesis failed for: %s", text[:40])
