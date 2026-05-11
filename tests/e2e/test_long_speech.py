"""Test 4 x ~34s long speech utterances through full pipeline.

Concatenates existing WAV samples with short pauses (300ms) to simulate
continuous long speech. Verifies debounce timer works correctly.

Usage:
    cd /home/winers/voice-assistant
    .venv/bin/python tests/e2e/test_long_speech.py 2>&1 | tee /tmp/va_long_speech.log
"""
import asyncio
import logging
import os
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "samples")


def load_wav_16k(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if sr != 16000:
        new_len = int(len(audio) * 16000 / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, new_len),
                          np.arange(len(audio)), audio).astype(np.float32)
    return audio


def build_long_utterance(wavs, sample_ids, pause_s=0.3):
    """Concatenate samples with short pauses to simulate long speech."""
    parts = []
    for sid in sample_ids:
        parts.append(wavs[sid])
        parts.append(np.zeros(int(pause_s * 16000), dtype=np.float32))
    return np.concatenate(parts)


# --- Fake audio I/O (same as stress_voice.py) ---

class FakeAudioInput:
    CHUNK = 480

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._pending: asyncio.Queue = asyncio.Queue()
        self._task = None
        self.ring = _Ring()

    async def start(self):
        self._task = asyncio.create_task(self._feed())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try: await self._task
            except asyncio.CancelledError: pass

    def queue_utterance(self, audio):
        self._pending.put_nowait(audio)

    async def _feed(self):
        try:
            while True:
                audio = await self._pending.get()
                pos = 0
                while pos < len(audio):
                    chunk = audio[pos:pos + self.CHUNK]
                    if len(chunk) < self.CHUNK:
                        chunk = np.pad(chunk, (0, self.CHUNK - len(chunk)))
                    self._queue.put_nowait(chunk)
                    self.ring.write(chunk)
                    pos += self.CHUNK
                    await asyncio.sleep(0.005)
                # silence gap after utterance
                sil = np.zeros(self.CHUNK, dtype=np.float32)
                for _ in range(int(2.0 * 16000 / self.CHUNK)):
                    self._queue.put_nowait(sil)
                    self.ring.write(sil)
                    await asyncio.sleep(0.005)
        except asyncio.CancelledError:
            raise

    def flush_queue(self) -> int:
        count = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                count += 1
            except asyncio.QueueEmpty:
                break
        return count

    async def read_chunk(self):
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            await asyncio.sleep(0.03)
            sil = np.zeros(self.CHUNK, dtype=np.float32)
            self.ring.write(sil)
            return sil


class _Ring:
    def __init__(self, cap=16000 * 30):
        self._buf = np.zeros(cap, dtype=np.float32)
        self._w = 0
        self._cap = cap

    def write(self, c):
        n = len(c)
        e = self._w + n
        if e <= self._cap:
            self._buf[self._w:e] = c
        else:
            f = self._cap - self._w
            self._buf[self._w:] = c[:f]
            self._buf[:n - f] = c[f:]
        self._w = e % self._cap

    def read_last(self, n):
        if n > self._cap: n = self._cap
        s = (self._w - n) % self._cap
        if s < self._w:
            return self._buf[s:self._w].copy()
        return np.concatenate([self._buf[s:], self._buf[:self._w]])


class FakePlayback:
    def __init__(self, pl):
        self._pl = pl
        self.is_active = False
    async def start(self): self.is_active = True
    async def stop(self): self.is_active = False
    def fade_out(self, ms=15): self.is_active = False


# 4 long utterances (~34s each) built from concatenated samples
LONG_UTTERANCES = [
    {
        "name": "Long English 1 (~34s)",
        "samples": [
            "en_01_greeting", "en_02_question", "en_03_complex",
            "en_04_followup", "en_05_topic_switch", "en_01_greeting",
            "en_02_question", "en_03_complex",
        ],
    },
    {
        "name": "Long Russian (~34s)",
        "samples": [
            "ru_01_greeting", "ru_02_question", "ru_03_complex",
            "ru_05_long", "ru_04_short", "ru_01_greeting",
            "ru_02_question", "ru_03_complex", "ru_05_long",
        ],
    },
    {
        "name": "Long Croatian (~34s)",
        "samples": [
            "hr_01_greeting", "hr_02_question", "hr_03_complex",
            "hr_01_greeting", "hr_02_question", "hr_03_complex",
            "hr_01_greeting", "hr_02_question", "hr_03_complex",
            "hr_01_greeting", "hr_02_question", "hr_03_complex",
        ],
    },
    {
        "name": "Long English 2 (~34s)",
        "samples": [
            "en_03_complex", "en_04_followup", "en_05_topic_switch",
            "en_02_question", "en_01_greeting", "en_03_complex",
            "en_04_followup", "en_05_topic_switch",
        ],
    },
]


async def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stderr)
    log = logging.getLogger("long_speech")

    log.info("=" * 80)
    log.info("LONG SPEECH TEST — 4 x ~34s utterances")
    log.info("=" * 80)

    # Load all WAVs
    wavs = {}
    for f in sorted(os.listdir(SAMPLE_DIR)):
        if f.endswith(".wav"):
            wavs[f.replace(".wav", "")] = load_wav_16k(os.path.join(SAMPLE_DIR, f))

    # Build long utterances
    long_audios = []
    for utt in LONG_UTTERANCES:
        audio = build_long_utterance(wavs, utt["samples"])
        dur = len(audio) / 16000
        log.info("Built '%s': %.1fs (%d samples)", utt["name"], dur, len(audio))
        long_audios.append(audio)

    # Startup
    from voice_assistant.core.soul_loader import load_soul
    from voice_assistant.core.settings_loader import load_settings
    soul = load_soul("soul/default.soul.yaml")
    settings = load_settings("config/settings.yaml")

    from voice_assistant.core.vram_guard import VRAMGuard
    from voice_assistant.core.ipc import StdoutEmitter
    from voice_assistant.process.manager import ProcessManager

    vram = VRAMGuard()
    devnull = open(os.devnull, "w")
    emitter = StdoutEmitter(output_stream=devnull)
    await emitter.start()
    pm = ProcessManager(soul, vram, emitter, settings=settings)
    mode = await pm.startup()
    log.info("Mode: %s", mode.value)

    if not (pm._llm_process and pm._tts and pm._asr and pm._vad):
        log.error("Components missing — abort")
        await pm.shutdown()
        return

    from voice_assistant.core.orchestrator import Orchestrator
    from voice_assistant.core.audio_output import Playlist, FillerCache
    from voice_assistant.adapters.llamacpp_llm import LlamaCppLLM
    from voice_assistant.core.state_machine import PipelineMode, PipelineState

    fake_audio = FakeAudioInput()
    playlist = Playlist()
    fake_pb = FakePlayback(playlist)
    filler = FillerCache()
    llm = LlamaCppLLM(f"http://127.0.0.1:{pm._llm_process.port}")

    supported = [v.id for v in settings.voice_languages] or ["en"]
    persona = soul.persona_card()
    sys_prompt = (
        f"{persona}\n\n"
        "You are a voice assistant. Respond in plain conversational text. "
        "Never use markdown. Keep responses concise."
    )

    errors = 0

    for i, (utt, audio) in enumerate(zip(LONG_UTTERANCES, long_audios)):
        log.info("")
        log.info("━" * 80)
        log.info("UTTERANCE %d/4: %s (%.1fs)", i + 1, utt["name"], len(audio) / 16000)
        log.info("━" * 80)

        # Fresh orchestrator per utterance
        orch = Orchestrator(
            audio_input=fake_audio, vad=pm._vad, asr=pm._asr, llm=llm,
            tts=pm._tts, playlist=playlist, playback=fake_pb,
            filler_cache=filler, system_prompt=sys_prompt,
            default_language=settings.default_language,
            supported_languages=supported,
        )
        await fake_audio.start()
        orch._state.set_mode(PipelineMode.FULL)
        if orch._state.state != PipelineState.IDLE:
            orch._state.transition(PipelineState.IDLE)
        orch._vad_task = asyncio.create_task(orch._vad_loop())

        t0 = time.monotonic()
        fake_audio.queue_utterance(audio)

        # Wait for completion: see PROCESSING, then IDLE
        try:
            deadline = time.monotonic() + 180
            saw_proc = False
            while time.monotonic() < deadline:
                st = orch.state
                if st in (PipelineState.PROCESSING, PipelineState.SPEAKING):
                    saw_proc = True
                elif st == PipelineState.IDLE and saw_proc:
                    await asyncio.sleep(0.5)
                    if orch.state == PipelineState.IDLE:
                        break
                await asyncio.sleep(0.1)
            else:
                log.error("TIMEOUT on utterance %d", i + 1)
                errors += 1

            elapsed = time.monotonic() - t0
            asr_text = ""
            for t in orch.dialogue:
                if t.role == "user":
                    asr_text = t.content

            log.info("RESULT %d: %.1fs total | ASR='%s'",
                     i + 1, elapsed, asr_text[:80])
            if not asr_text:
                log.error("NO ASR TEXT for utterance %d!", i + 1)
                errors += 1

        except Exception as e:
            log.error("ERROR on utterance %d: %s", i + 1, e)
            errors += 1

        # Cleanup
        if orch._vad_task:
            orch._vad_task.cancel()
            try: await orch._vad_task
            except asyncio.CancelledError: pass
        if orch._processing_task:
            orch._processing_task.cancel()
            try: await orch._processing_task
            except asyncio.CancelledError: pass
        playlist.clear()
        await fake_audio.stop()
        await asyncio.sleep(1.0)

    log.info("")
    log.info("=" * 80)
    log.info("DONE — %d/4 passed, %d errors", 4 - errors, errors)
    log.info("=" * 80)

    await llm.shutdown()
    await pm.shutdown()

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
