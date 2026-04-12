"""Voice stress test — 10 real multi-turn conversations via WAV samples.

Feeds pre-recorded audio through the full pipeline:
  WAV → VAD → ASR → LLM → TTS → Playlist

Prints a timestamps table at the end.

Usage:
    cd /home/winers/voice-assistant
    .venv/bin/python tests/e2e/stress_voice.py 2>&1 | tee /tmp/va_stress_voice.log
"""
import asyncio
import io
import logging
import os
import resource
import sys
import time
import wave
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "samples")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0


def get_gpu_mb():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(h).used / 1024 / 1024
    except Exception:
        return 0


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


# ---------------------------------------------------------------------------
# Fake audio I/O (same as real_e2e_test.py)
# ---------------------------------------------------------------------------

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
                # silence gap
                sil = np.zeros(self.CHUNK, dtype=np.float32)
                for _ in range(int(1.0 * 16000 / self.CHUNK)):
                    self._queue.put_nowait(sil)
                    self.ring.write(sil)
                    await asyncio.sleep(0.005)
        except asyncio.CancelledError:
            raise

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


# ---------------------------------------------------------------------------
# 10 Conversations — each 2-4 turns
# ---------------------------------------------------------------------------

CONVERSATIONS = [
    {"name": "C01 English greeting",
     "turns": [("en_01_greeting", "en"), ("en_02_question", "en")]},

    {"name": "C02 English deep",
     "turns": [("en_03_complex", "en"), ("en_04_followup", "en")]},

    {"name": "C03 English quick",
     "turns": [("en_06_short_yes", "en"), ("en_07_short_ok", "en"), ("en_08_short_what", "en")]},

    {"name": "C04 Russian chat",
     "turns": [("ru_01_greeting", "ru"), ("ru_02_question", "ru"), ("ru_03_complex", "ru")]},

    {"name": "C05 Croatian chat",
     "turns": [("hr_01_greeting", "hr"), ("hr_02_question", "hr"), ("hr_03_complex", "hr")]},

    {"name": "C06 Language switch en→ru",
     "turns": [("en_01_greeting", "en"), ("ru_01_greeting", "ru"), ("en_09_short_sure", "en")]},

    {"name": "C07 Long Russian",
     "turns": [("ru_05_long", "ru"), ("ru_04_short", "ru")]},

    {"name": "C08 Topic switch",
     "turns": [("en_02_question", "en"), ("en_05_topic_switch", "en"), ("en_10_stop", "en")]},

    {"name": "C09 Barge-in scenario",
     "turns": [("en_03_complex", "en"), ("barge_01_interrupt", "en", 3.0)]},

    {"name": "C10 Full mixed stress",
     "turns": [("en_01_greeting", "en"), ("ru_02_question", "ru"),
               ("hr_02_question", "hr"), ("en_04_followup", "en")]},
]


# ---------------------------------------------------------------------------
# Result row
# ---------------------------------------------------------------------------

@dataclass
class Row:
    conv: str
    turn: int
    sample: str
    lang: str
    audio_s: float = 0
    asr_s: float = 0
    asr_text: str = ""
    ttft_s: float = 0
    llm_s: float = 0
    tokens: int = 0
    tps: float = 0
    chunks: int = 0
    tts_wait_s: float = 0
    total_s: float = 0
    rss: float = 0
    gpu: float = 0
    error: str = ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stderr)
    log = logging.getLogger("stress")
    t0 = time.monotonic()

    log.info("=" * 80)
    log.info("VOICE STRESS TEST — 10 conversations")
    log.info("=" * 80)

    # --- startup ---
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
    log.info("Mode: %s  startup: %.1fs", mode.value, time.monotonic() - t0)

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
        "You are a voice assistant. Your responses will be spoken aloud via text-to-speech. "
        "Always respond in plain conversational text. Never use markdown, asterisks, "
        "bullet points, numbered lists, headers, or any formatting symbols. "
        "Keep responses concise and natural for spoken conversation."
    )

    # --- load WAVs ---
    wavs = {}
    for f in sorted(os.listdir(SAMPLE_DIR)):
        if f.endswith(".wav"):
            sid = f.replace(".wav", "")
            wavs[sid] = load_wav_16k(os.path.join(SAMPLE_DIR, f))
    log.info("Loaded %d WAV samples", len(wavs))

    # --- intercept orchestrator timing ---
    timing = {}  # filled per turn

    rows: list[Row] = []
    startup_s = time.monotonic() - t0

    # --- run conversations ---
    for ci, conv in enumerate(CONVERSATIONS):
        log.info("")
        log.info("━" * 80)
        log.info("CONVERSATION %d/10: %s", ci + 1, conv["name"])
        log.info("━" * 80)

        # Fresh orchestrator per conversation (clean dialogue)
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

        for ti, turn_def in enumerate(conv["turns"]):
            sample_id = turn_def[0]
            exp_lang = turn_def[1]
            barge_delay = turn_def[2] if len(turn_def) > 2 else None

            audio = wavs[sample_id]
            row = Row(conv=conv["name"], turn=ti + 1, sample=sample_id,
                      lang=exp_lang, audio_s=len(audio) / 16000)

            log.info("  Turn %d — %s [%s] (%.1fs)", ti + 1, sample_id, exp_lang, row.audio_s)

            turn_t0 = time.monotonic()
            dlg_before = len(orch.dialogue)

            if barge_delay:
                fake_audio.queue_utterance(audio)
                await asyncio.sleep(barge_delay)
            else:
                fake_audio.queue_utterance(audio)

            # Wait for turn to complete
            try:
                deadline = time.monotonic() + 120
                saw_proc = False
                while time.monotonic() < deadline:
                    st = orch.state
                    if st in (PipelineState.PROCESSING, PipelineState.SPEAKING):
                        saw_proc = True
                    elif st == PipelineState.IDLE and saw_proc:
                        await asyncio.sleep(0.3)
                        if orch.state == PipelineState.IDLE:
                            break
                    await asyncio.sleep(0.1)
                else:
                    row.error = "TIMEOUT"
            except Exception as e:
                row.error = str(e)

            row.total_s = time.monotonic() - turn_t0
            row.rss = get_rss_mb()
            row.gpu = get_gpu_mb()

            # Extract timing from new dialogue entries
            new_turns = orch.dialogue[dlg_before:]
            for dt in new_turns:
                if dt.role == "user" and dt.source == "voice":
                    row.asr_text = dt.content[:50]

            log.info("    done %.1fs | ASR='%s'", row.total_s, row.asr_text[:40])
            rows.append(row)
            await asyncio.sleep(0.5)

        # Stop orchestrator for this conversation
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

    # --- Parse timing from stderr logs ---
    # We need to parse the log output. Since logging goes to stderr and we're
    # teeing, we'll parse timing from the orchestrator's own log messages
    # that were already printed. For now, extract from the rows + log data.

    total_time = time.monotonic() - t0

    # --- Shutdown ---
    await llm.shutdown()
    await pm.shutdown()
    vram.shutdown()
    await emitter.stop()
    devnull.close()

    # --- PRINT TABLE ---
    # Print to stdout so it's visible even with log noise
    print("\n" + "=" * 130, file=sys.stderr)
    print("VOICE STRESS TEST — TIMESTAMPS TABLE", file=sys.stderr)
    print("=" * 130, file=sys.stderr)
    print(f"{'#':>3} {'Conv':>25} {'T':>2} {'Sample':>22} {'Lng':>3} "
          f"{'Audio':>6} {'Total':>7} {'RSS':>6} {'GPU':>7}  ASR Text", file=sys.stderr)
    print("-" * 130, file=sys.stderr)

    for i, r in enumerate(rows, 1):
        err = f" [{r.error}]" if r.error else ""
        print(f"{i:3d} {r.conv:>25} {r.turn:2d} {r.sample:>22} {r.lang:>3} "
              f"{r.audio_s:5.1f}s {r.total_s:6.1f}s {r.rss:5.0f}M {r.gpu:6.0f}M  "
              f"{r.asr_text[:40]}{err}", file=sys.stderr)

    print("-" * 130, file=sys.stderr)

    n = len(rows)
    print(f"{'':>3} {'AVERAGE':>25} {'':>2} {'':>22} {'':>3} "
          f"{sum(r.audio_s for r in rows)/n:5.1f}s {sum(r.total_s for r in rows)/n:6.1f}s "
          f"{rows[-1].rss:5.0f}M {rows[-1].gpu:6.0f}M", file=sys.stderr)

    errors = [r for r in rows if r.error]
    print(f"\nTotal: {total_time:.0f}s | Startup: {startup_s:.0f}s | "
          f"Turns: {n} | Errors: {len(errors)} | "
          f"RSS: {rows[0].rss:.0f}→{rows[-1].rss:.0f}M (+{rows[-1].rss - rows[0].rss:.0f}M) | "
          f"GPU: {rows[-1].gpu:.0f}M", file=sys.stderr)
    print("=" * 130, file=sys.stderr)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
