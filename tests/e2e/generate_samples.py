"""Generate real voice samples for E2E testing using edge-tts.

Produces 16kHz mono WAV files suitable for feeding into the VAD → ASR pipeline.

Usage:
    cd /home/winers/voice-assistant
    .venv/bin/python tests/e2e/generate_samples.py
"""
import asyncio
import io
import os
import struct
import sys

import edge_tts
import numpy as np

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "samples")

# Voice selections — natural sounding, one per language
VOICES = {
    "en": "en-US-GuyNeural",
    "ru": "ru-RU-DmitryNeural",
    "hr": "hr-HR-SreckoNeural",
}

# 20 samples: varied languages, lengths, complexity, test scenarios
SAMPLES = [
    # --- Normal conversation (en) ---
    {"id": "en_01_greeting",     "lang": "en", "text": "Hello, how are you today?"},
    {"id": "en_02_question",     "lang": "en", "text": "Tell me something interesting about electric cars."},
    {"id": "en_03_complex",      "lang": "en", "text": "I've been thinking about consciousness lately. Do you think machines can truly understand, or is it just pattern matching?"},
    {"id": "en_04_followup",     "lang": "en", "text": "That's fascinating. Can you elaborate on the philosophical implications?"},
    {"id": "en_05_topic_switch", "lang": "en", "text": "Let's talk about something different. What do you know about deep sea creatures?"},

    # --- Short utterances for rapid-fire / barge-in ---
    {"id": "en_06_short_yes",    "lang": "en", "text": "Yes."},
    {"id": "en_07_short_ok",     "lang": "en", "text": "OK."},
    {"id": "en_08_short_what",   "lang": "en", "text": "What?"},
    {"id": "en_09_short_sure",   "lang": "en", "text": "Sure, that sounds great!"},
    {"id": "en_10_stop",         "lang": "en", "text": "Stop. I want to ask you something else."},

    # --- Russian ---
    {"id": "ru_01_greeting",     "lang": "ru", "text": "Привет, как у тебя дела сегодня?"},
    {"id": "ru_02_question",     "lang": "ru", "text": "Расскажи мне что-нибудь интересное про космос."},
    {"id": "ru_03_complex",      "lang": "ru", "text": "Что ты думаешь о будущем технологий и искусственного интеллекта?"},
    {"id": "ru_04_short",        "lang": "ru", "text": "Да, конечно."},
    {"id": "ru_05_long",         "lang": "ru", "text": "А теперь давай поговорим о классической русской литературе. Кто твой любимый автор и какое произведение нравится больше всего?"},

    # --- Croatian ---
    {"id": "hr_01_greeting",     "lang": "hr", "text": "Bok, kako si danas?"},
    {"id": "hr_02_question",     "lang": "hr", "text": "Reci mi nešto zanimljivo o Hrvatskoj."},
    {"id": "hr_03_complex",      "lang": "hr", "text": "Što misliš o umjetnoj inteligenciji i njenom utjecaju na društvo?"},

    # --- Barge-in specific: meant to be played OVER assistant speech ---
    {"id": "barge_01_interrupt",  "lang": "en", "text": "Wait, wait, stop. I have a question."},
    {"id": "barge_02_redirect",   "lang": "en", "text": "Actually, never mind. Tell me about the weather instead."},
]


def _compress_silences(wav_bytes: bytes, max_silence_ms: int = 200,
                       threshold: float = 0.01, sr: int = 16000) -> bytes:
    """Compress internal silences in a 16kHz mono 16-bit WAV.

    Scans for runs of near-silence (amplitude < threshold) longer than
    max_silence_ms and trims them down. This prevents VAD from splitting
    multi-sentence utterances at natural TTS pauses.
    """
    import wave, io

    # Parse WAV
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        params = wf.getparams()
        frames = wf.readframes(wf.getnframes())

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    max_silence_samples = int(sr * max_silence_ms / 1000)

    # Find silence runs
    is_silent = np.abs(audio) < threshold
    result = []
    i = 0
    while i < len(audio):
        if is_silent[i]:
            # Find end of silence run
            j = i
            while j < len(audio) and is_silent[j]:
                j += 1
            silence_len = j - i
            if silence_len > max_silence_samples:
                # Keep only max_silence_samples of silence
                result.append(audio[i:i + max_silence_samples])
            else:
                result.append(audio[i:j])
            i = j
        else:
            result.append(audio[i:i + 1])
            i += 1

    compressed = np.concatenate(result)
    compressed_int16 = (compressed * 32768).clip(-32768, 32767).astype(np.int16)

    # Write back as WAV
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setparams(params._replace(nframes=len(compressed_int16)))
        wf.writeframes(compressed_int16.tobytes())

    return out.getvalue()


async def generate_one(sample: dict) -> None:
    """Generate one WAV sample using edge-tts and convert to 16kHz mono."""
    voice = VOICES[sample["lang"]]
    out_path = os.path.join(SAMPLE_DIR, f"{sample['id']}.wav")

    if os.path.exists(out_path):
        print(f"  SKIP {sample['id']} (exists)")
        return

    communicate = edge_tts.Communicate(sample["text"], voice)

    # Collect MP3 bytes
    mp3_data = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_data.extend(chunk["data"])

    if not mp3_data:
        print(f"  FAIL {sample['id']} — no audio data")
        return

    # Convert MP3 → 16kHz mono WAV using ffmpeg
    import subprocess
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", "pipe:0", "-ar", "16000", "-ac", "1",
         "-f", "wav", "pipe:1"],
        input=bytes(mp3_data), capture_output=True, timeout=30,
    )
    if proc.returncode != 0:
        print(f"  FAIL {sample['id']} — ffmpeg error: {proc.stderr[-200:]}")
        return

    wav_data = proc.stdout

    # Compress internal silences so VAD doesn't split mid-sentence.
    # TTS-generated audio has long pauses (>500ms) between sentences that
    # trigger VAD speech_end. Real human speech doesn't pause that long
    # mid-utterance. Compress any silence >200ms down to 200ms.
    wav_data = _compress_silences(wav_data, max_silence_ms=200)

    with open(out_path, "wb") as f:
        f.write(wav_data)

    # Report duration
    # WAV header: 44 bytes, then 16-bit mono at 16kHz
    data_size = len(wav_data) - 44
    duration_s = data_size / (16000 * 2)  # 16-bit = 2 bytes/sample
    print(f"  OK   {sample['id']:25s}  {duration_s:5.1f}s  {sample['lang']}  \"{sample['text'][:50]}\"")


async def main():
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    print(f"Generating {len(SAMPLES)} voice samples into {SAMPLE_DIR}/\n")

    for sample in SAMPLES:
        await generate_one(sample)

    # List all generated files
    files = sorted(f for f in os.listdir(SAMPLE_DIR) if f.endswith(".wav"))
    total_size = sum(os.path.getsize(os.path.join(SAMPLE_DIR, f)) for f in files)
    print(f"\nDone: {len(files)} files, {total_size / 1024:.0f} KB total")


if __name__ == "__main__":
    asyncio.run(main())
