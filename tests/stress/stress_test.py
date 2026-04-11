"""Stress test — sends many text turns, tracks memory, logs timestamps.

Usage:
    cd /home/winers/voice-assistant
    .venv/bin/python tests/stress/stress_test.py 2>/tmp/va_stress.log

Outputs a summary table to stderr at the end.
"""
import asyncio
import json
import os
import sys
import time
import resource

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def get_memory_mb():
    """Get current RSS in MB (no external deps)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB → MB
    except Exception:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def get_gpu_memory_mb():
    """Get GPU memory used by this process (approximate via pynvml)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024 / 1024
    except Exception:
        return 0


# --- Test messages: multilingual, increasing complexity ---
TURNS = [
    # English
    ("en", "Hello, how are you today?"),
    ("en", "Tell me something interesting about electric cars."),
    ("en", "What do you think about artificial intelligence and its impact on society?"),
    ("en", "Can you explain quantum computing in simple terms?"),
    ("en", "What's the most beautiful place you can imagine?"),
    # Russian
    ("ru", "Привет, как у тебя дела сегодня?"),
    ("ru", "Расскажи мне что-нибудь интересное про космос."),
    ("ru", "Что ты думаешь о будущем технологий?"),
    ("ru", "Какая твоя любимая книга и почему?"),
    ("ru", "Опиши идеальный день."),
    # Croatian
    ("hr", "Bok, kako si danas?"),
    ("hr", "Reci mi nešto zanimljivo o Hrvatskoj."),
    ("hr", "Što misliš o umjetnoj inteligenciji?"),
    # Back to English — long complex
    ("en", "I've been thinking a lot about consciousness lately. Do you think machines can truly be conscious, or is it just a simulation of consciousness? Where do you draw the line between genuine understanding and pattern matching?"),
    ("en", "That's fascinating. Can you elaborate on the philosophical implications of that view?"),
    # Rapid short turns (stress barge-in / quick turns)
    ("en", "Yeah."),
    ("en", "OK."),
    ("en", "Sure."),
    ("ru", "Да."),
    ("ru", "Конечно."),
    ("hr", "Da."),
    ("en", "Interesting."),
    # Long turns again
    ("en", "Let's talk about something completely different. What do you know about marine biology and deep sea creatures? I find the bioluminescent organisms particularly fascinating."),
    ("ru", "А теперь давай поговорим на русском. Расскажи мне о классической русской литературе. Кто твой любимый автор и какое произведение тебе нравится больше всего?"),
    ("en", "One final question — if you could travel anywhere in the world, where would you go and why?"),
]


async def run_stress_test():
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger("stress_test")

    logger.info("=" * 70)
    logger.info("STRESS TEST START — %d turns", len(TURNS))
    logger.info("=" * 70)

    # Record initial memory
    mem_samples = []
    gpu_samples = []
    t0 = time.monotonic()

    def sample_memory(label=""):
        rss = get_memory_mb()
        gpu = get_gpu_memory_mb()
        elapsed = time.monotonic() - t0
        mem_samples.append((elapsed, rss, label))
        gpu_samples.append((elapsed, gpu, label))
        logger.info("[%.1fs] MEM: RSS=%.0f MB  GPU=%.0f MB  (%s)",
                    elapsed, rss, gpu, label)
        return rss, gpu

    sample_memory("before_imports")

    # Import and initialize components
    from voice_assistant.core.soul_loader import load_soul
    from voice_assistant.core.settings_loader import load_settings
    sample_memory("after_soul_import")

    soul = load_soul("soul/default.soul.yaml")
    settings = load_settings("config/settings.yaml")
    sample_memory("after_soul_load")

    from voice_assistant.core.vram_guard import VRAMGuard
    vram = VRAMGuard()
    sample_memory("after_vram")

    from voice_assistant.core.ipc import StdoutEmitter
    # Use a dummy emitter that writes to /dev/null
    devnull = open(os.devnull, "w")
    emitter = StdoutEmitter(output_stream=devnull)
    await emitter.start()
    sample_memory("after_ipc")

    from voice_assistant.process.manager import ProcessManager
    pm = ProcessManager(soul, vram, emitter, settings=settings)
    mode = await pm.startup()
    logger.info("Pipeline mode: %s", mode.value)
    sample_memory("after_startup_all_models")

    # Build orchestrator
    if not (pm._llm_process and pm._tts and pm._asr and pm._vad):
        logger.error("Not all components loaded — aborting stress test")
        return

    from voice_assistant.core.orchestrator import Orchestrator
    from voice_assistant.core.audio_input import AudioInput
    from voice_assistant.core.audio_output import Playlist, PlaybackManager, FillerCache
    from voice_assistant.adapters.llamacpp_llm import LlamaCppLLM

    audio_input = AudioInput()
    playlist = Playlist()
    playback = PlaybackManager(playlist)
    filler_cache = FillerCache()
    llm = LlamaCppLLM(f"http://127.0.0.1:{pm._llm_process.port}")

    supported = [vl.id for vl in settings.voice_languages] or ["en"]
    orch = Orchestrator(
        audio_input=audio_input,
        vad=pm._vad,
        asr=pm._asr,
        llm=llm,
        tts=pm._tts,
        playlist=playlist,
        playback=playback,
        filler_cache=filler_cache,
        system_prompt="You are Peka, a voice assistant. Respond concisely.",
        default_language=settings.default_language,
        supported_languages=supported,
    )
    sample_memory("after_orchestrator_init")

    # Don't start the full orchestrator (no mic needed) — just call handle_text_input directly
    # But we need playback started for the playlist
    await playback.start()
    from voice_assistant.core.state_machine import PipelineMode
    orch._state.set_mode(PipelineMode.FULL)

    sample_memory("before_turns")

    # --- Run turns ---
    turn_stats = []
    for i, (expected_lang, text) in enumerate(TURNS, 1):
        turn_start = time.monotonic()
        logger.info("")
        logger.info("─── TURN %d/%d [%s] ───  '%s'", i, len(TURNS), expected_lang, text[:60])

        try:
            await orch.handle_text_input(text)
            if orch._processing_task:
                await asyncio.wait_for(orch._processing_task, timeout=120.0)
        except asyncio.TimeoutError:
            logger.error("TURN %d TIMED OUT after 120s", i)
        except Exception as e:
            logger.error("TURN %d ERROR: %s", i, e)

        turn_elapsed = time.monotonic() - turn_start
        rss, gpu = sample_memory(f"turn_{i}_{expected_lang}")

        # Check language detection
        detected_lang = orch._current_language
        lang_ok = "✓" if detected_lang == expected_lang else f"✗ got={detected_lang}"

        # Get response length
        resp_len = 0
        if orch.dialogue and orch.dialogue[-1].role == "assistant":
            resp_len = len(orch.dialogue[-1].content)

        turn_stats.append({
            "turn": i,
            "lang": expected_lang,
            "lang_ok": lang_ok,
            "time_s": turn_elapsed,
            "rss_mb": rss,
            "gpu_mb": gpu,
            "resp_chars": resp_len,
            "text": text[:50],
        })

        logger.info("TURN %d done: %.1fs, lang=%s, resp=%d chars, RSS=%.0f MB",
                    i, turn_elapsed, lang_ok, resp_len, rss)

        # Brief pause between turns
        await asyncio.sleep(0.5)

    # Final memory sample
    sample_memory("after_all_turns")

    # --- Summary ---
    total_time = time.monotonic() - t0
    peak_rss = max(s[1] for s in mem_samples)
    peak_gpu = max(s[1] for s in gpu_samples)
    initial_rss = mem_samples[0][1]

    logger.info("")
    logger.info("=" * 90)
    logger.info("STRESS TEST SUMMARY")
    logger.info("=" * 90)
    logger.info("Total time:     %.1fs", total_time)
    logger.info("Turns:          %d", len(TURNS))
    logger.info("Initial RSS:    %.0f MB", initial_rss)
    logger.info("Peak RSS:       %.0f MB", peak_rss)
    logger.info("RSS growth:     %.0f MB", peak_rss - initial_rss)
    logger.info("Peak GPU:       %.0f MB", peak_gpu)
    logger.info("Dialogue turns: %d", len(orch.dialogue))
    logger.info("")
    logger.info("%-5s %-4s %-12s %8s %8s %8s %s",
                "Turn", "Lang", "LangDetect", "Time(s)", "RSS(MB)", "GPU(MB)", "Text")
    logger.info("-" * 90)
    for s in turn_stats:
        logger.info("%-5d %-4s %-12s %8.1f %8.0f %8.0f %s",
                    s["turn"], s["lang"], s["lang_ok"], s["time_s"],
                    s["rss_mb"], s["gpu_mb"], s["text"])
    logger.info("-" * 90)
    logger.info("PEAK RSS: %.0f MB  |  PEAK GPU: %.0f MB  |  TOTAL: %.1fs",
                peak_rss, peak_gpu, total_time)
    logger.info("=" * 90)

    # Memory timeline
    logger.info("")
    logger.info("MEMORY TIMELINE:")
    for elapsed, rss, label in mem_samples:
        bar = "█" * int(rss / 100)
        logger.info("  [%6.1fs] %6.0f MB %s  %s", elapsed, rss, bar, label)

    # Cleanup
    await playback.stop()
    await pm.shutdown()
    vram.shutdown()
    await emitter.stop()
    devnull.close()

    logger.info("Stress test complete.")


if __name__ == "__main__":
    try:
        asyncio.run(run_stress_test())
    except KeyboardInterrupt:
        pass
