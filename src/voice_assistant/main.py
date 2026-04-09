"""Voice assistant entry point — stdin/stdout IPC, no server.

Startup sequence:
1. Load SOUL config
2. Initialize IPC (stdin reader + stdout emitter)
3. VRAM guard check
4. ProcessManager handles model startup sequencing
5. Enter command loop (stdin JSON commands)

All logs go to stderr. stdout is reserved for IPC JSON events.
"""
import asyncio
import logging
import sys

logger = logging.getLogger("voice_assistant")


def _list_pipewire_devices():
    """Query PipeWire via wpctl for sources (mics) and sinks (speakers)."""
    import re
    import subprocess

    result = subprocess.run(
        ["wpctl", "status"], capture_output=True, text=True, timeout=5
    )
    lines = result.stdout.splitlines()

    sources = []  # microphones
    sinks = []    # speakers
    default_source = -1
    default_sink = -1

    section = None
    for line in lines:
        # Check for section headers anywhere in the line
        if "Sinks:" in line and "Sink endpoints" not in line:
            section = "sinks"
            continue
        elif "Sources:" in line and "Source endpoints" not in line:
            section = "sources"
            continue
        elif "Sink endpoints:" in line or "Source endpoints:" in line:
            section = None
            continue
        elif "Streams:" in line or "Devices:" in line or "Clients:" in line:
            section = None
            continue
        elif "Video" in line:
            break

        if section:
            # Match device lines like:
            #  │  *   42. fifine Microphone Analog Stereo     [vol: 1.00]
            #  │      51. GA102 HD Audio Digital Stereo (HDMI) [vol: 0.72]
            m = re.search(r'(\*?)\s*(\d+)\.\s+(.+?)(?:\s+\[vol:.*\])?\s*$', line)
            if m:
                is_default = m.group(1) == '*'
                dev_id = int(m.group(2))
                name = m.group(3).strip()
                if not name:
                    continue
                entry = {"id": dev_id, "name": name}
                if section == "sources":
                    sources.append(entry)
                    if is_default:
                        default_source = dev_id
                elif section == "sinks":
                    sinks.append(entry)
                    if is_default:
                        default_sink = dev_id

    return sources, sinks, default_source, default_sink


async def _test_mic(emitter, device_id=None):
    """Record 3 seconds from default mic, emit live RMS levels."""
    import sounddevice as sd
    import numpy as np
    import threading
    import time

    try:
        sr = 16000
        duration = 3.0
        levels = []
        lock = threading.Lock()

        def callback(indata, frames, time_info, status):
            rms = float(np.sqrt(np.mean(indata ** 2)))
            level = min(1.0, rms * 10)
            with lock:
                levels.append(level)
            emitter.emit_signal("mic_test", status="level", level=level)

        emitter.emit_signal("mic_test", status="recording")

        def record():
            # Always use system default — PipeWire routes to selected device
            with sd.InputStream(samplerate=sr, channels=1, dtype='float32',
                                blocksize=int(sr * 0.1), callback=callback):
                time.sleep(duration)

        await asyncio.get_running_loop().run_in_executor(None, record)

        with lock:
            all_levels = list(levels)
        emitter.emit_signal("mic_test", status="done", levels=all_levels)
    except Exception as e:
        logger.error("Mic test failed: %s", e)
        emitter.emit_signal("mic_test", status="error", message=str(e))


async def _test_speaker(emitter, device_id=None):
    """Play a short test tone on system default speaker."""
    import sounddevice as sd
    import numpy as np

    try:
        sr = 44100
        duration = 1.0
        emitter.emit_signal("speaker_test", status="playing")

        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Pleasant two-tone chime
        tone = (0.3 * np.sin(2 * np.pi * 880 * t) +
                0.2 * np.sin(2 * np.pi * 1320 * t)).astype(np.float32)
        # Fade in/out
        fade = int(sr * 0.05)
        tone[:fade] *= np.linspace(0, 1, fade)
        tone[-fade:] *= np.linspace(1, 0, fade)

        # Use system default — PipeWire routes to selected device
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: sd.play(tone, samplerate=sr, blocking=True)
        )
        emitter.emit_signal("speaker_test", status="done")
    except Exception as e:
        logger.error("Speaker test failed: %s", e)
        emitter.emit_signal("speaker_test", status="error", message=str(e))


async def main() -> None:
    # Logging to stderr — stdout is IPC only
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    logger.info("voice-assistant v0.1.0 starting...")

    # Load SOUL config
    from voice_assistant.core.soul_loader import load_soul

    soul_path = sys.argv[1] if len(sys.argv) > 1 else "soul/default.soul.yaml"
    try:
        soul = load_soul(soul_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to load SOUL config: %s", e)
        sys.exit(1)
    logger.info("Loaded SOUL: %s (lang=%s, model=%s)",
                soul.name, soul.default_language, soul.llm_model)

    # Initialize IPC
    from voice_assistant.core.ipc import StdinReader, StdoutEmitter

    emitter = StdoutEmitter()
    await emitter.start()
    reader = StdinReader()

    # VRAM guard
    from voice_assistant.core.vram_guard import VRAMGuard

    vram = VRAMGuard()
    try:
        state = vram.check()
        logger.info("VRAM: %.1f GB free / %.1f GB total",
                     state.free_bytes / 1024**3, state.total_bytes / 1024**3)
        emitter.emit_signal("vram_usage",
                            free_gb=round(state.free_bytes / 1024**3, 1),
                            total_gb=round(state.total_bytes / 1024**3, 1))
    except RuntimeError as e:
        logger.error("VRAM guard init failed: %s", e)
        emitter.emit_error("vram", str(e))

    # Process manager handles startup sequencing
    from voice_assistant.process.manager import ProcessManager

    pm = ProcessManager(soul, vram, emitter)
    mode = await pm.startup()
    logger.info("Pipeline mode: %s", mode.value)

    # Wire up the orchestrator with all loaded components
    orch = None
    if pm._llm_process and pm._tts and pm._asr and pm._vad:
        try:
            from voice_assistant.core.orchestrator import Orchestrator
            from voice_assistant.core.audio_input import AudioInput
            from voice_assistant.core.audio_output import Playlist, PlaybackManager, FillerCache
            from voice_assistant.adapters.llamacpp_llm import LlamaCppLLM

            audio_input = AudioInput()
            playlist = Playlist()
            playback = PlaybackManager(playlist)
            filler_cache = FillerCache()
            llm = LlamaCppLLM(f"http://127.0.0.1:{pm._llm_process.port}")

            orch = Orchestrator(
                audio_input=audio_input,
                vad=pm._vad,
                asr=pm._asr,
                llm=llm,
                tts=pm._tts,
                playlist=playlist,
                playback=playback,
                filler_cache=filler_cache,
            )

            # Wire orchestrator state changes to IPC
            def on_state_change(old_state, new_state):
                emitter.emit_state_change(new_state.value)
            orch._state._on_change = on_state_change

            # Wire LLM token streaming to IPC
            original_llm_stream = llm.stream
            async def stream_with_ipc(messages, sampling=None):
                async for token in original_llm_stream(messages, sampling):
                    emitter.emit_token(token)
                    yield token
            llm.stream = stream_with_ipc

            # Wire transcript events — patch _process_voice_turn and handle_text_input
            original_handle_text = orch.handle_text_input
            async def handle_text_with_ipc(text):
                emitter.emit({"event": "transcript", "role": "user", "text": text, "source": "text"})
                await original_handle_text(text)
                if orch._dialogue and orch._dialogue[-1].role == "assistant":
                    t = orch._dialogue[-1]
                    emitter.emit({"event": "transcript", "role": "assistant",
                                  "text": t.content, "source": t.source,
                                  "interrupted": t.interrupted, "spoken_text": t.partial})
            orch.handle_text_input = handle_text_with_ipc

            original_voice_turn = orch._process_voice_turn
            async def voice_turn_with_ipc(audio):
                await original_voice_turn(audio)
                # Emit both user transcript and assistant response
                for t in orch._dialogue[-2:]:
                    if t.role == "user":
                        emitter.emit({"event": "transcript", "role": "user",
                                      "text": t.content, "source": t.source})
                    elif t.role == "assistant":
                        emitter.emit({"event": "transcript", "role": "assistant",
                                      "text": t.content, "source": t.source,
                                      "interrupted": t.interrupted, "spoken_text": t.partial})
            orch._process_voice_turn = voice_turn_with_ipc

            await orch.start()
            logger.info("Orchestrator started — full pipeline active")
        except Exception as e:
            logger.error("Failed to start orchestrator: %s", e)
            orch = None
    else:
        logger.warning("Not all components available — orchestrator not started")

    # Command handler
    loop = asyncio.get_event_loop()

    def handle_command(cmd: dict) -> None:
        cmd_type = cmd.get("cmd")
        logger.info("IPC command: %s", cmd_type)

        if cmd_type == "text_input":
            text = cmd.get("text") or cmd.get("content", "")
            if text and orch:
                logger.info("Text input: %s", text[:80])
                asyncio.ensure_future(orch.handle_text_input(text))
            elif text:
                logger.warning("Text input but orchestrator not running")

        elif cmd_type == "set_mode":
            new_mode = cmd.get("mode", "")
            logger.info("Mode change request: %s", new_mode)

        elif cmd_type == "reload_soul":
            path = cmd.get("path", soul_path)
            logger.info("SOUL reload request: %s", path)

        elif cmd_type == "mute_toggle":
            logger.info("Mute toggle")
            if orch and orch._audio:
                # Toggle audio input mute
                orch._audio._muted = not getattr(orch._audio, '_muted', False)

        elif cmd_type == "new_conversation":
            logger.info("New conversation")
            if orch:
                orch._dialogue.clear()

        elif cmd_type == "list_audio_devices":
            logger.info("Audio device list requested")
            try:
                inputs, outputs, def_in, def_out = _list_pipewire_devices()
                emitter.emit_signal("audio_devices",
                                    inputs=inputs, outputs=outputs,
                                    default_input=def_in,
                                    default_output=def_out)
            except Exception as e:
                logger.error("Failed to list audio devices: %s", e)
                emitter.emit_signal("audio_devices", inputs=[], outputs=[],
                                    default_input=-1, default_output=-1)

        elif cmd_type == "set_audio_device":
            dev_type = cmd.get("type", "")
            dev_id = cmd.get("device_id", -1)
            logger.info("Set audio device: %s = %s", dev_type, dev_id)
            # Will be wired to audio_input/audio_output when pipeline is active

        elif cmd_type == "test_mic":
            dev_id = cmd.get("device_id", None)
            logger.info("Mic test requested on device %s", dev_id)
            asyncio.ensure_future(_test_mic(emitter, dev_id))

        elif cmd_type == "test_speaker":
            dev_id = cmd.get("device_id", None)
            logger.info("Speaker test requested on device %s", dev_id)
            asyncio.ensure_future(_test_speaker(emitter, dev_id))

        elif cmd_type == "get_status":
            logger.info("Status request — re-emitting all component states")
            pm.emit_all_status()

        elif cmd_type == "start_component":
            component = cmd.get("component", "")
            logger.info("Start component: %s", component)
            asyncio.ensure_future(pm.start_component(component))

        elif cmd_type == "stop_component":
            component = cmd.get("component", "")
            logger.info("Stop component: %s", component)
            asyncio.ensure_future(pm.stop_component(component))

        elif cmd_type == "shutdown":
            logger.info("Shutdown requested via IPC")
            raise KeyboardInterrupt

        else:
            logger.warning("Unknown command: %s", cmd_type)

    reader.on_command(handle_command)
    await reader.start()

    emitter.emit_state_change("IDLE")
    logger.info("Ready — waiting for commands on stdin")

    # Keep running until stdin closes or shutdown requested
    try:
        await asyncio.Event().wait()
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        logger.info("Shutting down...")
        if orch:
            await orch.stop()
        await pm.shutdown()
        vram.shutdown()
        await emitter.stop()
        await reader.stop()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
