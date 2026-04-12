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


async def _tts_test(emitter, tts, text, lang=None):
    """Synthesize text directly via TTS and play through speakers (no LLM)."""
    import time as _time
    try:
        voice_params = {"language": lang} if lang else {}
        emitter.emit_signal("synth_start", text=text, lang=lang)
        start = _time.monotonic()
        audio = await tts.synthesize(text, voice_params)
        elapsed_ms = int((_time.monotonic() - start) * 1000)
        duration_s = len(audio) / 24000
        rtf = (elapsed_ms / 1000) / duration_s if duration_s > 0 else 0

        # Play directly via sounddevice
        import sounddevice as sd
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: sd.play(audio, samplerate=24000, blocking=True)
        )
        emitter.emit_signal("synth_end", text=text, lang=lang, rtf=round(rtf, 3),
                            duration_ms=elapsed_ms)
    except Exception as e:
        logger.error("TTS test failed: %s", e)
        emitter.emit_signal("synth_end", text=text, lang=lang, rtf=0, duration_ms=0,
                            error=str(e))


async def _save_settings_yaml(settings):
    """Save voice/language config back to settings.yaml."""
    try:
        import yaml
        from pathlib import Path
        p = Path(settings.path)
        if p.exists():
            data = yaml.safe_load(p.read_text())
        else:
            data = {}
        if "voice" not in data:
            data["voice"] = {}
        data["voice"]["languages"] = [
            {"id": vl.id, "reference_audio": vl.reference_audio}
            for vl in settings.voice_languages
        ]
        data["language"] = {"default": settings.default_language}
        p.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))
        logger.info("Settings saved to %s", p)
    except Exception as e:
        logger.error("Failed to save settings: %s", e)


async def _clone_voice_for_lang(pm, emitter, settings, lang, reference_audio):
    """Clone voice for a single language using the already-loaded TTS model."""
    try:
        from voice_assistant.core.voice_clone import get_profile_path
        import torch

        if not pm._tts or pm._tts._model is None:
            raise RuntimeError("TTS model not loaded")

        emitter.emit_signal("voice_clone_progress", lang=lang, status="Cloning...")

        prompt = await pm._tts.clone_voice(lang, reference_audio)

        profile_path = get_profile_path(f"voice_{lang}")
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "ref_audio_tokens": prompt.ref_audio_tokens,
                "ref_text": prompt.ref_text,
                "ref_rms": prompt.ref_rms,
            },
            profile_path,
        )

        for vl in settings.voice_languages:
            if vl.id == lang:
                vl.reference_audio = reference_audio
                break

        emitter.emit_signal("voice_clone_progress", lang=lang, status="complete")
        logger.info("Voice cloned for '%s', saved to %s", lang, profile_path)
    except Exception as e:
        logger.error("Voice cloning failed for '%s': %s", lang, e)
        emitter.emit_signal("voice_clone_progress", lang=lang, status="error",
                            message=str(e))


async def main() -> None:
    # Logging to stderr — stdout is IPC only
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    logger.info("voice-assistant v0.1.0 starting...")

    # Load SOUL personality (name, personality, backstory, mood)
    from voice_assistant.core.soul_loader import load_soul

    soul_path = sys.argv[1] if len(sys.argv) > 1 else "soul/default.soul.yaml"
    try:
        soul = load_soul(soul_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to load SOUL config: %s", e)
        sys.exit(1)
    logger.info("Loaded SOUL: %s", soul.name)

    # Load infrastructure settings (LLM, voice, language, memory)
    from voice_assistant.core.settings_loader import load_settings

    settings_path = sys.argv[2] if len(sys.argv) > 2 else "config/settings.yaml"
    settings = load_settings(settings_path)
    logger.info("Loaded settings: model=%s, lang=%s, voices=%s",
                settings.llm_model, settings.default_language,
                [vl.id for vl in settings.voice_languages])

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

    pm = ProcessManager(soul, vram, emitter, settings=settings)
    mode = await pm.startup()
    logger.info("Pipeline mode: %s", mode.value)

    # Wire up the orchestrator with all loaded components
    orch = None
    llm = None
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

            # Build system prompt from SOUL personality
            persona = soul.persona_card()
            system_prompt = (
                f"{persona}\n\n"
                "You are a voice assistant. Your responses will be spoken aloud via text-to-speech. "
                "Always respond in plain conversational text. Never use markdown, asterisks, "
                "bullet points, numbered lists, headers, or any formatting symbols. "
                "Keep responses concise and natural for spoken conversation."
            )

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
                system_prompt=system_prompt,
                default_language=settings.default_language,
                supported_languages=supported,
            )

            # Wire orchestrator state changes to IPC
            def on_state_change(old_state, new_state):
                emitter.emit_state_change(new_state.value)
            orch._state._on_change = on_state_change

            # Wire chunk-by-chunk transcript to IPC (spoken text appears progressively)
            def on_chunk_synthesized(text):
                if not orch._generation_cancelled:
                    emitter.emit({"event": "chunk_spoken", "text": text})
            orch._on_chunk_synthesized = on_chunk_synthesized

            # Wire LLM token streaming to IPC
            original_llm_stream = llm.stream
            async def stream_with_ipc(messages, sampling=None, thinking=False):
                async for token in original_llm_stream(messages, sampling, thinking=thinking):
                    if orch._generation_cancelled:
                        break  # stop emitting tokens after barge-in
                    emitter.emit_token(token)
                    yield token
            llm.stream = stream_with_ipc

            # Wire transcript events — patch _process_turn (called by both text and voice paths)
            # Text input: frontend already shows user turn locally, so only emit for voice
            # Assistant transcript: always emit after turn completes
            original_process_turn = orch._process_turn
            async def process_turn_with_ipc(user_text, source="text"):
                # For voice input, emit user transcript (text input is shown locally by frontend)
                if source == "voice":
                    emitter.emit({"event": "transcript", "role": "user",
                                  "text": user_text, "source": source})
                await original_process_turn(user_text, source)
                # Emit assistant transcript after turn completes (LLM + TTS + playback done)
                if orch._dialogue and orch._dialogue[-1].role == "assistant":
                    t = orch._dialogue[-1]
                    emitter.emit({"event": "transcript", "role": "assistant",
                                  "text": t.content, "source": t.source,
                                  "interrupted": t.interrupted, "spoken_text": t.partial})
            orch._process_turn = process_turn_with_ipc

            await orch.start()
            logger.info("Orchestrator started — full pipeline active")
        except Exception as e:
            logger.error("Failed to start orchestrator: %s", e)
            orch = None
    else:
        logger.warning("Not all components available — orchestrator not started")

    # Command handler
    loop = asyncio.get_event_loop()

    def _log_task_exception(task: asyncio.Task) -> None:
        """Log exceptions from fire-and-forget tasks."""
        if not task.cancelled() and task.exception():
            logger.exception("Unhandled error in async task", exc_info=task.exception())

    def handle_command(cmd: dict) -> None:
        cmd_type = cmd.get("cmd")
        logger.info("IPC command: %s", cmd_type)

        if cmd_type == "text_input":
            text = cmd.get("text") or cmd.get("content", "")
            if text and orch:
                logger.info("Text input: %s", text[:80])
                task = asyncio.ensure_future(orch.handle_text_input(text))
                task.add_done_callback(_log_task_exception)
            elif text:
                logger.warning("Text input but orchestrator not running")

        elif cmd_type == "set_mode":
            new_mode = cmd.get("mode", "")
            logger.info("Mode change request: %s", new_mode)

        elif cmd_type == "reload_soul":
            path = cmd.get("path", soul_path)
            content = cmd.get("content")
            logger.info("SOUL reload request: %s", path)
            try:
                from pathlib import Path as _Path
                # If UI sent edited content, save it to disk first
                if content:
                    _Path(path).write_text(content, encoding="utf-8")
                    logger.info("SOUL YAML saved to %s (%d chars)", path, len(content))

                from voice_assistant.core.soul_loader import load_soul as _reload_soul
                new_soul = _reload_soul(path)
                soul.name = new_soul.name
                soul.personality = new_soul.personality
                soul.backstory = new_soul.backstory
                soul.mood_default = new_soul.mood_default
                soul.mood_range = new_soul.mood_range
                logger.info("SOUL reloaded: %s", soul.name)
                emitter.emit_signal("soul_reloaded", name=soul.name)
                # Re-emit YAML content so UI updates
                saved = _Path(path).read_text(encoding="utf-8")
                emitter.emit({"event": "soul_yaml", "content": saved})
            except Exception as e:
                logger.error("Failed to reload SOUL: %s", e)
                emitter.emit_signal("soul_reload_error", message=str(e))

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

        elif cmd_type == "tts_test":
            text = cmd.get("text", "")
            lang = cmd.get("lang")
            if text and pm._tts:
                logger.info("TTS test [%s]: %s", lang, text[:80])
                asyncio.ensure_future(_tts_test(emitter, pm._tts, text, lang))
            elif not pm._tts:
                emitter.emit_signal("synth_start", text=text, lang=lang)
                emitter.emit_signal("synth_end", text=text, lang=lang, rtf=0,
                                    duration_ms=0, error="TTS not loaded")

        elif cmd_type == "get_tts_settings":
            logger.info("TTS settings requested")
            langs = []
            for vl in settings.voice_languages:
                from voice_assistant.core.voice_clone import get_profile_path
                profile = get_profile_path(f"voice_{vl.id}")
                langs.append({
                    "id": vl.id,
                    "reference_audio": vl.reference_audio,
                    "has_profile": profile.exists(),
                })
            loaded = pm._tts.available_languages if pm._tts else []
            emitter.emit_signal("tts_settings",
                                languages=langs,
                                loaded_languages=loaded,
                                default_language=settings.default_language)

        elif cmd_type == "update_tts_languages":
            lang_ids = cmd.get("languages", [])[:3]  # max 3
            logger.info("Update TTS languages: %s", lang_ids)
            from voice_assistant.core.settings_loader import VoiceLanguageConfig
            from voice_assistant.core.voice_clone import get_profile_path
            # Preserve existing reference_audio for languages that stay
            existing = {vl.id: vl.reference_audio for vl in settings.voice_languages}
            old_ids = set(existing.keys())
            new_ids = set(lang_ids)

            settings.voice_languages = [
                VoiceLanguageConfig(id=lid, reference_audio=existing.get(lid))
                for lid in lang_ids
            ]

            # Sync TTS voice_prompts: unload removed, load new if cached
            if pm._tts:
                for removed in old_ids - new_ids:
                    pm._tts.unload_language(removed)
                for added in new_ids - old_ids:
                    profile = get_profile_path(f"voice_{added}")
                    if profile.exists():
                        try:
                            pm._tts.load_voice_profile_sync(added, str(profile))
                        except Exception as e:
                            logger.warning("Failed to load profile for '%s': %s", added, e)

            # Ensure default language is still valid
            if settings.default_language not in new_ids and lang_ids:
                settings.default_language = lang_ids[0]
                if pm._tts:
                    pm._tts.set_language(settings.default_language)

            asyncio.ensure_future(_save_settings_yaml(settings))
            # Re-emit settings
            handle_command({"cmd": "get_tts_settings"})

        elif cmd_type == "set_default_language":
            lang = cmd.get("lang", "en")
            configured = {vl.id for vl in settings.voice_languages}
            if lang not in configured:
                logger.warning("Cannot set default to '%s' — not in configured languages %s", lang, configured)
                return
            logger.info("Set default language: %s", lang)
            settings.default_language = lang
            if pm._tts:
                pm._tts.set_language(lang)
            if orch:
                orch._current_language = lang
            asyncio.ensure_future(_save_settings_yaml(settings))

        elif cmd_type == "clone_voice_for_lang":
            lang = cmd.get("lang", "")
            ref_audio = cmd.get("reference_audio", "")
            if lang and ref_audio:
                import os
                if not os.path.isfile(ref_audio):
                    emitter.emit_signal("voice_clone_progress", lang=lang, status="error",
                                        message=f"File not found: {ref_audio}")
                    return
                logger.info("Clone voice for '%s': %s", lang, ref_audio)

                async def _clone_then_save():
                    await _clone_voice_for_lang(pm, emitter, settings, lang, ref_audio)
                    await _save_settings_yaml(settings)

                asyncio.ensure_future(_clone_then_save())

        elif cmd_type == "get_soul_yaml":
            logger.info("SOUL YAML requested")
            try:
                from pathlib import Path
                content = Path(soul_path).read_text(encoding="utf-8")
                emitter.emit({"event": "soul_yaml", "content": content})
            except Exception as e:
                logger.error("Failed to read SOUL YAML: %s", e)
                emitter.emit({"event": "soul_yaml", "content": "", "error": str(e)})

        elif cmd_type == "validate_soul":
            yaml_content = cmd.get("content", "")
            logger.info("SOUL validation requested (%d chars)", len(yaml_content))
            try:
                import yaml
                data = yaml.safe_load(yaml_content)
                errors = []
                if not isinstance(data, dict):
                    errors.append("YAML must be a mapping")
                else:
                    if not data.get("name"):
                        errors.append("Missing required field: name")
                    if not data.get("personality"):
                        errors.append("Missing required field: personality")
                    if "llm" in data and not data["llm"].get("model"):
                        errors.append("Missing llm.model")
                emitter.emit({"event": "soul_validation",
                              "valid": len(errors) == 0, "errors": errors})
            except yaml.YAMLError as e:
                emitter.emit({"event": "soul_validation",
                              "valid": False, "errors": [f"YAML parse error: {e}"]})
            except Exception as e:
                emitter.emit({"event": "soul_validation",
                              "valid": False, "errors": [str(e)]})

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
        if llm:
            await llm.shutdown()
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
