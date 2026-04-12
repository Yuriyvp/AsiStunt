"""Process manager for llama.cpp server subprocess and full startup sequencing.

LlamaCppProcess: spawn, health check polling, graceful shutdown, crash detection.
ProcessManager: startup sequencing, error escalation, degraded mode transitions.

The llama.cpp server is a separate process that holds 13-14GB VRAM independently.
Process isolation means Python crashes don't lose the loaded model.
"""
import asyncio
import logging
import signal
import subprocess
import time
from pathlib import Path

import aiohttp

from voice_assistant.core.state_machine import PipelineMode

logger = logging.getLogger(__name__)

HEALTH_POLL_INTERVAL = 2.0
HEALTH_TIMEOUT = 60.0


class LlamaCppProcess:
    """Manages a llama.cpp server subprocess."""

    def __init__(
        self,
        server_binary: str = "llama-server",
        model_path: str = "models/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf",
        port: int = 8080,
        ctx_size: int = 8192,
        gpu_layers: int = 999,
        threads: int = 4,
        batch_size: int = 4096,
        ubatch_size: int = 2048,
        flash_attn: bool = True,
        cache_type_k: str = "q8_0",
        cache_type_v: str = "q8_0",
    ):
        self._binary = server_binary
        self._model = model_path
        self._port = port
        self._process: subprocess.Popen | None = None

        self._args = [
            server_binary,
            "--model", model_path,
            "--ctx-size", str(ctx_size),
            "--cache-type-k", cache_type_k,
            "--cache-type-v", cache_type_v,
            "--n-gpu-layers", str(gpu_layers),
            "--threads", str(threads),
            "--batch-size", str(batch_size),
            "--ubatch-size", str(ubatch_size),
            "--port", str(port),
            "--host", "127.0.0.1",
            "--parallel", "1",
            "--metrics",
            "--log-disable",
            "--mlock",
            "--seed", "-1",
        ]
        if flash_attn:
            self._args.extend(["--flash-attn", "on"])

    async def start(self) -> None:
        """Spawn llama.cpp server and wait for health check."""
        binary_path = Path(self._binary)
        if not binary_path.exists():
            raise FileNotFoundError(f"llama-server binary not found: {self._binary}")

        # Set LD_LIBRARY_PATH to include the binary's directory for shared libs
        import os
        env = os.environ.copy()
        bin_dir = str(binary_path.parent.resolve())
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{bin_dir}:{existing}" if existing else bin_dir

        logger.info("Starting llama.cpp server: %s", " ".join(self._args))
        self._process = subprocess.Popen(
            self._args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
        )
        logger.info("llama.cpp PID: %d, waiting for health...", self._process.pid)
        await self._wait_for_health()

    async def _wait_for_health(self) -> None:
        """Poll /health until OK or timeout."""
        url = f"http://127.0.0.1:{self._port}/health"
        deadline = asyncio.get_event_loop().time() + HEALTH_TIMEOUT

        async with aiohttp.ClientSession() as session:
            while asyncio.get_event_loop().time() < deadline:
                if self._process and self._process.poll() is not None:
                    stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                    raise RuntimeError(
                        f"llama.cpp server exited with code {self._process.returncode}: {stderr[:500]}"
                    )
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get("status") == "ok":
                                logger.info("llama.cpp server healthy")
                                return
                            logger.debug("Health status: %s (waiting for 'ok')", data.get("status"))
                except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                    pass
                await asyncio.sleep(HEALTH_POLL_INTERVAL)

        raise TimeoutError(f"llama.cpp server not healthy after {HEALTH_TIMEOUT}s")

    async def stop(self) -> None:
        """Gracefully stop the llama.cpp server."""
        if self._process is None:
            return
        logger.info("Stopping llama.cpp server (PID %d)", self._process.pid)
        self._process.send_signal(signal.SIGTERM)
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("llama.cpp didn't stop gracefully, killing")
            self._process.kill()
            self._process.wait()
        self._process = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def port(self) -> int:
        return self._port


# Stability window — reset failure counter after this many seconds of stable operation
STABILITY_WINDOW_S = 300  # 5 minutes


class ProcessManager:
    """Manages all AI model processes with startup sequencing and error escalation.

    Startup order (mandatory — GPU models load sequentially):
    1. Voice profile check — clone if needed (full GPU)
    2. llama.cpp server — wait for health OK
    3. OmniVoice inference mode — load into remaining VRAM
    4. CPU models (VAD, ASR) — parallel, no VRAM impact

    Error escalation per component:
    1st failure: silent restart
    2nd failure: restart with VRAM guard (reduced ctx, quantized KV)
    3+: stop retrying, emit error banner
    Counter resets after 5 minutes of stable operation.
    """

    def __init__(self, soul_config, vram_guard, ipc_emitter, settings=None):
        self._soul = soul_config
        self._settings = settings  # infrastructure settings (LLM, voice, etc.)
        self._vram = vram_guard
        self._ipc = ipc_emitter
        self._failure_counts: dict[str, int] = {}
        self._failure_times: dict[str, float] = {}
        self._mode = PipelineMode.DISABLED

        # Component references (set during startup)
        self._llm_process: LlamaCppProcess | None = None
        self._component_states: dict[str, str] = {
            "llm": "idle", "tts": "idle", "asr": "idle", "vad": "idle",
        }
        # Actual model instances
        self._tts = None
        self._asr = None
        self._vad = None

    @property
    def mode(self) -> PipelineMode:
        return self._mode

    @property
    def llm_process(self) -> LlamaCppProcess | None:
        return self._llm_process

    def _set_component_state(self, component: str, state: str) -> None:
        """Update internal state and emit signal."""
        self._component_states[component] = state
        self._ipc.emit_signal("process_state_change", component=component, state=state)

    def emit_all_status(self) -> None:
        """Re-emit current state for all components. Called when frontend connects."""
        for comp, state in self._component_states.items():
            self._ipc.emit_signal("process_state_change", component=comp, state=state)
        self._ipc.emit_signal("process_state_change",
                              component="system", state=self._mode.value)

    async def startup(self) -> PipelineMode:
        """Execute full startup sequence. Returns the resulting PipelineMode."""
        self._ipc.emit_state_change("DISABLED")
        self._ipc.emit_signal("process_state_change", component="system", state="starting")

        llm_ok = False
        tts_ok = False

        try:
            # Step 1: llama.cpp server
            self._set_component_state("llm", "starting")
            self._llm_process = LlamaCppProcess(
                server_binary=str(Path(self._settings.llm_model).parent.parent / "bin" / "llama-server"),
                model_path=self._settings.llm_model,
                port=self._settings.llm_port,
                ctx_size=self._settings.llm_ctx_size,
                gpu_layers=self._settings.llm_gpu_layers,
                threads=self._settings.llm_threads,
                batch_size=self._settings.llm_batch_size,
                flash_attn=self._settings.llm_flash_attn,
            )
            try:
                await self._llm_process.start()
                llm_ok = True
                self._set_component_state("llm", "ready")
            except Exception as e:
                logger.error("LLM startup failed: %s", e)
                self._ipc.emit_error("llm", f"Failed to start: {e}")
                self._set_component_state("llm", "error")
                self._llm_process = None

            # Step 3: TTS load (OmniVoice) + per-language voice profiles
            self._set_component_state("tts", "starting")
            try:
                from voice_assistant.adapters.omnivoice_tts import OmniVoiceTTS
                from voice_assistant.core.voice_clone import get_profile_path
                self._tts = OmniVoiceTTS()
                await self._tts.load()

                # Load cached voice profiles for each configured language
                for vl in self._settings.voice_languages:
                    profile_path = get_profile_path(f"voice_{vl.id}")
                    if profile_path.exists():
                        try:
                            self._tts.load_voice_profile_sync(vl.id, str(profile_path))
                            logger.info("Loaded cached voice profile for '%s'", vl.id)
                        except Exception as e:
                            logger.warning("Failed to load profile for '%s': %s", vl.id, e)

                self._tts.set_language(self._settings.default_language)
                tts_ok = True
                self._set_component_state("tts", "ready")
            except Exception as e:
                logger.error("TTS startup failed: %s", e)
                self._ipc.emit_error("tts", f"Failed to start: {e}")
                self._set_component_state("tts", "error")
                self._tts = None

            # Step 4: CPU models
            self._set_component_state("vad", "starting")
            try:
                from voice_assistant.models.silero_vad import SileroVAD
                self._vad = SileroVAD()
                self._set_component_state("vad", "ready")
            except Exception as e:
                logger.error("VAD startup failed: %s", e)
                self._set_component_state("vad", "error")
                self._vad = None

            self._set_component_state("asr", "starting")
            try:
                from voice_assistant.adapters.parakeet_asr import ParakeetASR
                supported = [vl.id for vl in self._settings.voice_languages] or ["en"]
                self._asr = ParakeetASR(supported_languages=supported)
                self._set_component_state("asr", "ready")
            except Exception as e:
                logger.error("ASR startup failed: %s", e)
                self._set_component_state("asr", "error")
                self._asr = None

        except Exception as e:
            logger.exception("Startup failed: %s", e)
            self._ipc.emit_error("system", f"Startup failed: {e}")

        # Warmup phase — prime all models before accepting user input
        await self._run_warmups()

        # Determine pipeline mode based on what started
        self._mode = self._determine_mode(llm_ok, tts_ok)
        logger.info("Startup complete: mode=%s (llm=%s, tts=%s)", self._mode.value, llm_ok, tts_ok)
        self._ipc.emit_signal("process_state_change", component="system", state=self._mode.value)
        return self._mode

    async def _run_warmups(self) -> None:
        """Run warmup inference on all loaded models. Failures are non-fatal."""
        self._ipc.emit_signal("process_state_change", component="system", state="warming")

        async def _warmup_component(name: str, coro_or_callable):
            self._set_component_state(name, "warming")
            t0 = time.monotonic()
            try:
                if asyncio.iscoroutinefunction(coro_or_callable):
                    await coro_or_callable()
                elif callable(coro_or_callable):
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, coro_or_callable)
                elapsed = time.monotonic() - t0
                logger.info("Warmup %s completed in %.2fs", name, elapsed)
            except Exception:
                elapsed = time.monotonic() - t0
                logger.warning("Warmup %s failed after %.2fs (non-fatal)", name, elapsed, exc_info=True)
            finally:
                self._set_component_state(name, "ready")

        # GPU warmups sequential, CPU warmups concurrent with GPU
        gpu_warmups = []
        cpu_warmups = []

        if self._llm_process and self._llm_process.is_running:
            from voice_assistant.adapters.llamacpp_llm import LlamaCppLLM
            llm_client = LlamaCppLLM(f"http://127.0.0.1:{self._llm_process.port}")

            async def _llm_warmup():
                try:
                    await llm_client.warmup()
                finally:
                    await llm_client.shutdown()

            gpu_warmups.append(("llm", _llm_warmup))
        elif self._llm_process:
            logger.warning("LLM warmup skipped: process not running (port conflict?)")

        if self._tts is not None:
            gpu_warmups.append(("tts", self._tts.warmup))

        if self._vad is not None:
            cpu_warmups.append(("vad", self._vad.warmup))

        if self._asr is not None:
            cpu_warmups.append(("asr", self._asr.warmup))

        async def _gpu_sequence():
            for name, fn in gpu_warmups:
                await _warmup_component(name, fn)

        tasks = [_gpu_sequence()]
        for name, fn in cpu_warmups:
            tasks.append(_warmup_component(name, fn))

        await asyncio.gather(*tasks)

    async def _warmup_single(self, component: str) -> None:
        """Warmup a single component after restart."""
        self._set_component_state(component, "warming")
        t0 = time.monotonic()
        try:
            if component == "llm" and self._llm_process and self._llm_process.is_running:
                from voice_assistant.adapters.llamacpp_llm import LlamaCppLLM
                llm = LlamaCppLLM(f"http://127.0.0.1:{self._llm_process.port}")
                try:
                    await llm.warmup()
                finally:
                    await llm.shutdown()
            elif component == "tts" and self._tts is not None:
                await self._tts.warmup()
            elif component == "asr" and self._asr is not None:
                await self._asr.warmup()
            elif component == "vad" and self._vad is not None:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._vad.warmup)
            logger.info("Warmup %s completed in %.2fs", component, time.monotonic() - t0)
        except Exception:
            logger.warning("Warmup %s failed (non-fatal)", component, exc_info=True)
        finally:
            self._set_component_state(component, "ready")

    def _determine_mode(self, llm_ok: bool, tts_ok: bool) -> PipelineMode:
        """Determine pipeline mode from component availability."""
        if llm_ok and tts_ok:
            return PipelineMode.FULL
        elif llm_ok and not tts_ok:
            return PipelineMode.TEXT_ONLY
        elif not llm_ok and tts_ok:
            return PipelineMode.TRANSCRIBE
        else:
            return PipelineMode.DISABLED

    def record_failure(self, component: str) -> int:
        """Record a failure for a component. Returns consecutive failure count."""
        now = time.monotonic()
        last_time = self._failure_times.get(component, 0)

        # Reset counter after stability window
        if now - last_time > STABILITY_WINDOW_S:
            self._failure_counts[component] = 0

        self._failure_counts[component] = self._failure_counts.get(component, 0) + 1
        self._failure_times[component] = now
        count = self._failure_counts[component]

        logger.warning("Component '%s' failure #%d", component, count)
        return count

    async def handle_failure(self, component: str) -> PipelineMode:
        """Handle a component failure with escalating response.

        Returns the new PipelineMode after handling.
        """
        count = self.record_failure(component)
        self._ipc.emit_error(component, f"Failure #{count}", consecutive=count)

        if count == 1:
            # Silent restart
            logger.info("Attempting silent restart of %s", component)
            self._ipc.emit_signal("process_state_change",
                                  component=component, state="restarting")
            if component == "llm" and self._llm_process:
                try:
                    await self._llm_process.stop()
                    await self._llm_process.start()
                    self._ipc.emit_signal("process_state_change",
                                          component=component, state="ready")
                    return self._mode  # mode unchanged
                except Exception as e:
                    logger.error("Silent restart failed for %s: %s", component, e)

        elif count == 2:
            # Restart with VRAM-guarded settings
            logger.info("Attempting VRAM-guarded restart of %s", component)
            self._ipc.emit_signal("process_state_change",
                                  component=component, state="restarting_optimized")
            if component == "llm":
                vram_state = self._vram.check()
                if vram_state.under_pressure:
                    # Recreate with reduced settings
                    if self._llm_process:
                        await self._llm_process.stop()
                    self._llm_process = LlamaCppProcess(
                        server_binary=self._llm_process._binary if self._llm_process else "llama-server",
                        model_path=self._settings.llm_model,
                        port=self._settings.llm_port,
                        ctx_size=6144,  # reduced from 8192
                        gpu_layers=self._settings.llm_gpu_layers,
                        threads=self._settings.llm_threads,
                        batch_size=self._settings.llm_batch_size,
                        flash_attn=self._settings.llm_flash_attn,
                        cache_type_k="q4_0",  # quantized KV
                        cache_type_v="q4_0",
                    )
                    try:
                        await self._llm_process.start()
                        self._ipc.emit_signal("process_state_change",
                                              component=component, state="ready")
                        self._ipc.emit_signal("system.vram_optimized",
                                              ctx_size=6144, cache_type="q4_0")
                        return self._mode
                    except Exception as e:
                        logger.error("VRAM-guarded restart failed for %s: %s", component, e)

        # count >= 3 or restart failed: degrade
        logger.error("Component '%s' failed %d times — degrading pipeline", component, count)
        self._ipc.emit_error(component,
                             f"Failed {count} times — manual intervention needed",
                             consecutive=count)

        # Recalculate mode
        llm_ok = self._llm_process is not None and self._llm_process.is_running
        tts_ok = self._tts is not None
        if component == "tts":
            tts_ok = False
        if component == "llm":
            llm_ok = False

        self._mode = self._determine_mode(llm_ok, tts_ok)
        self._ipc.emit_signal("process_state_change",
                              component="system", state=self._mode.value)
        return self._mode

    async def start_component(self, component: str) -> str:
        """Start a single component. Returns its new state ('ready' or 'error')."""
        self._set_component_state(component, "starting")
        try:
            if component == "llm":
                if self._llm_process and self._llm_process.is_running:
                    self._set_component_state("llm", "ready")
                    return "ready"
                self._llm_process = LlamaCppProcess(
                    server_binary=str(Path(self._settings.llm_model).parent.parent / "bin" / "llama-server"),
                    model_path=self._settings.llm_model,
                    port=self._settings.llm_port,
                    ctx_size=self._settings.llm_ctx_size,
                    gpu_layers=self._settings.llm_gpu_layers,
                    threads=self._settings.llm_threads,
                    batch_size=self._settings.llm_batch_size,
                    flash_attn=self._settings.llm_flash_attn,
                )
                await self._llm_process.start()

            elif component == "tts":
                if self._tts is not None:
                    self._set_component_state("tts", "ready")
                    return "ready"
                from voice_assistant.adapters.omnivoice_tts import OmniVoiceTTS
                from voice_assistant.core.voice_clone import get_profile_path
                self._tts = OmniVoiceTTS()
                await self._tts.load()
                # Load any cached voice profiles
                for vl in self._settings.voice_languages:
                    profile_path = get_profile_path(f"voice_{vl.id}")
                    if profile_path.exists():
                        try:
                            self._tts.load_voice_profile_sync(vl.id, str(profile_path))
                        except Exception:
                            pass
                self._tts.set_language(self._settings.default_language)

            elif component == "asr":
                if self._asr is not None:
                    self._set_component_state("asr", "ready")
                    return "ready"
                from voice_assistant.adapters.parakeet_asr import ParakeetASR
                supported = [vl.id for vl in self._settings.voice_languages] or ["en"]
                self._asr = ParakeetASR(supported_languages=supported)

            elif component == "vad":
                if self._vad is not None:
                    self._set_component_state("vad", "ready")
                    return "ready"
                from voice_assistant.models.silero_vad import SileroVAD
                self._vad = SileroVAD()

            else:
                self._set_component_state(component, "error")
                return "error"

            # Warmup the just-started component
            try:
                await self._warmup_single(component)
            except Exception:
                logger.warning("Warmup failed for %s after restart (non-fatal)", component, exc_info=True)

            self._set_component_state(component, "ready")
            self._recalculate_mode()
            return "ready"

        except Exception as e:
            logger.error("Failed to start %s: %s", component, e)
            self._ipc.emit_error(component, f"Failed to start: {e}")
            self._set_component_state(component, "error")
            self._recalculate_mode()
            return "error"

    async def stop_component(self, component: str) -> str:
        """Stop a single component. Returns 'idle'."""
        self._set_component_state(component, "stopping")
        try:
            if component == "llm" and self._llm_process:
                await self._llm_process.stop()
                self._llm_process = None
            elif component == "tts" and self._tts:
                await self._tts.shutdown()
                self._tts = None
            elif component == "asr" and self._asr:
                await self._asr.shutdown()
                self._asr = None
            elif component == "vad" and self._vad:
                self._vad = None
        except Exception as e:
            logger.error("Failed to stop %s: %s", component, e)

        self._set_component_state(component, "idle")
        self._recalculate_mode()
        return "idle"

    def _recalculate_mode(self) -> None:
        """Recalculate and emit pipeline mode based on running components."""
        llm_ok = self._llm_process is not None and self._llm_process.is_running
        tts_ok = self._tts is not None
        self._mode = self._determine_mode(llm_ok, tts_ok)
        self._ipc.emit_signal("process_state_change", component="system", state=self._mode.value)

    async def shutdown(self) -> None:
        """Shut down all managed processes."""
        if self._llm_process:
            await self._llm_process.stop()
            self._llm_process = None
        self._mode = PipelineMode.DISABLED
        logger.info("ProcessManager shutdown complete")
