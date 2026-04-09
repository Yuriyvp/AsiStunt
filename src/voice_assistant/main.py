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

    # Command handler
    def handle_command(cmd: dict) -> None:
        cmd_type = cmd.get("cmd")
        logger.info("IPC command: %s", cmd_type)

        if cmd_type == "text_input":
            content = cmd.get("content", "")
            if content:
                logger.info("Text input: %s", content[:80])
                # Will be routed to orchestrator.handle_text_input() when wired

        elif cmd_type == "set_mode":
            new_mode = cmd.get("mode", "")
            logger.info("Mode change request: %s", new_mode)

        elif cmd_type == "reload_soul":
            path = cmd.get("path", soul_path)
            logger.info("SOUL reload request: %s", path)

        elif cmd_type == "mute_toggle":
            logger.info("Mute toggle")

        elif cmd_type == "new_conversation":
            logger.info("New conversation")

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
