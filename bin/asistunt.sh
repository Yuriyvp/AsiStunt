#!/bin/bash
export VOICE_ASSISTANT_ROOT="/home/winers/voice-assistant"
cd "$VOICE_ASSISTANT_ROOT"
source .venv/bin/activate
export PATH="$VOICE_ASSISTANT_ROOT/bin:$PATH"
exec "$VOICE_ASSISTANT_ROOT/ui/src-tauri/target/release/ui"
