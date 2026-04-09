# Architecture

Detailed technical documentation for AsiStunt — a fully local voice assistant.

## Table of Contents

- [System Overview](#system-overview)
- [Pipeline](#pipeline)
- [State Machine](#state-machine)
- [Backend (Python)](#backend-python)
- [Frontend (Tauri + React)](#frontend-tauri--react)
- [IPC Protocol](#ipc-protocol)
- [Memory System](#memory-system)
- [Error Handling](#error-handling)
- [Audio Pipeline](#audio-pipeline)
- [Voice Cloning](#voice-cloning)
- [SOUL Personality System](#soul-personality-system)
- [Debug System](#debug-system)
- [Signal Types](#signal-types)

---

## System Overview

AsiStunt runs as a **Tauri 2 desktop application** that spawns a **Python sidecar process**. The Tauri shell (Rust) manages the window, system tray, global shortcuts, and IPC bridge. The Python backend handles all AI inference — VAD, ASR, LLM, TTS — and audio I/O.

```
┌─────────────────────────────────────────────────────────┐
│                    Tauri Shell (Rust)                    │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │   Webview    │  │  Global      │  │  System Tray  │  │
│  │  (React 19)  │  │  Shortcuts   │  │               │  │
│  └──────┬───────┘  └──────┬───────┘  └───────┬───────┘  │
│         │                 │                   │          │
│         └─────────────────┼───────────────────┘          │
│                           │                              │
│              python_event / send_command                  │
│                           │                              │
│         ┌─────────────────┴──────────────────┐           │
│         │         Sidecar Bridge             │           │
│         │   stdin (JSON) ↔ stdout (JSON)     │           │
│         └─────────────────┬──────────────────┘           │
└───────────────────────────┼──────────────────────────────┘
                            │
┌───────────────────────────┼──────────────────────────────┐
│                    Python Backend                        │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │                  Orchestrator                     │    │
│  │                                                   │    │
│  │  AudioInput → VAD → ASR → LLM → Chunker → TTS   │    │
│  │                                    ↓              │    │
│  │              PlaybackManager ← Playlist           │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────┐   │
│  │ StateMachine│  │ProcessManager│  │  VRAM Guard    │   │
│  └────────────┘  └──────────────┘  └────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

---

## Pipeline

### Voice Turn Flow

```
1. Microphone (16 kHz, mono)
2. RNNoise denoising (ctypes → librnnoise.so, 48 kHz internal, resample back to 16 kHz)
3. RMS Normalizer (-20 dBFS target, fast attack 0.01 / slow release 0.001)
4. CaptureRing (30-second circular buffer, 480 samples/frame)
5. Silero VAD (512-sample windows, threshold 0.5, min_speech 250ms, min_silence 300ms)
6. Speech extraction → drain_speech_samples()
7. Parakeet ASR (offline transducer, CPU, INT8, RTF ~0.15–0.35)
8. Language detection (langdetect library — Parakeet result.lang is empty)
9. Context assembly (persona card + mood + rolling summary + dialogue history)
10. llama.cpp LLM (streaming SSE via HTTP, GPU subprocess)
11. Mood signal parser (strips <mood_signal> tags from token stream)
12. Sentence chunker (30–150 char chunks at clause/sentence boundaries)
13. OmniVoice TTS (in-process, GPU, float16, 24 kHz output)
14. Playlist queuing (AudioChunks with text tracking)
15. PlaybackManager → sounddevice output (24 kHz)
```

### Text Input Flow

Same as voice from step 9 onward. Text input skips steps 1-8, goes directly to `IDLE → PROCESSING`.

### Barge-In Flow

```
1. User starts speaking during SPEAKING state
2. 300ms gate check (prevents accidental triggers from short sounds)
3. PlaybackManager.fade_out(30ms crossfade)
4. Playlist.drop_future() — remove unplayed chunks
5. LLM.cancel() — close HTTP connection
6. Mark last assistant turn as interrupted (record spoken vs. unspoken text)
7. State → INTERRUPTED → LISTENING (or PROCESSING for text barge-in)
8. New turn begins processing
```

---

## State Machine

Defined in `core/state_machine.py`.

### States

| State | Description |
|---|---|
| `IDLE` | Waiting for input. VAD loop running if mic active. |
| `LISTENING` | VAD detected speech start. Accumulating audio. |
| `PROCESSING` | ASR complete, LLM generating response. |
| `SPEAKING` | TTS output playing through speakers. |
| `INTERRUPTED` | Barge-in detected. Transitional state. |

### Valid Transitions

```
IDLE        → LISTENING, PROCESSING
LISTENING   → IDLE, PROCESSING
PROCESSING  → SPEAKING, IDLE, PROCESSING (self — for retry)
SPEAKING    → IDLE, INTERRUPTED
INTERRUPTED → LISTENING, PROCESSING, IDLE
```

### Modes

| Mode | Description |
|---|---|
| `FULL` | All components operational |
| `TEXT_ONLY` | LLM works, TTS down — text responses only |
| `TRANSCRIBE` | ASR works, LLM down — transcription only |
| `DISABLED` | Both LLM and TTS down — show error banner |

---

## Backend (Python)

### Entry Point — `main.py`

- Configures logging to stderr (stdout reserved for IPC)
- Loads SOUL config → initializes IPC → checks VRAM → starts ProcessManager
- Enters stdin command loop handling: `text_input`, `set_mode`, `reload_soul`, `mute_toggle`, `new_conversation`, `shutdown`

### Orchestrator — `core/orchestrator.py`

Central coordinator wiring all pipeline stages.

**Key class:** `Orchestrator`
- Owns: StateMachine, AudioInput, AudioOutput, VAD, ASR, LLM, TTS, ContextBuilder, RollingSummary, MoodState, SentenceChunker, MoodSignalParser
- Maintains dialogue history as list of `Turn` dataclasses

**Key methods:**
| Method | Purpose |
|---|---|
| `start()` / `stop()` | Lifecycle management, VAD loop init |
| `handle_text_input(text)` | Process text, handle barge-in during SPEAKING |
| `_vad_loop()` | Continuous VAD event processing |
| `_execute_barge_in()` | Fade-out, cancel LLM/TTS, mark interrupted |
| `_process_voice_turn(audio)` | ASR → text processing |
| `_process_turn(text, source)` | Full pipeline: context → LLM → mood → chunk → TTS |
| `_synthesize_and_queue(text)` | TTS with mood voice params |

**Constants:** `BARGE_IN_GATE_MS = 300`, `LISTENING_TIMEOUT_S = 30`

### Audio Input — `core/audio_input.py`

| Class | Purpose |
|---|---|
| `RNNoiseWrapper` | ctypes wrapper for librnnoise.so; 16↔48 kHz resampling |
| `Normalizer` | RMS-based leveling (-20 dBFS), DC removal, gain clamp ±20 dB |
| `CaptureRing` | 30-second circular buffer (480,000 samples), thread-safe |
| `AudioInput` | Orchestrator: sounddevice → denoise → normalize → ring |

**Constants:** `SAMPLE_RATE=16000`, `CHANNELS=1`, `FRAME_DURATION_MS=30`, `RING_SIZE=480000`

### Audio Output — `core/audio_output.py`

| Class | Purpose |
|---|---|
| `AudioChunk` | Dataclass: audio (f32), text, source, played flag |
| `Playlist` | Ordered chunk list with drop_future/clear, text tracking |
| `PlaybackManager` | Reads from playlist via sounddevice callback, 30ms fade-out |
| `FillerCache` | Pre-rendered filler phrases with rate limiting (max 1/3 turns) |

**Constants:** `OUTPUT_SAMPLE_RATE=24000`, `CROSSFADE_SAMPLES=720`

### Adapters

**Parakeet ASR** (`adapters/parakeet_asr.py`):
- sherpa-onnx `OfflineRecognizer.from_transducer()` factory
- Runs sync decode in executor (CPU-bound)
- Language detection via `langdetect` library (Parakeet's `result.lang` is empty)
- Returns: `{text, language, confidence, rtf, duration_s}`

**llama.cpp LLM** (`adapters/llamacpp_llm.py`):
- OpenAI-compatible HTTP client to localhost:8080
- SSE streaming via aiohttp
- Cancel by closing HTTP connection (no /cancel endpoint)
- Default sampling: temperature 0.75, top_p 0.9, top_k 40, min_p 0.05, repeat_penalty 1.1

**OmniVoice TTS** (`adapters/omnivoice_tts.py`):
- In-process inference (float16 on GPU)
- Supports description-based and clone-based voice modes
- Injects mood voice params (speed, pitch shift, energy, instruct tags)
- Returns: f32 numpy array at 24 kHz

### Ports (Abstract Interfaces)

```python
# ports/asr.py
class ASRPort(ABC):
    async def transcribe(audio, sample_rate=16000) -> dict
    async def shutdown()

# ports/llm.py
class LLMPort(ABC):
    async def stream(messages, sampling) -> AsyncIterator[str]
    async def cancel()
    async def health_check() -> bool
    async def tokenize(text) -> int

# ports/tts.py
class TTSPort(ABC):
    async def synthesize(text, voice_params) -> np.ndarray
    async def load_voice_profile(profile_path)
    async def shutdown()
```

### Process Manager — `process/manager.py`

**LlamaCppProcess:**
- Spawns `bin/llama-server` with model path, GPU layers, context size
- Health polling until server responds
- Graceful shutdown via SIGTERM → SIGKILL

**ProcessManager — Startup Sequence (GPU-ordered):**
1. Voice profile check / clone if needed
2. llama.cpp server start + health check
3. OmniVoice TTS load
4. CPU models (VAD, ASR)

**Error Escalation:**
| Failure # | Action |
|---|---|
| 1st | Silent restart |
| 2nd | Restart with VRAM mitigations (ctx=6144, kv=q4_0) |
| 3rd+ | Degrade pipeline mode, emit error banner |

Counter resets after 5 minutes of stable operation.

### Supporting Modules

**Mood** (`core/mood.py`):
- Tone→Mood mapping (happy→playful, sad→concerned, angry→calm, neutral→warm)
- Mood→Voice params (speed, pitch_shift, energy, instruct tags)
- Intensity decay: `*= 0.85` per turn, reset to warm below 0.2

**Mood Signal Parser** (`core/mood_signal_parser.py`):
- Strips `<mood_signal>user_tone=X, intensity=Y</mood_signal>` from LLM tokens
- Handles partial tags split across tokens
- Default neutral mood after 50 tokens if no tag

**Sentence Chunker** (`core/sentence_chunker.py`):
- First chunk: ≥30 chars, split at clause boundary (`,;—:`)
- Subsequent: 60–150 chars, split at sentence boundary (`.!?…`)
- Safety valve: force flush at 150 chars on word boundary
- Abbreviation-aware (Mr., Dr., vs., etc. don't trigger splits)

**VRAM Guard** (`core/vram_guard.py`):
- pynvml GPU memory polling (5s idle, 1s active)
- Pressure threshold: 512 MB free
- Mitigations: cache-type-k=q4_0, cache-type-v=q4_0, ctx-size=6144

---

## Frontend (Tauri + React)

### Tauri Backend — `ui/src-tauri/src/lib.rs`

- Spawns Python sidecar via `.venv/bin/python -u -m voice_assistant.main`
- Pipes stdout JSON → `python_event` Tauri events → webview
- Pipes stderr → Tauri console log with `[python]` prefix
- `send_command` Tauri command → JSON → Python stdin
- Global shortcuts: Ctrl+Shift+Space (mute), Ctrl+Shift+D (debug), Ctrl+Shift+C (compact)
- System tray: New Conversation, Settings, Debug, Quit

### App Layout — `ui/src/App.jsx`

**Two modes:**

| Mode | Window Size | Elements |
|---|---|---|
| Compact | 300 x 400 | Orb (180px) + StatusLine + MicPill + LanguageBadge + LastExchange |
| Expanded | 500 x 700 | Top bar (Orb 40px + StatusLine + MicPill) + Transcript + TextInput |

**Overlays:** Settings (modal, focus-trapped), DebugWindow (tabbed), Wizard (first-run), ContextMenu (right-click)

### Components

| Component | Purpose |
|---|---|
| `Orb` | Canvas-animated orb with simplex noise displacement. Responds to state (speed, amplitude, glow) and mood (color). Energy prop adds visual boost. |
| `MicPill` | Pill button cycling through: listening → muted → PTT → text-only. Color-coded borders. |
| `StatusLine` | ARIA live region showing state text: "Listening...", "Thinking...", "Interrupted". |
| `MoodGlow` | Mood label with color-coded text. Auto-hides after 5s. Only shows non-neutral moods. |
| `Transcript` | Chat view with user/AI bubbles. Auto-scroll unless user scrolled up. Shows last 50 turns. Interrupted turns show spoken vs. unspoken text. |
| `TextInput` | Input field + send button. Enter to submit, Shift+Enter for newline. |
| `Settings` | Full-screen modal with focus trap. Sections: Language, Input/Output, Window, Conversation. Two-step confirmation for destructive actions. |
| `DebugWindow` | 7-tab dashboard: Dashboard, VAD, LLM, TTS, Pipeline, SOUL, Events. |

### Wizard Steps

| Step | Component | Purpose |
|---|---|---|
| 1 | WelcomeStep | Component status indicators (LLM, TTS, ASR, VAD) |
| 2 | ConsentStep | Privacy disclosure + mic consent checkbox |
| 3 | AudioCheckStep | Live mic level bar, threshold detection |
| 4 | HeadphoneCheckStep | Headphone/speaker selection, feedback warning |
| 5 | LanguageStep | Multi-select grid (1-5 languages) |
| 6 | PersonaStep | Persona card selection with descriptions |
| 7 | VoicePreviewStep | Play button, clone progress bar |
| 8 | DoneStep | Success confirmation |

### Debug Pages

| Page | Visualizations |
|---|---|
| Dashboard | 4-metric row (state, mood, input level, TPS) + VRAM bar + latency breakdown (Recharts) |
| VADPage | Energy LineChart (200 samples) + threshold slider + VAD event log |
| LLMPage | Stat boxes (TPS, TTFT, tokens) + live token stream + prompt inspector |
| TTSPage | Synthesis status + test bench + history log with RTF (color-coded) |
| AudioPipelinePage | SVG state diagram (5 nodes) + transition log + playlist grid |
| SOULPage | YAML editor textarea + Validate/Reload buttons + error panel |
| EventLogPage | Category filters + pause/resume + 500-entry log (color-coded, monospace) |

### Hooks

**`useSidecar()`** — connects to Python sidecar via Tauri events:
- Returns `{state, events, sendCommand}`
- Buffers last 100 events
- Parses `state_change` signals to track pipeline state

**`useSignals(signalTypes[])`** — filters incoming events by type:
- Returns `signals` object keyed by signal type
- Ref-based subscriptions to prevent re-renders

### Styling

- Dark theme via CSS custom properties (`theme.css`)
- rem-based spacing for accessibility scaling
- Focus-visible rings: 2px solid green + 4px box-shadow
- Smooth transitions: 0.3s state, 1.5s mood
- Custom scrollbar: 6px, transparent track

---

## IPC Protocol

### Direction: Frontend → Python (stdin)

JSON objects, one per line:

```json
{"cmd": "text_input", "text": "Hello, how are you?"}
{"cmd": "set_mode", "mode": "TEXT_ONLY"}
{"cmd": "mute_toggle"}
{"cmd": "new_conversation"}
{"cmd": "reload_soul", "persona": "Aria"}
{"cmd": "set_languages", "languages": ["en", "hr"]}
{"cmd": "shutdown"}
```

### Direction: Python → Frontend (stdout)

JSON objects, one per line. Emitted by `StdoutEmitter`:

```json
{"type": "state_change", "state": "LISTENING", "ts": 1712700000.0}
{"type": "transcript", "role": "user", "text": "Hello", "language": "en", "confidence": 0.95}
{"type": "transcript", "role": "assistant", "text": "Hi there!", "interrupted": false}
{"type": "token", "text": " world"}
{"type": "signal", "signal": "input_level", "data": {"level": 0.42}, "ts": 1712700000.0}
{"type": "signal", "signal": "mood_change", "data": {"mood": "playful"}, "ts": 1712700000.0}
{"type": "error", "source": "tts", "message": "OmniVoice crashed", "consecutive": 2}
```

### Throttling

High-frequency signals are throttled to prevent UI flooding:
- `input_level`: 100ms minimum interval
- `vram_usage`: 1000ms minimum interval
- Queue max: 256 events (drop-oldest on overflow)

---

## Memory System

### Context Window Budget (8192 tokens)

```
┌──────────────────────────────┐
│ Persona Card          (800)  │
│ Mood Instruction       (50)  │
│ Rolling Summary       (600)  │
│ Dialogue History    (~6050)  │ ← fits ~15-20 turns
│ Overhead              (100)  │
│ Generation Headroom   (500)  │
└──────────────────────────────┘
```

### Rolling Summary (L2 Memory)

- Triggers when dialogue tokens exceed 80% of dialogue budget (~4840 tokens)
- LLM generates incremental summary integrating previous summary + new turns
- Compressed if summary exceeds 600 tokens
- Persisted to `data/summaries/current.txt`
- Minimum 4 turns always kept in dialogue (never evicted)

### Context Assembly

The `ContextBuilder` assembles OpenAI-compatible messages:

```python
[
    {"role": "system", "content": "{persona_card}\n\n{mood_instruction}\n\n{summary}"},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    # ... recent dialogue turns
]
```

Long individual turns are truncated: head (40%) + `[...]` + tail (40%), max 2000 tokens per turn.

---

## Error Handling

### 3-Tier Escalation

```
Component Failure
       │
       ▼
   1st failure → Silent restart
       │
       ▼
   2nd failure → VRAM-guarded restart
       │          (ctx=6144, kv=q4_0)
       │
       ▼
   3rd failure → Degrade pipeline mode
                  + error banner to UI
```

### Degradation Rules

| LLM | TTS | Mode |
|---|---|---|
| OK | OK | FULL |
| OK | Down | TEXT_ONLY |
| Down | OK | TRANSCRIBE |
| Down | Down | DISABLED |

### Stability Window

Failure counter resets after **5 minutes** of stable operation. This prevents permanent degradation from transient failures.

---

## Audio Pipeline

### Input Path

```
Microphone (16 kHz, mono, float32)
    │
    ▼
RNNoise Denoising
  - Resample 16→48 kHz (linear interpolation)
  - Process 480-sample frames at 48 kHz
  - Resample 48→16 kHz
    │
    ▼
RMS Normalizer
  - Target: -20 dBFS
  - Fast attack (0.01), slow release (0.001)
  - DC removal filter
  - Gain clamp: ±20 dB
    │
    ▼
CaptureRing (30s circular buffer)
    │
    ▼
Silero VAD
  - 512-sample windows (32ms at 16kHz)
  - Threshold: 0.5
  - Min speech: 250ms, Min silence: 300ms
```

### Output Path

```
TTS Audio (24 kHz, float32)
    │
    ▼
AudioChunk (audio + text metadata)
    │
    ▼
Playlist (ordered queue)
    │
    ▼
PlaybackManager
  - sounddevice output callback
  - 30ms crossfade on fade-out (720 samples)
  - Text-played tracking
```

### Filler System

Pre-rendered filler phrases ("hmm", "uh", "let me think...") play during LLM latency:
- Rate limited: max 1 in 3 turns, max 4 per minute
- Only plays when pipeline latency > 450ms
- Recency tracking prevents repetition

---

## Voice Cloning

### Cache Key

`SHA256(file_contents + description)[:16]` — stored as `.cachekey` alongside `.voiceprofile`.

### Clone Pipeline

1. Load OmniVoice model
2. Create `VoiceClonePrompt` from reference audio
3. Save to `voice_db/profiles/{soul_name}.voiceprofile` (torch.save)
4. Pre-render 8 filler phrases with cloned voice
5. Unload model, free VRAM

### Cache Invalidation

Profile re-cloned on startup if:
- `.voiceprofile` or `.cachekey` missing
- Stored cache key doesn't match computed key (audio or description changed)

---

## SOUL Personality System

### Schema (JSON Schema Draft 07)

```yaml
name: string (required)
version: integer >= 2 (required)
personality: string (required)
backstory: string (optional)
voice:
  method: "description" | "clone"
  description: string
  reference_audio: string (optional, for clone)
  filler_style: string
mood:
  default: string
  range: [string]
language:
  available: [string]
  default: string
  detection_threshold: number (optional)
llm:
  model: string
  ctx_size: integer (optional)
  port: integer (optional)
  gpu_layers: integer (optional)
  threads: integer (optional)
  batch_size: integer (optional)
  flash_attn: boolean (optional)
  sampling: object (optional)
memory:
  summary_style: string (optional)
  summary_budget_trigger: number (optional)
session_restore_hours: number (optional)
```

### Loading

1. Parse YAML file
2. Validate against `soul/schema.json` (jsonschema)
3. Return `SoulConfig` dataclass with all fields + `.raw` (original dict)
4. `persona_card()` concatenates backstory + personality for system prompt

---

## Debug System

### Signal Architecture

24 signal types defined in `debug/signal_types.py`:

```
VAD:    speech_start, speech_end, barge_in, input_level
ASR:    transcript, detected_language, language_switch
LLM:    request_start, first_token, complete, tokens_per_sec
TTS:    synth_start, synth_end
State:  state_change, mood_change, filler_played
System: end_to_end_latency, vram_usage, vram_optimized,
        device_change, voice_clone_progress, process_state_change, error
```

### Data Flow

```
Python StdoutEmitter
    │ (JSON to stdout, throttled)
    ▼
Tauri lib.rs
    │ (parse JSON, emit python_event)
    ▼
useSidecar hook
    │ (buffer events, extract state)
    ▼
useSignals hook
    │ (filter by signal type)
    ▼
Debug pages (Dashboard, VAD, LLM, TTS, Pipeline, SOUL, Events)
```

---

## Signal Types

Complete reference of all IPC signal types:

| Signal | Direction | Payload | Throttle |
|---|---|---|---|
| `speech_start` | Python → UI | `{timestamp}` | — |
| `speech_end` | Python → UI | `{timestamp, duration_ms}` | — |
| `barge_in` | Python → UI | `{timestamp}` | — |
| `input_level` | Python → UI | `{level: 0.0-1.0}` | 100ms |
| `transcript` | Python → UI | `{role, text, language, confidence}` | — |
| `detected_language` | Python → UI | `{language}` | — |
| `language_switch` | Python → UI | `{from, to}` | — |
| `request_start` | Python → UI | `{timestamp}` | — |
| `first_token` | Python → UI | `{latency_ms}` | — |
| `complete` | Python → UI | `{tokens, duration_ms}` | — |
| `tokens_per_sec` | Python → UI | `{tps}` | — |
| `synth_start` | Python → UI | `{text}` | — |
| `synth_end` | Python → UI | `{duration_ms, rtf}` | — |
| `state_change` | Python → UI | `{state, previous}` | — |
| `mood_change` | Python → UI | `{mood, intensity}` | — |
| `filler_played` | Python → UI | `{text}` | — |
| `end_to_end_latency` | Python → UI | `{total_ms, vad, asr, llm, tts}` | — |
| `vram_usage` | Python → UI | `{used, total, free, pressure}` | 1000ms |
| `vram_optimized` | Python → UI | `{mitigations[]}` | — |
| `device_change` | Python → UI | `{input, output}` | — |
| `voice_clone_progress` | Python → UI | `{percent: 0-100}` | — |
| `process_state_change` | Python → UI | `{process, state}` | — |
| `error` | Python → UI | `{source, message, consecutive}` | — |
