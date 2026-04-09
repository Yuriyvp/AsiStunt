# AsiStunt

**Fully local voice assistant for Ubuntu 24.04 with RTX 3090.**

Private, real-time, conversational AI that runs entirely on your hardware. No cloud services, no data leaves your machine.

---

## Features

- **Real-time voice conversation** — speak naturally, get spoken responses
- **Barge-in support** — interrupt the assistant mid-sentence, it stops and listens
- **Multi-language** — Croatian, English, German, and more (auto-detected per turn)
- **Mood-aware responses** — voice tone adapts to conversation context (warm, playful, calm, concerned)
- **Voice cloning** — clone any voice from a reference audio sample
- **SOUL personality system** — YAML-based persona configuration (backstory, personality, voice style)
- **Rolling memory** — L2 summaries keep long conversations coherent within token budgets
- **Degraded modes** — gracefully handles component failures (TEXT_ONLY, TRANSCRIBE, DISABLED)
- **3-tier error escalation** — silent restart, VRAM-guarded restart, degraded mode
- **VRAM guard** — monitors GPU memory pressure, applies mitigations automatically
- **Desktop app** — Tauri 2 shell with compact/expanded modes, system tray, global shortcuts
- **Setup wizard** — 8-step onboarding (audio check, language, persona, voice preview)
- **Debug dashboard** — 7-tab real-time monitoring (VAD, ASR, LLM, TTS, pipeline, SOUL, events)
- **Accessibility** — ARIA roles, focus trapping, keyboard navigation, rem-based scaling

## Architecture

```
Microphone → RNNoise → Silero VAD → Parakeet ASR → llama.cpp LLM
                                                        ↓
                              Speaker ← Playlist ← OmniVoice TTS ← Sentence Chunker ← Mood Parser
```

| Component | Model | Runtime |
|---|---|---|
| Voice Activity Detection | Silero VAD v5 | CPU (sherpa-onnx) |
| Speech Recognition | Parakeet TDT 0.6B v3 INT8 | CPU (sherpa-onnx) |
| Language Model | Gemma 4 26B-A4B IQ4_XS | GPU (llama.cpp subprocess) |
| Text-to-Speech | OmniVoice (k2-fsa) | GPU (in-process, float16) |
| Audio Denoising | RNNoise | CPU (librnnoise.so via ctypes) |

**State Machine:** `IDLE → LISTENING → PROCESSING → SPEAKING → IDLE` (with `INTERRUPTED` for barge-in)

**Degraded Modes:** `FULL` | `TEXT_ONLY` (TTS down) | `TRANSCRIBE` (LLM down) | `DISABLED` (both down)

## Requirements

### Hardware
- **GPU:** NVIDIA RTX 3090 (24 GB VRAM) or equivalent
- **CPU:** AMD Zen 3 or better (ASR runs on CPU)
- **RAM:** 32 GB recommended
- **Microphone + speakers/headphones**

### Software
- Ubuntu 24.04 LTS
- Python 3.12+
- Node.js 20+ and npm
- Rust toolchain (for Tauri)
- CUDA 12.x with NVIDIA drivers
- `librnnoise.so` (system or bundled)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Yuriyvp/AsiStunt.git
cd AsiStunt
```

### 2. Set up Python environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. Download models

Place models in the `models/` directory:

```
models/
├── gemma-4-26B-A4B-it-UD-IQ4_XS.gguf    # LLM (quantized GGUF)
├── silero_vad.onnx                         # VAD model
└── sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/  # ASR model
    ├── model.onnx
    ├── tokens.txt
    └── ...
```

### 4. Set up llama.cpp

Place the pre-built `llama-server` binary in `bin/`:

```bash
# Build from source or download a release
# Must be compiled with CUDA support
bin/llama-server --version
```

### 5. Install UI dependencies

```bash
cd ui
npm install
cd ..
```

### 6. Build and run

```bash
# Development mode (hot reload)
cd ui
npm run tauri dev

# Production build
npm run tauri build
```

### Quick start (Python backend only)

```bash
source .venv/bin/activate
python -m voice_assistant.main
```

## Usage

### Desktop App

The app starts in **compact mode** — a draggable orb with mic controls:

- **Click the orb area** to see it animate with voice state
- **Mic pill** cycles through: Listening → Muted → Push-to-Talk → Text Only
- **Right-click** for context menu (New Conversation, Settings, Debug, Quit)
- **Expand button** (bottom-right ⤢) switches to full view with transcript

### Global Shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl+Shift+Space` | Toggle mute |
| `Ctrl+Shift+D` | Toggle debug dashboard |
| `Ctrl+Shift+C` | Toggle compact/expanded mode |

### System Tray

Right-click the tray icon for: New Conversation, Settings, Debug, Quit.

### SOUL Configuration

Persona files live in `soul/` as YAML:

```yaml
name: Aria
version: 2
personality: "You are Aria, a warm and witty conversational partner..."
backstory: "Created to be a helpful companion..."
voice:
  method: description  # or "clone"
  description: "A warm, friendly female voice"
  # reference_audio: path/to/sample.wav  # for clone method
  filler_style: natural
mood:
  default: warm
  range: [neutral, warm, playful, calm, concerned]
language:
  available: [en, hr, de]
  default: hr
llm:
  model: gemma-4-26B-A4B-it-UD-IQ4_XS.gguf
  ctx_size: 8192
  gpu_layers: 99
  flash_attn: true
  sampling:
    temperature: 0.75
    top_p: 0.9
    top_k: 40
memory:
  summary_style: narrative
```

## Testing

```bash
source .venv/bin/activate

# Run all tests (231 tests, ~13s)
pytest tests/ -v

# Unit tests only (173 tests)
pytest tests/unit/ -v

# Integration tests only (58 tests)
pytest tests/integration/ -v

# Specific test suites
pytest tests/integration/test_barge_in.py -v
pytest tests/integration/test_voice_pipeline.py -v
pytest tests/integration/test_degraded_modes.py -v
pytest tests/integration/test_language_switching.py -v
pytest tests/integration/test_voice_clone.py -v
```

### Frontend build check

```bash
cd ui
npm run build          # Vite build
source "$HOME/.cargo/env" && cargo check   # Rust check (from ui/src-tauri/)
```

## Project Structure

```
AsiStunt/
├── src/voice_assistant/
│   ├── main.py                  # Entry point, IPC command loop
│   ├── core/
│   │   ├── orchestrator.py      # Central pipeline coordinator
│   │   ├── state_machine.py     # 5-state FSM with validated transitions
│   │   ├── audio_input.py       # RNNoise → Normalizer → CaptureRing
│   │   ├── audio_output.py      # Playlist → PlaybackManager → FillerCache
│   │   ├── context_builder.py   # Token-budgeted LLM prompt assembly
│   │   ├── sentence_chunker.py  # Stream → TTS-ready chunks (30-150 chars)
│   │   ├── mood.py              # Mood state + voice parameter mapping
│   │   ├── mood_signal_parser.py # Strips <mood_signal> tags from LLM stream
│   │   ├── ipc.py               # stdin/stdout JSON IPC with throttling
│   │   ├── soul_loader.py       # YAML config + JSON schema validation
│   │   ├── voice_clone.py       # Voice profile caching (SHA256 keys)
│   │   └── vram_guard.py        # GPU memory monitor + pressure mitigations
│   ├── adapters/
│   │   ├── parakeet_asr.py      # sherpa-onnx Parakeet TDT (CPU)
│   │   ├── llamacpp_llm.py      # llama.cpp HTTP streaming client
│   │   └── omnivoice_tts.py     # OmniVoice in-process TTS (GPU)
│   ├── ports/
│   │   ├── asr.py               # ASR abstract interface
│   │   ├── llm.py               # LLM abstract interface
│   │   └── tts.py               # TTS abstract interface
│   ├── models/
│   │   └── silero_vad.py        # Silero VAD wrapper (sherpa-onnx)
│   ├── memory/
│   │   ├── context_window.py    # Token budget constants (8192 ctx)
│   │   └── rolling_summary.py   # L2 incremental LLM summaries
│   ├── process/
│   │   └── manager.py           # llama.cpp subprocess + error escalation
│   └── debug/
│       └── signal_types.py      # 24 signal type enums
├── ui/
│   ├── src/
│   │   ├── App.jsx              # Root: compact/expanded modes, wizard
│   │   ├── components/
│   │   │   ├── Orb.jsx          # Animated canvas orb (simplex noise)
│   │   │   ├── MicPill.jsx      # Mic state toggle (4 modes)
│   │   │   ├── StatusLine.jsx   # Pipeline state indicator
│   │   │   ├── MoodGlow.jsx     # Mood color indicator
│   │   │   ├── Transcript.jsx   # Chat-style conversation view
│   │   │   ├── TextInput.jsx    # Text input with send button
│   │   │   ├── Settings.jsx     # Settings overlay (focus-trapped)
│   │   │   ├── DebugWindow.jsx  # 7-tab debug dashboard
│   │   │   └── wizard/          # 8-step onboarding wizard
│   │   │       ├── WelcomeStep.jsx
│   │   │       ├── ConsentStep.jsx
│   │   │       ├── AudioCheckStep.jsx
│   │   │       ├── HeadphoneCheckStep.jsx
│   │   │       ├── LanguageStep.jsx
│   │   │       ├── PersonaStep.jsx
│   │   │       ├── VoicePreviewStep.jsx
│   │   │       └── DoneStep.jsx
│   │   ├── pages/               # Debug dashboard pages
│   │   │   ├── Dashboard.jsx    # Overview: latency, VRAM, TPS
│   │   │   ├── VADPage.jsx      # Energy chart + VAD events
│   │   │   ├── LLMPage.jsx      # Token stream + stats
│   │   │   ├── TTSPage.jsx      # Synthesis log + test bench
│   │   │   ├── AudioPipelinePage.jsx  # State diagram + playlist
│   │   │   ├── SOULPage.jsx     # YAML editor + validation
│   │   │   └── EventLogPage.jsx # Filtered event log
│   │   ├── hooks/
│   │   │   ├── useSidecar.js    # Python sidecar IPC bridge
│   │   │   └── useSignals.js    # Signal type filtering
│   │   └── styles/
│   │       └── theme.css        # Dark theme, CSS variables, a11y
│   ├── src-tauri/
│   │   └── src/lib.rs           # Sidecar spawn, shortcuts, tray
│   └── package.json
├── tests/
│   ├── unit/                    # 173 unit tests
│   └── integration/             # 58 integration tests
├── soul/
│   └── schema.json              # SOUL YAML validation schema
├── models/                      # AI models (gitignored)
├── bin/                         # llama-server binary (gitignored)
├── data/                        # Session/summary storage (gitignored)
├── voice_db/                    # Voice profiles + fillers (gitignored)
├── pyproject.toml               # Python project config
└── .gitignore
```

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Desktop Shell | Tauri | 2.10.3 |
| Frontend | React + Vite | 19.1 + 7.0 |
| Charts | Recharts | 3.8.1 |
| Backend | Python (asyncio) | 3.12 |
| ASR Runtime | sherpa-onnx | — |
| LLM Runtime | llama.cpp | — |
| TTS | OmniVoice (k2-fsa) | 0.1.3 |
| Audio I/O | sounddevice | 0.4.7 |
| GPU Monitor | pynvml | 13.0.1 |
| Language Detection | langdetect | 1.0.9 |

## License

All rights reserved. This is a private project.
