# Contributing

Guidelines for working on AsiStunt.

## Prerequisites

- Ubuntu 24.04 LTS
- Python 3.12+
- Node.js 20+ / npm
- Rust toolchain (rustup)
- NVIDIA GPU with CUDA 12.x (RTX 3090 recommended)
- `librnnoise.so` installed system-wide or bundled

## Development Setup

```bash
# Clone
git clone https://github.com/Yuriyvp/AsiStunt.git
cd AsiStunt

# Python
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# UI
cd ui && npm install && cd ..

# Models (not in repo — see README.md)
# Place in models/ directory
```

## Running

```bash
# Python backend only
source .venv/bin/activate
python -m voice_assistant.main

# Full app (Tauri + React + Python)
cd ui
npm run tauri dev
```

## Testing

```bash
source .venv/bin/activate

# All tests
pytest tests/ -v

# Unit only
pytest tests/unit/ -v

# Integration only
pytest tests/integration/ -v

# Single file
pytest tests/unit/test_state_machine.py -v
```

## Code Style

- **Python:** ruff, line-length 100, target Python 3.12
- **Lint check:** `ruff check src/ tests/`
- **Frontend:** standard React/JSX conventions

## Project Layout

```
src/voice_assistant/     # Python backend
  core/                  # Pipeline components
  adapters/              # Model-specific implementations
  ports/                 # Abstract interfaces (ASR, LLM, TTS)
  models/                # VAD wrapper
  memory/                # Context window + rolling summary
  process/               # Subprocess management
  debug/                 # Signal types

ui/                      # Tauri + React frontend
  src/components/        # React components
  src/pages/             # Debug dashboard pages
  src/hooks/             # useSidecar, useSignals
  src-tauri/             # Rust backend

tests/
  unit/                  # 173 unit tests
  integration/           # 58 integration tests
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical documentation.

## Adding a New Adapter

1. Create abstract port in `ports/` (if new capability)
2. Implement adapter in `adapters/` extending the port
3. Register in `process/manager.py` startup sequence
4. Add unit tests in `tests/unit/`
5. Add integration tests in `tests/integration/`

## Adding a New SOUL Persona

1. Create `soul/{name}.soul.yaml` following `soul/schema.json`
2. Required fields: name, version (2), personality, voice, mood, language, llm
3. Optional: backstory, memory config, session_restore_hours
4. Validate: the app validates against schema on load

## Commit Guidelines

- Keep commits focused — one logical change per commit
- Run `pytest tests/ -v` before committing
- Run `cd ui && npm run build` to verify frontend builds
