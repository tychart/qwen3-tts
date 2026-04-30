This `qwen3-tts` repo contains:

**Main app**: `main.py` — TTS inference using Qwen-TTS model.

**Utilities**: `start_awx_workflow.py` — standalone AWX automation script.

**Dependencies**:
- Uses `uv` as package manager (`uv sync` to install deps).
- Python 3.12+ required.
- Flash Attention 2 (`flash-attn==2.8.3`) with build-time deps: packaging, wheel, setuptools, ninja, torch — install via `[tool.uv.extra-build-dependencies]` in `pyproject.toml`.

**Model**:
- Hugging Face models: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` (small), `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (big).

**Required for TTS**:
- CUDA GPU with compute capability >= 7.5.
- Model loads with `torch.bfloat16`.
- Run with `uv run python main.py --text-file story.txt --model small`.

**Testing**:
- No tests in repo.