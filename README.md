# NOVA — Desktop Voice Assistant

<p align="center">
  <strong>Production-grade, offline-capable voice assistant for Linux / macOS / Windows</strong><br/>
  Built to the same engineering standard as Google Assistant — but with full local control.
</p>

---

## Architecture Overview

```
 AudioFrame        TranscriptResult    IntentObject      ActionPlan
    |                    |                  |                 |
[AudioEngine] -> [WakeWordDetector] -> [STT] -> [NLUPipeline] -> [ContextManager]
                                                                        |
                                                               [TaskPlanner]
                                                                        |
                                                               [SafetyLayer]
                                                                        |
                                                               [ExecutionEngine]
                                                                        |
                                                          ExecutionResult |
                                                               [ResponseGenerator/TTS]
```

## Subsystems

| Module | Responsibility | Technology |
|---|---|---|
| `AudioEngine` | Mic capture, noise gating, VAD | PyAudio + webrtcvad |
| `WakeWordDetector` | Always-on "Hey Nova" detection | OpenWakeWord (MIT) |
| `STT` | Streaming transcription, offline-first | faster-whisper (Whisper.cpp) |
| `NLUPipeline` | Intent + entity extraction, hybrid routing | regex + spaCy + LLM |
| `ContextManager` | Session memory, coreference resolution | Pure Python + JSON persistence |
| `TaskPlanner` | Intent → ActionPlan | Template YAML + LLM planner |
| `ExecutionEngine` | Dispatch steps; OS, browser, UI automation | asyncio + Playwright + pyautogui |
| `ResponseGenerator` | TTS synthesis | Piper TTS / Coqui TTS |
| `PluginRegistry` | Dynamic skill loading | Python importlib |
| `SafetyLayer` | Permission model, confirmation gates | Regex + risk classification |

## Quick Start

### Prerequisites

- Python 3.11+
- `portaudio` (Linux: `sudo apt install portaudio19-dev`)
- `wmctrl` + `xdotool` (Linux window management)

### Installation

```bash
git clone https://github.com/KRAZATEC/nova-desktop-voice-assistant.git
cd nova-desktop-voice-assistant

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium

# Download spaCy language model
python -m spacy download en_core_web_sm
```

### Running Nova

```bash
python -m nova.main
```

Or with the CLI:
```bash
nova start
```

## Project Structure

```
nova-desktop-voice-assistant/
├── nova/
│   ├── __init__.py          # Package version & metadata
│   ├── core/
│   │   ├── pipeline.py       # Main orchestrator (asyncio)
│   │   ├── config.py         # NovaConfig Pydantic settings
│   │   ├── logger.py         # Structured logging setup
│   │   └── exceptions.py     # Nova-specific exception hierarchy
│   ├── audio/
│   │   ├── engine.py         # Mic capture + noise gating
│   │   ├── vad.py            # Voice activity detection
│   │   └── wake_word.py      # OpenWakeWord integration
│   ├── stt/
│   │   ├── base.py           # STTBase abstract interface
│   │   ├── whisper_adapter.py # faster-whisper offline STT
│   │   └── online_adapter.py  # Deepgram / Google STT fallback
│   ├── nlu/
│   │   ├── pipeline.py       # Hybrid NLU: rule -> ML -> LLM
│   │   ├── schemas.py        # IntentObject, ActionPlan Pydantic models
│   │   └── intent_classifier.py # spaCy ML classifier
│   ├── context/
│   │   ├── manager.py        # ContextManager + SessionContext
│   │   └── resolver.py       # Coreference resolution
│   ├── planner/
│   │   ├── task_planner.py   # Template + LLM planner
│   │   ├── template_loader.py # YAML skill definitions
│   │   └── llm_planner.py    # LLM-driven plan generation
│   ├── executor/
│   │   ├── engine.py         # Async execution loop
│   │   ├── os_actions.py     # Cross-platform OS automation
│   │   ├── browser_actions.py # Playwright browser automation
│   │   ├── ui_actions.py     # pyautogui fallback
│   │   └── receipts.py       # Execution logging + undo stack
│   ├── tts/
│   │   ├── base.py           # TTSBase abstract interface
│   │   ├── piper_adapter.py  # Piper TTS (offline, fastest)
│   │   └── elevenlabs_adapter.py # ElevenLabs (cloud, best quality)
│   ├── safety/
│   │   ├── validator.py      # Risk classification + confirmation
│   │   ├── audit_log.py      # Append-only audit log
│   │   └── permission_manager.py # YAML permission manifest
│   ├── plugins/
│   │   ├── registry.py       # Dynamic plugin discovery
│   │   ├── base_plugin.py    # AbstractPlugin interface
│   │   └── builtin/          # Built-in skills (weather, reminders, etc.)
│   ├── llm/
│   │   ├── client.py         # Provider-agnostic LLM client
│   │   ├── ollama_adapter.py # Local Ollama (Mistral-7B, Phi-3)
│   │   └── prompt_templates.py # Reusable prompt templates
│   └── config/
│       ├── settings.yaml     # Nova runtime config
│       ├── permissions.yaml  # Per-capability permissions
│       └── skills/           # YAML skill definitions
├── tests/
│   ├── unit/               # Unit tests per module
│   └── integration/        # End-to-end pipeline tests
├── scripts/
│   ├── setup.sh            # One-shot setup script
└──   └── install_models.py   # Download STT/TTS models
```

## NLU Hybrid Routing

```
Utterance
    |
    +---> Rule-based regex (< 2ms)  --- high-confidence match --> IntentObject
    |
    +---> spaCy text-cat (< 50ms)  --- confidence >= 0.72 ----> IntentObject
    |
    +---> LLM structured output (< 800ms) -------------------- > IntentObject
    |
    +---> unknown intent
```

## Latency Budget (mid-range laptop, no GPU)

| Stage | Target | Technology |
|---|---|---|
| Wake word detection | < 50ms | OpenWakeWord |
| STT (3s utterance) | < 600ms | faster-whisper small |
| NLU (rule path) | < 5ms | regex |
| NLU (ML path) | < 50ms | spaCy |
| NLU (LLM path) | < 800ms | Ollama Phi-3 |
| Task planning (template) | < 50ms | YAML lookup |
| Task planning (LLM) | < 1200ms | Ollama |
| First TTS word | < 300ms | Piper TTS |
| **Total (fast path)** | **< 1.0s** | Rule + Template |

## Safety Model

All actions are classified before execution:

- **LOW** (open app, search) — auto-approved
- **MEDIUM** (write file, submit form) — verbal confirmation required
- **HIGH** (delete file, send email) — explicit "confirm" required
- **CRITICAL** (format drive, system files) — always blocked

Forbidden patterns (regex-blocked): `/etc/`, `rm -rf`, `sudo`, `curl | bash`, `mkfs`

## Technology Stack

| Component | Choice | Rationale |
|---|---|---|
| Wake word | OpenWakeWord | Free, MIT, offline, 1-2% CPU |
| STT offline | faster-whisper | 4x faster than original Whisper, low WER |
| STT online | Deepgram | Lowest latency for noisy desktop audio |
| NLU ML | spaCy text-cat | Trainable, < 50ms, no GPU needed |
| LLM local | Ollama (Phi-3/Mistral-7B) | Runs on CPU with GGUF 4-bit quantization |
| LLM cloud | Claude API | Best reasoning for complex commands |
| TTS offline | Piper TTS | < 100ms TTFB, natural voice, fully local |
| TTS cloud | ElevenLabs | Best naturalness for premium UX |
| Browser | Playwright | Async, stable, cross-browser |
| OS automation | pyautogui + xdotool | Cross-platform fallback |

## Plugin System

Plugins are auto-discovered from `nova/plugins/builtin/` and any user plugin directory.

Each plugin must implement `AbstractPlugin`:

```python
class MyPlugin(AbstractPlugin):
    @property
    def supported_intents(self) -> list[str]:
        return ["my_custom_intent"]
    
    async def execute(self, intent: IntentObject, context: ContextManager) -> ExecutionResult:
        ...
```

## Contributing

1. Fork the repo
2. Create a feature branch
3. Run `pytest tests/` and ensure all tests pass
4. Submit a PR with description

## License

MIT — see [LICENSE](LICENSE)
