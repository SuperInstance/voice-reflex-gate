# Voice Reflex Gate

> *STT output as hash key for deterministic response routing.*

The voice reflex gate sits between your speech-to-text layer and your model cascade. It takes the STT text output, fuzzy-matches it against known request patterns, and returns a cached response if a match is found — zero model invocation, zero GPU cycles, near-zero latency.

## Architecture

```
STT text ──▶ Gate
               │
               ├─ Tier 1: EXACT MATCH (instant, confidence ≥ 0.95)
               │     Reflex key = hash(text + context_vector)
               │     → Return cached response
               │
               ├─ Tier 2: FUZZY MATCH (confidence-weighted, 0.60–0.94)
               │     Fuzzy-match against known patterns
               │     → Return cached response with confidence score
               │
               └─ Tier 3: NO MATCH (confidence < 0.60)
                     Cascade to model
                     → Learner observes response
                     → After 3 similar requests, compiles a new reflex
```

## The Reflex Key

The reflex key combines:
- **STT text** (normalized)
- **Context vector**: time of day, recent command history, operational mode, location state

Same words in different contexts → different reflexes. "Check depth" at the dock vs underway produce different responses.

## Temporal Validity & Decay

Each reflex carries a temporal validity window (weather: 30min, tide: 6hr, navigation: until conditions change). Reflexes that aren't accessed fade — confidence decays daily based on recency, seasonal, and context-drift factors. Three kinds of forgetting:

1. **Fading** — gradual confidence decay; reflex still matchable but treated with skepticism
2. **Supersession** — old reflex replaced when context demands a different response
3. **Eviction** — reflex actively removed when dangerous or contradicted by external data

## Learning Loop

When the model handles a request, the learner observes whether the STT+context was similar to previous model-handled requests. After 3 similar requests with consistent responses, it auto-compiles a reflex. The 4th request is handled by the cache — no model needed.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```python
from voice_reflex import Gate, Context, ReflexCache

# Build the gate
cache = ReflexCache()
gate = Gate(cache=cache)

# Process a voice command
ctx = Context(
    time_of_day="morning",
    operational_mode="cruising",
    location_state="underway",
    recent_commands=["check weather", "plot course"],
)
result = gate.process("what's the weather", ctx)

if result.hit:
    print(f"Reflex hit (confidence={result.confidence}): {result.response}")
else:
    print("No reflex — cascade to model")
    # ... invoke model ...
    gate.learn("what's the weather", ctx, model_response)
```

## Running Tests

```bash
pytest
```

## License

MIT — see [LICENSE](LICENSE).

---

*The system gets faster as you use it — not because the model is getting faster, but because more of your commands are handled by reflex. The model isn't invoked at all for most things.*
