# nla-token-selector

Which token deserves an NLA explanation? A Natural Language Autoencoder (NLA)
reads a language model's residual-stream activation at one token and writes a
paragraph explaining it, at roughly 500 generated tokens per explanation. You
cannot explain every position in a long transcript, so this project asks whether
a cheap number, computed in one ordinary forward pass before any NLA call, picks
the positions where the NLA lands on a task-relevant explanation.

Spun out of the per-claim calibration project (`~/LocalProjects/calibrated_nla`).
Inference-only: it drives externally launched SGLang servers over HTTP and reads
cached activations from disk.

## Benchmarks

- **OpenPromptInjection** and **Tensor Trust** — prompt injection.
- **Liars' Bench** and the **taboo organisms** (secret words moon/ship/snow) — deception.

Four open NLA targets: `q7` Qwen2.5-7B, `g12` Gemma-3-12B, `g27` Gemma-3-27B,
`l70` Llama-3.3-70B.

## Layout

```
src/nla_token_selector/   NLAClientLite + vendored NLA inference + settings
configs/base.yaml         four-model registry
selector/                 the experiment code (see selector/README.md for the map)
findings/token-selector/  result write-ups
paper_results/bridge/     committed per-token CSV tables
paper/                    manuscript
results/                  gitignored
```

## Setup

Python ≥3.13 and `uv`.

```bash
uv sync
python selector/signals.py --selftest        # offline signal-math self-test
```

Actual runs need externally launched SGLang servers and the park models; see
`selector/README.md` for the reproduce steps and `CLAUDE.md` for the architecture.
