# Selecting The Most Informative Tokens in Natural Language Autoencoders

Which token deserves an NLA explanation? A Natural Language Autoencoder (NLA)
reads a language model's residual-stream activation at one token and writes a
paragraph explaining it, at roughly 500 generated tokens per explanation. You
cannot explain every position in a long transcript, so this project asks whether
a cheap number, computed in one ordinary forward pass before any NLA call, picks
the positions where the NLA lands on a task-relevant explanation.

## Benchmarks

- **OpenPromptInjection** and **Tensor Trust** for prompt injection.
- **Liars' Bench** and the **taboo organisms** (secret words moon/ship/snow) for deception.

Four open NLA targets: `q7` Qwen2.5-7B, `g12` Gemma-3-12B, `g27` Gemma-3-27B,
`l70` Llama-3.3-70B.

## Setup

Python 3.13 and `uv`.

```bash
uv sync
python selector/signals.py --selftest        # offline signal-math self-test
```
