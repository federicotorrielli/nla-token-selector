"""Project-wide data shapes.

- `Generation`: one AV decode (explanation, per-token log-probs, per-token
  strings). Returned by `NLAClientLite.generate` and cached in
  `explanations.parquet`.

- `ClaimPrediction`: one calibration unit. A claim is identified by
  `(position_idx, bullet_idx, claim_idx)`, carries its extracted text and a
  method's confidence, and flags whether the judge labelled it verifiable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple


class Generation(NamedTuple):
    explanation: str
    tok_logprobs: list[float]
    tok_strings: list[str]


@dataclass
class ClaimPrediction:
    """One predicted-confidence record for a single atomic claim."""

    position_idx: int
    bullet_idx: int
    claim_idx: int
    claim_text: str
    confidence: float
    is_supported: bool | None
    is_verifiable: bool
    method: str
    method_metadata: dict = field(default_factory=dict)
