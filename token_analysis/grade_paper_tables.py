"""token_analysis/grade_paper_tables.py — shade the paper's tables.

Every table in `paper/tables/` that carries comparable numbers is graded with the
one ramp defined in `table_colour.py`, so a reader sees the shape of a result
before reading a digit. The scale is fixed by what the column measures rather
than by the table it sits in, which keeps one colour language across the paper:

  AUROC        diverging around chance at 0.5, blue above and red below, so a
               signal running backwards is visible as a hue rather than as a
               number below a half.
  lift         diverging around zero, since Tensor Trust localizes negatively.
  rate, gain   sequential blue from zero, since neither can be negative.

Two tables are left plain: the signal definitions, which carry no measurements,
and any column of counts, which are not comparable with the rates beside them.

Running this twice is a no-op: a cell already carrying a fill is left alone.

Usage:
    uv run python token_analysis/grade_paper_tables.py --check
    uv run python token_analysis/grade_paper_tables.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from token_analysis.table_colour import grade_latex, ungrade_latex  # noqa: E402

TABLES = Path(__file__).resolve().parents[1] / "paper" / "tables"

AUROC = {"kind": "diverging", "centre": 0.5, "half_range": 0.35}
LIFT = {"kind": "diverging", "centre": 0.0, "half_range": 0.35}


def rate(vmin: float, vmax: float) -> dict:
    return {"kind": "sequential", "vmin": vmin, "vmax": vmax}


# file stem -> {one-based column: scale}
SPECS: dict[str, dict[int, dict]] = {
    # Dataset, Model, Signal 1, Signal 2, Held-out delta, 95% CI
    "model_best_ensembles": {5: rate(0.0, 0.20)},
    # Dataset, Shared ensemble, Full, delta, Held-out, delta. The two gain
    # columns stay plain: a second scale beside the AUROC columns would invite
    # a comparison between quantities that do not share a range.
    "dataset_shared_ensembles": {3: AUROC, 5: AUROC},
    # Dataset, Model, Positions, Base rate, Lift all, Lift within role
    "label_localization": {4: rate(0.0, 1.0), 5: LIFT, 6: LIFT},
    # Strategy, then one column per model. Every lift here is positive and they
    # span a narrow band, so the scale starts at the smallest rather than at
    # zero; otherwise the ordering the text claims is invisible.
    "opi_strategy": {2: rate(0.15, 0.37), 3: rate(0.15, 0.37),
                     4: rate(0.15, 0.37), 5: rate(0.15, 0.37)},
    # Model, Positions, On-task, Own, Other, Own, Other, moon, ship, snow. One
    # scale over every rate column, so the near-empty control columns read as
    # near-empty beside the recovery columns.
    "taboo_transfer": {3: rate(0.0, 0.35), 4: rate(0.0, 0.35), 5: rate(0.0, 0.35),
                       6: rate(0.0, 0.35), 7: rate(0.0, 0.35), 8: rate(0.0, 0.35),
                       9: rate(0.0, 0.35), 10: rate(0.0, 0.35)},
    # Dataset, Model, Ensemble, then six AUROC columns
    "full_aurocs_model_best": {4: AUROC, 5: AUROC, 6: AUROC, 7: AUROC, 8: AUROC, 9: AUROC},
    # Dataset, Shared ensemble, three AUROC columns
    "full_aurocs_dataset_shared": {3: AUROC, 4: AUROC, 5: AUROC},
}
PLAIN = ["signals"]                      # definitions, nothing to grade
GENERATED = ["signal_auroc", "signal_controls",   # graded by signal_eval.py
             "signal_direction", "signal_position"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="report what would change without writing")
    args = ap.parse_args(argv)

    known = set(SPECS) | set(PLAIN) | set(GENERATED)
    unknown = sorted(p.stem for p in TABLES.glob("*.tex") if p.stem not in known)
    if unknown:
        print(f"note: no grading decision recorded for {', '.join(unknown)}")

    changed = 0
    for stem, columns in SPECS.items():
        path = TABLES / f"{stem}.tex"
        if not path.exists():
            print(f"missing: {path}")
            continue
        before = path.read_text()
        after = grade_latex(ungrade_latex(before), columns)
        cells = after.count(r"\cellcolor") - before.count(r"\cellcolor")
        if after == before:
            print(f"  {stem}: already graded")
            continue
        changed += 1
        print(f"  {stem}: {cells} cells shaded")
        if not args.check:
            path.write_text(after)
    print(f"\n{changed} table(s) {'would change' if args.check else 'updated'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
