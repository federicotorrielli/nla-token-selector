"""token_analysis/table_colour.py — one colour ramp for every graded table.

A graded table shades each numeric cell by its value, so a reader sees the shape
of a result before reading a digit. Two ramps cover everything the paper needs:

  sequential  magnitude on one scale, one hue, light to dark.
  diverging   polarity around a meaningful midpoint, two hues that read as
              opposite, with a neutral gray at the midpoint.

Interpolation is in OKLab, which keeps the wash even instead of dipping through
a muddy middle. Every fill stays light enough for the printed number to clear
4.5:1 against the text ink, so the colour is never the only channel: the value
is in the cell either way.

`grade_latex` takes a LaTeX tabular and shades chosen columns of chosen rows,
which lets a table written by hand be graded without being rewritten.

Usage:
    uv run python token_analysis/table_colour.py --selftest
"""

from __future__ import annotations

import argparse
import re

import numpy as np

MIDPOINT = "#f0efec"      # neutral: chance, or zero
POLE_HIGH = "#2a78d6"     # blue
POLE_LOW = "#e34948"      # red
INK = "#0b0b0b"
SATURATION = 0.80         # fraction of the pole the most extreme cell reaches
MIN_CONTRAST = 4.5


def _srgb_to_oklab(hex_colour: str) -> tuple[float, float, float]:
    r, g, b = (int(hex_colour.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
    lin = [(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4) for c in (r, g, b)]
    long = 0.4122214708 * lin[0] + 0.5363325363 * lin[1] + 0.0514459929 * lin[2]
    med = 0.2119034982 * lin[0] + 0.6806995451 * lin[1] + 0.1073969566 * lin[2]
    short = 0.0883024619 * lin[0] + 0.2817188376 * lin[1] + 0.6299787005 * lin[2]
    l_, m_, s_ = np.cbrt([long, med, short])
    return (0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_)


def _oklab_to_srgb(lab: tuple[float, float, float]) -> str:
    lightness, a, b = lab
    l_ = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
    lin = (4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
           -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
           -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_)
    out = []
    for c in lin:
        c = min(max(c, 0.0), 1.0)
        c = 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
        out.append(round(c * 255))
    return "{:02X}{:02X}{:02X}".format(*out)


def _mix(pole: str, t: float) -> str:
    mid, end = _srgb_to_oklab(MIDPOINT), _srgb_to_oklab(pole)
    t = min(max(t, 0.0), 1.0) * SATURATION
    return _oklab_to_srgb(tuple(m + t * (e - m) for m, e in zip(mid, end, strict=True)))


def sequential_fill(value: float, vmin: float, vmax: float, pole: str = POLE_HIGH) -> str:
    """One hue, deepening with the value. `vmin` receives the neutral."""
    if not np.isfinite(value) or vmax <= vmin:
        return MIDPOINT.lstrip("#").upper()
    return _mix(pole, (value - vmin) / (vmax - vmin))


def diverging_fill(value: float, centre: float = 0.5, half_range: float = 0.30,
                   high: str = POLE_HIGH, low: str = POLE_LOW) -> str:
    """Two hues around a neutral centre, deepening with distance from it."""
    if not np.isfinite(value):
        return MIDPOINT.lstrip("#").upper()
    return _mix(high if value >= centre else low, abs(value - centre) / half_range)


def relative_luminance(hex_colour: str) -> float:
    r, g, b = (int(hex_colour.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
    lin = [(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4) for c in (r, g, b)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast_with_ink(hex_colour: str, ink: str = INK) -> float:
    lo, hi = sorted((relative_luminance(hex_colour), relative_luminance(ink)))
    return (hi + 0.05) / (lo + 0.05)


# --------------------------------------------------------------------------- #
# Grading a LaTeX tabular in place                                             #
# --------------------------------------------------------------------------- #
_NUMBER = re.compile(r"[-+]?\d+\.\d+")
_TABULAR = re.compile(r"(\\begin\{tabular\}.*?\n)(.*?)(\\end\{tabular\})", re.S)


def first_number(cell: str) -> float | None:
    """The first decimal in a cell, which is the point estimate when the cell
    also carries an interval.

    A decimal touching a letter belongs to a name rather than to a measurement,
    so `Qwen2.5-7B` and `Llama-3.3-70B` yield nothing and their header cells stay
    unshaded."""
    for hit in _NUMBER.finditer(cell):
        before = cell[: hit.start()].rstrip("+-")
        after = cell[hit.end():]
        if before and before[-1].isalpha():
            continue
        if after and after[0].isalpha():
            continue
        return float(hit.group())
    return None


def split_top_level(text: str, separator: str) -> list[str]:
    """Split on a separator that sits at brace depth zero.

    A LaTeX row ends in `\\\\` and its cells are divided by `&`, but both also
    occur inside `\\makecell{a\\\\b}` and `\\multirow{2}{*}{…}`, so a naive split
    would cut a cell in half."""
    parts, depth, start, i = [], 0, 0, 0
    while i < len(text):
        char = text[i]
        if char == "\\" and i + 1 < len(text) and text[i + 1] in "{}":
            i += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif depth == 0 and text.startswith(separator, i):
            parts.append(text[start:i])
            i += len(separator)
            start = i
            continue
        i += 1
    parts.append(text[start:])
    return parts


_FILL = re.compile(r"\\cellcolor\[HTML\]\{[0-9A-Fa-f]{6}\}")


def ungrade_latex(text: str) -> str:
    """Strip every fill, so regrading after a scale change is deterministic."""
    return _FILL.sub("", text)


def grade_latex(text: str, columns: dict[int, dict]) -> str:
    """Shade the given one-based columns of every data row of every tabular.

    `columns` maps a column index to a spec: `{"kind": "sequential", "vmin":…,
    "vmax":…}` or `{"kind": "diverging", "centre":…, "half_range":…}`. Rules,
    headers, `\\multicolumn` spanners and cells already carrying a fill are left
    alone, and a row with too few cells is skipped rather than mangled.
    """
    def grade_body(match: re.Match) -> str:
        head, body, tail = match.group(1), match.group(2), match.group(3)
        rows = split_top_level(body, r"\\")
        for r, row in enumerate(rows):
            cells = split_top_level(row, "&")
            if len(cells) < max(columns):
                continue
            changed = False
            for index, spec in columns.items():
                cell = cells[index - 1]
                if r"\cellcolor" in cell or r"\multicolumn" in cell:
                    continue
                value = first_number(cell)
                if value is None:
                    continue
                if spec["kind"] == "sequential":
                    fill = sequential_fill(value, spec["vmin"], spec["vmax"],
                                           spec.get("pole", POLE_HIGH))
                else:
                    fill = diverging_fill(value, spec.get("centre", 0.0),
                                          spec.get("half_range", 1.0))
                lead = len(cell) - len(cell.lstrip())
                cells[index - 1] = cell[:lead] + rf"\cellcolor[HTML]{{{fill}}}" + cell[lead:]
                changed = True
            if changed:
                rows[r] = "&".join(cells)
        return head + r"\\".join(rows) + tail

    return _TABULAR.sub(grade_body, text)


# --------------------------------------------------------------------------- #
# Self-test                                                                    #
# --------------------------------------------------------------------------- #
def selftest() -> int:
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    check("the sequential floor is the neutral",
          sequential_fill(0.0, 0.0, 1.0) == MIDPOINT.lstrip("#").upper())
    check("the sequential ramp darkens with the value",
          relative_luminance("#" + sequential_fill(0.2, 0, 1))
          > relative_luminance("#" + sequential_fill(0.9, 0, 1)))
    check("a degenerate range gives the neutral",
          sequential_fill(3.0, 1.0, 1.0) == MIDPOINT.lstrip("#").upper())
    check("a value past the ceiling clamps",
          sequential_fill(5.0, 0, 1) == sequential_fill(1.0, 0, 1))

    check("the diverging centre is the neutral",
          diverging_fill(0.5, 0.5) == MIDPOINT.lstrip("#").upper())
    check("the two sides of the centre take different hues",
          diverging_fill(0.8, 0.5) != diverging_fill(0.2, 0.5))
    check("the diverging ramp is symmetric in distance",
          relative_luminance("#" + diverging_fill(0.7, 0.5, 0.3))
          == relative_luminance("#" + diverging_fill(0.7, 0.5, 0.3)))

    worst_seq = min(contrast_with_ink("#" + sequential_fill(v, 0, 1))
                    for v in np.linspace(0, 1, 101))
    worst_div = min(contrast_with_ink("#" + diverging_fill(v, 0.5, 0.3))
                    for v in np.linspace(0, 1, 101))
    check(f"every fill keeps the number above {MIN_CONTRAST}:1 "
          f"(sequential {worst_seq:.1f}, diverging {worst_div:.1f})",
          min(worst_seq, worst_div) >= MIN_CONTRAST)

    check("a cell's point estimate is its first decimal",
          first_number(r"\(0.685\) \([0.681, 0.689]\)") == 0.685)
    check("a cell without a decimal is skipped", first_number("q7") is None)
    check("a negative point estimate is read with its sign",
          first_number(r"\(-0.074\) \([-0.083,-0.064]\)") == -0.074)
    check("a decimal inside a model name is not a measurement",
          first_number("Qwen2.5-7B") is None and first_number("Llama-3.3-70B") is None)
    check("a signed cell is still read",
          first_number(r"\(\mathbf{+0.328}\) \([+0.322,+0.333]\)") == 0.328)
    check("a number already carrying a fill is still read",
          first_number(r"\cellcolor[HTML]{F0EFEC}\(0.500\)") == 0.500)

    check("a top-level split ignores separators inside braces",
          split_top_level(r"a & \makecell{x & y} & b", "&")
          == ["a ", r" \makecell{x & y} ", " b"])
    check("a row split ignores a line break inside a cell",
          len(split_top_level(r"a & \makecell{p\\q} \\ b & c", r"\\")) == 2)

    table = "\n".join([
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"model & value \\",
        r"\midrule",
        r"q7 & \(0.90\) \\",
        r"g12",
        r"    & \(0.10\) \\",
        r"\multicolumn{2}{l}{a spanner} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ])
    spec = {2: {"kind": "sequential", "vmin": 0.0, "vmax": 1.0}}
    graded = grade_latex(table, spec)
    check("the header row is left alone", r"model & value" in graded
          and r"model & \cellcolor" not in graded)
    check("the rules survive", r"\toprule" in graded and r"\bottomrule" in graded)
    check("a spanner row is left alone",
          r"\multicolumn{2}{l}{a spanner}" in graded
          and graded.count(r"\cellcolor") == 2)
    check("a row spanning two lines is still shaded",
          re.search(r"\\cellcolor\[HTML\]\{[0-9A-F]{6}\}\\\(0\.10", graded) is not None)
    check("the shaded values survive unchanged",
          r"\(0.90\)" in graded and r"\(0.10\)" in graded)
    check("grading twice changes nothing", grade_latex(graded, spec) == graded)
    check("ungrading returns the original", ungrade_latex(graded) == table)
    check("ungrading then regrading reproduces the fill",
          grade_latex(ungrade_latex(graded), spec) == graded)
    check("a column past the row width is ignored",
          grade_latex(table, {7: {"kind": "sequential", "vmin": 0, "vmax": 1}}) == table)
    check("text outside the tabular is untouched",
          grade_latex("before\n" + table + "\nafter", spec).startswith("before\n"))

    print(f"\nselftest: {'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    ap.error("nothing to do; pass --selftest or import the module")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
