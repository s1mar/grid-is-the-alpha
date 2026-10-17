"""Write paper/verify_output.txt: the ground truth a reviewer needs to reconcile every cell.

The panel driver ships this file alongside the manuscript. Without it a numeric reviewer cannot
check a single number and correctly says so, which is the most expensive way to spend a round.
Generated from data/numbers.json so it can never drift from what the paper's macros say.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
PAPER = os.path.abspath(os.path.join(HERE, "..", "paper"))


def flatten(obj, prefix=""):
    out = []
    if isinstance(obj, dict):
        for k in sorted(obj):
            out += flatten(obj[k], f"{prefix}{k}.")
    elif isinstance(obj, list):
        out.append((prefix.rstrip("."), json.dumps(obj)[:200]))
    else:
        out.append((prefix.rstrip("."), obj))
    return out


def main() -> None:
    n = json.load(open(os.path.join(DATA, "numbers.json"), encoding="utf-8"))
    macros = {}
    for line in open(os.path.join(PAPER, "numbers.tex"), encoding="utf-8"):
        if line.startswith("\\newcommand"):
            name = line[line.index("{\\") + 2: line.index("}")]
            val = line[line.index("}{") + 2: line.rindex("}")]
            macros[name] = val

    lines = [
        "GROUND TRUTH FOR THE ICAIF '26 SUBMISSION",
        "",
        "Two blocks. The first is every macro the manuscript can print, with the value it prints.",
        "The second is the full analysis output the macros are derived from. A number in the paper",
        "that does not appear in block 1 is a defect; a macro whose value disagrees with block 2 is",
        "a defect. Everything is produced by code/analyze.py from the grid and by",
        "code/make_macros.py from that; nothing here was typed.",
        "",
        "=== BLOCK 1: macros as printed in the manuscript ===",
    ]
    for k in sorted(macros):
        lines.append(f"  \\{k:<28} = {macros[k]}")

    lines += ["", "=== BLOCK 2: full analysis output (data/numbers.json) ===", ""]
    for k, v in flatten(n):
        lines.append(f"  {k:<52} = {v}")

    path = os.path.join(PAPER, "verify_output.txt")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {path} ({len(lines)} lines, {len(macros)} macros)")


if __name__ == "__main__":
    main()
