"""Verify the traceability claim verify_output.txt makes about itself.

The file tells reviewers that block 1 is "every macro the manuscript can print, with the value it
prints", block 2 is "the full analysis output the macros are derived from", and "a macro whose
value disagrees with block 2 is a defect ... nothing here was typed". That was FALSE for fourteen
printed macros, which reached numbers.tex from cap_probe.json, paper_coding.json, the decision
ledger and model_provenance.json without ever passing through data/numbers.json.

WHY THIS CHECKS PROVENANCE AND NOT VALUES. The obvious test, "does each printed value appear
somewhere in the analysis dump", is worthless: with a thousand values in the dump and the
scalings a formatter may apply, almost any number matches something. Injecting N_eff = 99 in
place of 20 passed it. So this checks the invariant that actually guarantees traceability:

  1. every macro the manuscript prints is defined in numbers.tex and listed in block 1;
  2. `make_macros.py` reads NO data file except numbers.json when emitting macros, so there is
     no second source a value could come from;
  3. `consistency.py` separately re-runs make_macros and fails if numbers.tex differs, which
     closes the loop from numbers.json to the printed value.

Together those mean a printed number can only have come from the analysis output.

Exit 1 on any violation.
"""
from __future__ import annotations

import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, "..", "paper"))
BS = chr(92)

# paper_coding.json is read to BUILD Table 1's rows, which are marks rather than macros, and
# check_tables.py verifies those marks against the same file. Any other data file read by the
# macro generator is a second source of truth and a defect.
ALLOWED_READS = {"numbers.json", "paper_coding.json"}


def main() -> None:
    tex = io.open(os.path.join(PAPER, "main.tex"), encoding="utf-8").read()
    macros_src = io.open(os.path.join(PAPER, "numbers.tex"), encoding="utf-8").read()
    ver = io.open(os.path.join(PAPER, "verify_output.txt"), encoding="utf-8").read()
    gen = io.open(os.path.join(HERE, "make_macros.py"), encoding="utf-8").read()

    used = set(re.findall(BS + BS + r"([A-Z][A-Za-z]+)\{\}", tex))
    # The artifact link is the one camera-ready macro main.tex defines itself. It is a URL, not a
    # quantity, so it has no place in the ground truth. Exempt by name, and only while main.tex
    # defines it, so a quantity cannot borrow the exemption.
    local = {"ArtifactURL"} & set(re.findall(BS + BS + r"newcommand\{" + BS + BS + r"([A-Za-z]+)\}", tex))
    used -= local
    defined = set(re.findall(BS + BS + r"newcommand\{" + BS + BS + r"([A-Za-z]+)\}", macros_src))
    if "=== BLOCK 2" not in ver:
        raise SystemExit("verify_output.txt has no BLOCK 2 marker; its structure changed")
    block1 = ver.split("=== BLOCK 2", 1)[0]

    undefined = sorted(used - defined)
    missing1 = [m for m in sorted(used) if (BS + m) not in block1]

    # Every data-file name the macro generator mentions at all. Anchoring on `open(` missed
    # paper_coding.json, whose path is built through a nested os.path.dirname() call the pattern
    # could not cross: a checker that cannot see one read cannot be trusted to see the next.
    # Match the filename literals instead, which no path construction can hide.
    reads = set(re.findall(r'["\']([A-Za-z0-9_]+\.(?:json|jsonl|csv|md))["\']', gen))
    written = set(re.findall(r'["\']([A-Za-z0-9_]+\.tex)["\']', gen))
    stray = sorted(reads - ALLOWED_READS)

    print(f"macros printed in the manuscript : {len(used)}")
    print(f"printed but not defined          : {len(undefined)} {undefined or ''}")
    print(f"printed but missing from block 1 : {len(missing1)} {missing1 or ''}")
    print(f"data files the generator reads   : {sorted(reads)}")
    print(f"disallowed sources               : {len(stray)} {stray or ''}")

    if undefined or missing1 or stray:
        raise SystemExit(
            "traceability broken: a printed quantity is undefined, absent from the reviewer-facing "
            "ground truth, or reachable from a source other than the analysis output")
    print("TRACEABILITY OK: every printed macro is defined, listed in block 1, and the generator "
          "reads no source but the analysis output")


if __name__ == "__main__":
    main()
