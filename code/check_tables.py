"""Verify every table's CONTENT against its source, not just its shape.

The consistency gate checks that the factor grid has the right SIZE. It cannot see whether the
level values printed in Table 2 are the level values the code actually ran: a transposed list or a
stale number would pass every numeric gate, because no number in the paper traces to them.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys

BS = chr(92)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import backtest as B  # noqa: E402

fails = []


def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        fails.append(msg)


tex = open(os.path.join(ROOT, "paper", "main.tex"), encoding="utf-8", newline="").read()
grid_tbl = tex[tex.index("label{tab:grid}"): tex.index("end{table}", tex.index("label{tab:grid}"))]

print("Table 2: printed factor levels against the code that ran")
for name, vals in (("model", ["Qwen2.5-7B", "Mistral-7B-v0.2", "Phi-3-mini"]),
                   ("truncation", [str(v) for v in B.TRUNCS]),
                   ("threshold", [f"{v:.2f}" for v in B.THRESHES]),
                   ("holding", [str(v) for v in B.HOLDS])):
    for v in vals:
        check(v in grid_tbl, f"Table 2 shows {name} level {v!r}")
# the code's own levels, so a change in backtest.py without a change in the table is caught
check(B.TRUNCS == [1200, 4000], f"code truncations are {B.TRUNCS}, table says 1200, 4000")
check(B.HOLDS == [1, 3, 5], f"code holds are {B.HOLDS}, table says 1, 3, 5")
check(B.THRESHES == [0.0, 0.10, 0.20], f"code thresholds are {B.THRESHES}")
check(B.ENTRIES == ["open", "close"], f"code entries are {B.ENTRIES}")
check(B.NEUTRALS == ["none", "market"], f"code neutralisations are {B.NEUTRALS}")
check(B.PARSES == ["strict", "lenient"], f"code parsers are {B.PARSES}")
check(B.SIGNALS == ["level", "xrank", "demean"], f"code signals are {B.SIGNALS}")
for m in B.MODELS:
    short = {"qwen2.5:7b-instruct": "Qwen2.5-7B",
             "mistral:7b-instruct-v0.2-q4_0": "Mistral-7B-v0.2",
             "phi3:mini": "Phi-3-mini"}[m]
    check(short in grid_tbl, f"code model {m} appears in Table 2 as {short}")

print("\nTable 3: compliance cells against numbers.json")
comp = json.load(open(os.path.join(ROOT, "data", "numbers.json"), encoding="utf-8"))["compliance"]
gen = open(os.path.join(ROOT, "paper", "compliance.tex"), encoding="utf-8").read()
short = {"qwen2.5:7b-instruct": "Qwen2.5-7B", "mistral:7b-instruct-v0.2-q4_0": "Mistral-7B-v0.2",
         "phi3:mini": "Phi-3-mini"}
for row in gen.splitlines():
    if "&" not in row or "prompt" in row:
        continue
    cells = [c.strip() for c in row.replace(BS + BS, "").split("&")]
    if not cells[0]:
        continue            # the column-header row, whose first cell is empty
    model = [k for k, v in short.items() if v == cells[0]]
    check(bool(model), f"Table 3 row {cells[0]!r} maps to a model in numbers.json")
    if model:
        for i, p in enumerate(["P1", "P2", "P3"]):
            # Two decimals, matching make_macros. Integer rounding printed 99.98 as "100" and
            # 0.15 as "0", which made two sentences in the body literally false.
            want = f"{100 * comp[f'{model[0]}|{p}']:.2f}"
            check(cells[i + 1] == want,
                  f"Table 3 {cells[0]} {p}: shows {cells[i+1]}, numbers.json says {want}")

print("\nTable 1: coded marks against paper_coding.json")
coding = json.load(open(os.path.join(ROOT, "data", "paper_coding.json"), encoding="utf-8"))
gen1 = open(os.path.join(ROOT, "paper", "coding.tex"), encoding="utf-8").read()
cols = ["prompt_text", "holding", "entry_timing", "pipeline_sensitivity", "costs"]
for p in coding["papers"]:
    check(p["key"] in gen1, f"Table 1 cites {p['key']}")
    n_yes = sum(1 for c in cols if p[c] == "yes")
    if all(p[c] == "unretrieved" for c in cols):
        continue
    line = [l for l in gen1.splitlines() if p["key"] in l]
    if line:
        check(line[0].count(BS + "checkmark") == n_yes,
              f"Table 1 {p['short']}: {line[0].count(BS+'checkmark')} ticks, coding has {n_yes}")

print("\nTables 4 and 5: every cell is a macro, none is a literal")
for lab in ("tab:refconfigs", "tab:multiplicity"):
    blk = tex[tex.index("label{" + lab + "}"):
              tex.index("end{table}", tex.index("label{" + lab + "}"))]
    body = blk[blk.index("midrule"):] if "midrule" in blk else blk
    body = re.sub(re.escape(BS) + r"[A-Za-z]+\{\}", "", body)
    body = re.sub(r"\$[^$]*\$", "", body)
    stray = re.findall(r"(?<![A-Za-z0-9.])\d[\d,.]*", body)
    check(not stray, f"{lab} has no hard-coded numbers outside macros: {stray}")

print("\nCitations")
bib = open(os.path.join(ROOT, "paper", "refs.bib"), encoding="utf-8").read()
defined = set(re.findall(r"@\w+\{([^,]+),", bib))
used = set()
# Match every natbib/acmart citation form, not just \cite. Matching only \cite made this check
# fail OPEN in one direction and CLOSED in the other: a \citet reference is invisible to it, so a
# paper written entirely in \citet would report "0 cited" and every key as unused, which reads
# exactly like a real defect, while a genuinely uncited key could hide behind a \citet elsewhere.
for m in re.finditer(re.escape(BS) + r"cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", tex):
    used.update(k.strip() for k in m.group(1).split(","))
check(not (used - defined), f"every cited key is defined: missing {sorted(used - defined)}")
check(not (defined - used), f"every defined key is cited: unused {sorted(defined - used)}")
entries = re.split(r"\n(?=@)", bib)
for e in entries:
    m = re.match(r"@\w+\{([^,]+),", e.lstrip())
    if m:
        # Walk the CONTIGUOUS comment block above the entry, not a fixed window. A six-line window
        # reported a false failure on the one entry whose verification note runs to seven lines,
        # which is the entry that was corrected most carefully.
        idx = bib.index(e)
        block, lines = [], bib[:idx].rstrip().splitlines()
        for line in reversed(lines):
            if line.startswith("%"):
                block.append(line)
            elif line.strip() == "":
                continue
            else:
                break
        check(any("VERIFIED" in l for l in block),
              f"{m.group(1)} carries a VERIFIED comment ({len(block)} comment lines found)")
bbl = os.path.join(ROOT, "paper", "main.bbl")
if os.path.exists(bbl):
    check("VERIFIED" not in open(bbl, encoding="utf-8").read(),
          "no verification note leaked into the printed bibliography")
print(f"\n  {len(defined)} references, {len(used)} cited")

print()
if fails:
    print(f"TABLE/CITATION CHECK FAILED: {len(fails)}")
    sys.exit(1)
print("TABLE/CITATION CHECK PASSED")
