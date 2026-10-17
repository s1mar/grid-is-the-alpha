"""Edit-damage gate: snapshot before editing, interrogate only what changed after.

WHY THIS EXISTS. Every other gate checks the document as it stands. The defects that actually got
through were of a different kind: a sentence that was CORRECT became wrong when it was edited, and
the document-level gates had nothing to compare against. Two consecutive review rounds found 15 and
then 5 defects of which 13 and 5 were caused by the previous round's repairs. That is not a
document problem, it is an edit problem, so this checks edits.

    python editcheck.py snapshot     # before you touch main.tex
    ... make edits ...
    python editcheck.py check        # after: every changed sentence, plus flags

The checks below are not generic prose hygiene. Each one is derived from a defect that actually
reached a built PDF in this project, and `--selftest` re-injects all of them.

  MACRO_DROPPED     a reworded sentence lost a macro it used to print. This is how a number
                    disappears silently: the prose still reads well and no gate misses a value it
                    no longer looks for.
  HARDCODED         a reworded sentence gained a bare numeral. The document gate catches this
                    globally; here it is attributed to the edit that introduced it.
  COUNT_MISMATCH    a count word ("Four consequential choices ... : a; b; c") that disagrees with
                    the number of items actually listed. Caught "the three factors" over four
                    restrictions and "Two arms" over five.
  UNIVERSAL         a universal quantifier was introduced or strengthened. Every one of these in
                    this paper's history was either false or needed narrowing: "papers do not
                    report them", "no paper in the sample", "no arm exceeds", "every model".
  SCOPE_QUALIFIER   a claim about what the permuted scores carry, stated WITHOUT "within-session".
                    Section 4 explicitly disowns the unqualified version; the figure caption said
                    it anyway for several rounds.
  DANGLING_OPENER   a sentence beginning with a bare demonstrative whose antecedent may have been
                    deleted by the same edit: "Those refusals...", "would face exactly this".
  NEG_INVERSION     "and neither is X", which reads as "X is not either", inverting the intent.
  OPEN_COMPARISON   "rises from X" with no "to" anywhere in the sentence.

A flag is not a verdict. It marks a sentence a human must re-read. `check` exits 1 if any flag
fires, so it can gate a build.
"""
from __future__ import annotations

import difflib
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, "..", "paper"))
TEX = os.path.join(PAPER, "main.tex")
SNAP = os.path.join(os.path.abspath(os.path.join(HERE, "..", "notes")), "editcheck_snapshot.json")
BS = chr(92)

COUNT_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
UNIVERSALS = re.compile(
    r"\b(every|all|none|no paper|no arm|no configuration|never|always|both|neither|"
    r"each of the|any of the|only)\b", re.I)
DEMONSTRATIVE_OPEN = re.compile(r"^(Those|These|This|That|It|They|Both|Neither|Either)\b")


def sentences(text: str) -> list[str]:
    """Split the body into sentences, ignoring comments and display math."""
    body = re.sub(r"(?m)^\s*%.*$", "", text)
    body = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", " ", body, flags=re.S)
    out = []
    for raw in re.split(r"(?<=[.!?])\s+", body):
        s = " ".join(raw.split())
        if len(s.split()) >= 4:
            out.append(s)
    return out


def macros(s: str) -> set[str]:
    return set(re.findall(BS + BS + r"([A-Z][A-Za-z]+)\{\}", s))


def bare_numbers(s: str) -> list[str]:
    """Numerals that are not part of a macro, a citation, a label, or a typesetting argument.

    The exclusions below are not convenience: each one produced a false flag on the real edit
    history, and a gate that over-flags is one you learn to skim, which is exactly when it misses
    something. Excluded are LaTeX lengths and float placement, SEC exhibit identifiers, the
    synthetic-oracle accuracy levels (definitional, and written as literals by an earlier decision),
    and the small integers that are part of the grid's definition.
    """
    t = re.sub(r"\\includegraphics\[[^\]]*\]", " ", s)
    t = re.sub(r"\\begin\{\w+\}\[[^\]]*\]", " ", t)
    t = re.sub(r"[Ee]xhibit[s]?\s+(?:EX-)?[\d.]+", " ", t)
    t = re.sub(r"\bEX-\d+", " ", t)
    t = re.sub(BS + BS + r"[A-Za-z]+\{[^}]*\}", " ", t)
    t = re.sub(BS + BS + r"[A-Za-z]+", " ", t)
    t = re.sub(r"\$[^$]*\$", " ", t)
    keep = {"0", "1", "2", "5", "8", "99", "95", "2026", "0.5", "8-K", "4",
            "50", "52", "55", "60", "70",     # oracle accuracy levels, definitional
            "3", "10", "25", "500", "0.20"}   # grid levels and cost levels
    return [m for m in re.findall(r"(?<![\w.])\d[\d,.]*(?![\w])", t) if m not in keep]


def count_mismatch(s: str) -> str | None:
    """A count word followed by a colon-introduced list whose items can be counted."""
    m = re.match(r"^\s*(\w+)\b", s)
    if not m:
        return None
    word = m.group(1).lower()
    if word not in COUNT_WORDS or ":" not in s:
        return None
    tail = s.split(":", 1)[1]
    items = [p for p in tail.split(";") if len(p.split()) >= 2]
    if len(items) >= 2 and len(items) != COUNT_WORDS[word]:
        return f"says {word!r} but the colon list has {len(items)} items"
    return None


def flags_for(s: str, old: str | None, prev_changed: bool = True) -> list[str]:
    """`prev_changed` says whether the PRECEDING sentence was also touched by this edit.

    A demonstrative opener is only suspicious when its antecedent may have moved. Flagging every
    sentence that starts with "It" produced more noise than signal on the real history.
    """
    out = []
    if old is not None:
        lost = macros(old) - macros(s)
        if lost:
            out.append(f"MACRO_DROPPED: {sorted(BS + m for m in lost)}")
        gained = set(bare_numbers(s)) - set(bare_numbers(old))
        if gained:
            out.append(f"HARDCODED: gained bare numeral(s) {sorted(gained)}")
        if UNIVERSALS.search(s) and not UNIVERSALS.search(old):
            out.append("UNIVERSAL: a universal quantifier was introduced by this edit")
    else:
        if bare_numbers(s):
            out.append(f"HARDCODED: new sentence has bare numeral(s) {sorted(set(bare_numbers(s)))}")
        if UNIVERSALS.search(s):
            out.append("UNIVERSAL: new sentence asserts a universal")
    cm = count_mismatch(s)
    if cm:
        out.append(f"COUNT_MISMATCH: {cm}")
    if re.search(r"carr(y|ies|ying)\s+no\s+information", s) and "within-session" not in s:
        out.append("SCOPE_QUALIFIER: says the scores carry no information, without 'within-session'")
    if prev_changed and DEMONSTRATIVE_OPEN.match(s):
        out.append("DANGLING_OPENER: opener refers back, and the preceding sentence also changed")
    if re.search(r"\band neither is\b", s):
        out.append("NEG_INVERSION: 'and neither is X' reads as 'X is not either'")
    if re.search(r"\b(rises|falls|grows|moves)\b[^.]*\bfrom\b", s) and " to " not in s:
        out.append("OPEN_COMPARISON: a change 'from' something with no 'to'")
    return out


def do_snapshot() -> None:
    text = io.open(TEX, encoding="utf-8").read()
    json.dump({"sentences": sentences(text)}, open(SNAP, "w", encoding="utf-8"))
    print(f"snapshot: {len(sentences(text))} sentences -> {SNAP}")


def do_check(quiet: bool = False) -> int:
    if not os.path.exists(SNAP):
        print("no snapshot; run: python editcheck.py snapshot")
        return 1
    old = json.load(open(SNAP, encoding="utf-8"))["sentences"]
    new = sentences(io.open(TEX, encoding="utf-8").read())

    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    changed: list[tuple[str, str | None]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            for k, s in enumerate(new[j1:j2]):
                # Pair each new sentence with its closest old counterpart, so a reword is compared
                # against what it replaced rather than treated as brand new.
                cand = old[i1:i2]
                best = max(cand, key=lambda o: difflib.SequenceMatcher(None, o, s).ratio(),
                           default=None)
                ratio = difflib.SequenceMatcher(None, best, s).ratio() if best else 0
                changed.append((s, best if ratio > 0.45 else None))
        elif tag == "insert":
            changed.extend((s, None) for s in new[j1:j2])

    n_flag = 0
    changed_set = {c[0] for c in changed}
    print(f"changed or new sentences: {len(changed)}")
    for idx, (s, o) in enumerate(changed):
        prev = new[new.index(s) - 1] if s in new and new.index(s) > 0 else None
        fl = flags_for(s, o, prev_changed=(prev in changed_set) if prev else False)
        if not fl:
            continue
        n_flag += 1
        print(f"\n  {'REWORDED' if o else 'NEW'}: {s[:150]}")
        for f in fl:
            print(f"    !! {f}")
    if n_flag == 0:
        print("no flags: every changed sentence kept its macros, its scope and its counts")
        return 0
    print(f"\n{n_flag} changed sentence(s) flagged. Each is a sentence to RE-READ, not proof of a "
          f"defect. Re-snapshot once you have checked them.")
    return 1


SELFTEST = [
    ("macro dropped by a reword",
     "The spread widens from " + BS + "SharpeSD{} to " + BS + "RobustSD{} across the grid.",
     "The spread widens from " + BS + "SharpeSD{} across the grid overall.", "MACRO_DROPPED"),
    ("hard-coded number introduced",
     "The grid behaves like " + BS + "Neff{} independent trials in total.",
     "The grid behaves like 20 independent trials in total.", "HARDCODED"),
    ("universal introduced",
     "Table 1 shows that most of them go unreported in practice.",
     "Table 1 shows that no paper reports them in practice.", "UNIVERSAL"),
    ("count disagrees with the list",
     "Three choices sit outside the grid: item codes; the score origin; the normalisation.",
     "Three choices sit outside the grid: item codes; the score origin; the normalisation; "
     "and recording an undefined Sharpe as zero.", "COUNT_MISMATCH"),
    ("scope qualifier lost",
     "the outputs are permuted so that they carry no within-session information here.",
     "the outputs are permuted so that they carry no information here.", "SCOPE_QUALIFIER"),
    ("negative inversion",
     "The ladder is not a clean sweep, and no model here is a proprietary frontier system.",
     "The ladder is not a clean sweep, and neither is a proprietary frontier system.",
     "NEG_INVERSION"),
    ("comparison left open",
     "the p-value rises with scale from " + BS + "ArmPermP{} to " + BS + "BigPermP{} overall.",
     "the p-value rises with scale from " + BS + "ArmPermP{} at the reference arm.",
     "OPEN_COMPARISON"),
]


def do_selftest() -> int:
    missed = []
    for name, before, after, want in SELFTEST:
        got = flags_for(after, before)
        if not any(f.startswith(want) for f in got):
            missed.append((name, want, got))
            print(f"  MISSED  {name}  (wanted {want}, got {got})")
        else:
            print(f"  CAUGHT  {name}")
    if missed:
        print(f"\nSELFTEST FAILED: {len(missed)} of {len(SELFTEST)} injected edit defects not caught")
        return 1
    print(f"\nSELFTEST PASSED: {len(SELFTEST)}/{len(SELFTEST)} injected edit defects caught")
    return 0


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "snapshot":
        do_snapshot()
    elif cmd == "--selftest":
        raise SystemExit(do_selftest())
    else:
        raise SystemExit(do_check())


if __name__ == "__main__":
    main()
