"""The gate. Runs every round; a red line here blocks the build.

Design rules this file follows, each of which exists because ignoring it has cost a paper before:
  * every check that locates a region ASSERTS it found one, so a failed anchor cannot pass silently;
  * every positive check is paired with a negative clause naming what must NOT come back;
  * the venue facts are checked, because a number-tracing gate is structurally blind to a city;
  * `--selftest` re-injects historical and hypothetical defects and fails if any is not caught.

    python consistency.py            # gate the current tree
    python consistency.py --selftest # prove the gate catches what it claims to
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
# The released artifact ships DECISIONS_REDACTED.md and not DECISIONS.md, and none of notes/. Only in
# that layout do the authoring-only checks below step aside; the authoring tree runs all of them.
RELEASED = (not os.path.exists(os.path.join(ROOT, "DECISIONS.md"))
            and os.path.exists(os.path.join(ROOT, "DECISIONS_REDACTED.md")))
DATA = os.path.join(ROOT, "data")

FAILS: list[str] = []
CHECKS = {"n": 0}


def check(ok: bool, msg: str) -> bool:
    CHECKS["n"] += 1
    if not ok:
        FAILS.append(msg)
    return ok


def read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def read_raw(path: str) -> str:
    """Read without newline translation, so line endings and stray control bytes survive."""
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read()


# --------------------------------------------------------------------------- venue
# FETCHED 2026-07-31 from icaif2026.org/call-for-papers.html and recorded in VENUE_FACTS.md.
# Each entry is (must appear, [must NOT appear]). The negative clause is the half that catches a
# plausible invention; a positive-only check passes on a paper that says nothing at all.
VENUE = [
    (r"November 14--17,?\s*\n?\s*2026", ["November 14--16", "October 14--17", "November 4--7"]),
    # CAMERA-READY (2026-10-17): the gates are INVERTED from the submission era. The
    # rights block is ACM's verbatim Publication Release Confirmation (2026-10-14), whose city string
    # is "Milan, AA, Italy"; it is kept exactly as issued.
    (r"Milan, AA, Italy", ["Seoul", "Munich", "Milan, Spain", "Rome, Italy", "Milano, Italy"]),
    (r"ICAIF '26", ["ICAIF '25", "ICAIF 2025", "ICAIF '27"]),
    (r"\\copyrightyear\{2026\}", ["\\copyrightyear{2025}", "\\copyrightyear{2027}"]),
    (r"\\documentclass\[sigconf\]\{acmart\}",
     ["\\documentclass[sigconf,anonymous]{acmart}", "[sigconf,review", "\\documentclass[manuscript"]),
    (r"\\setcopyright\{cc\}", ["\\setcopyright{rightsretained}", "\\setcopyright{acmlicensed}"]),
    (r"\\setcctype\{by\}", ["\\setcctype{by-nc"]),
    (r"\\acmDOI\{10\.1145/3858063\.3858869\}", []),
    (r"\\acmISBN\{979-8-4007-3087-0/2026/11\}", []),
    (r"7th ACM International Conference on AI in Finance", []),
    (r"\\pdfsuppressptexinfo=-1", []),
    # Author block as on the four ASE VoRs; the submission-era placeholders must not return. Checked
    # by STRUCTURE, so this shipped file carries no identity strings; the values themselves are
    # checked on the rendered page and in the publisher's metadata form.
    (r"\\author\{(?!Anonymous)[A-Z][^}]+\}", ["Anonymous Author", "Anonymous Institution"]),
    (r"\\orcid\{\d{4}-\d{4}-\d{4}-\d{3}[\dX]\}", []),
    (r"\\institution\{(?!Anonymous)[^}]+\}", []),
    (r"\\country\{[A-Z][^}]+\}", []),
    (r"\\email\{[^@}\s]+@[^}\s]+\.[a-z]{2,}\}", []),
]


# FETCHED 2026-07-31 from the three model cards; recorded in notes/model_facts.md. The paper's
# contamination argument rests on these, and a number-tracing gate cannot see a date in prose.
MODEL_FACTS = [
    (r"October 2023", ["October 2024", "October 2022", "cutoff of 2025"]),
    (r"ten months", ["two years", "three months"]),
    (r"Qwen2\.5-7B", ["Qwen2-7B", "Qwen3-7B"]),
    (r"Mistral-7B-v0\.2", ["Mistral-7B-v0.1", "Mistral-8B"]),
    (r"Phi-3-mini", ["Phi-2-mini", "Phi-4-mini"]),
    # FETCHED 2026-08-03 from github.com/QwenLM/Qwen3.6: "2026-04-16: Qwen3.6-35B-A3B is now
    # availabe on Hugging Face Hub and ModelScope." This date falls INSIDE the sample window, so
    # the paper must say the release-date bound does not cover the scale arm. The negative clause
    # names the claim that must never come back: that all four models predate the window.
    (r"release-date bound covers the grid, not the scale arm",
     ["All four models .{0,40}predate", "all four models .{0,40}predate",
      "four models predate the window"]),
]


def gate_models(tex: str) -> None:
    for pos, negs in MODEL_FACTS:
        check(re.search(pos, tex) is not None, f"model provenance fact missing: /{pos}/")
        for neg in negs:
            check(re.search(neg, tex) is None, f"main.tex states a wrong model fact: {neg!r}")

    # The scale model's release date is PRINTED THROUGH A MACRO, so a check against main.tex would
    # only ever see the macro name. Tie the printed value to the fetched fact instead, and require
    # the manuscript to actually print it. This date falls inside the sample window, which is why
    # the paper's contamination scope depends on it being right.
    prov_path = os.path.join(DATA, "model_provenance.json")
    check(os.path.exists(prov_path), "data/model_provenance.json is missing, so the scale model's "
                                     "release date is not traceable to anything")
    if os.path.exists(prov_path):
        prov = json.load(open(prov_path, encoding="utf-8"))
        rel = prov.get("qwen3.6:latest", {}).get("release")
        check(rel == "2026-04-16",
              f"the scale model's recorded release date is {rel!r}, not the fetched 2026-04-16")
        check("release_source" in prov.get("qwen3.6:latest", {}),
              "the scale model's release date has no recorded source")
        macros_txt = read(os.path.join(HERE, "numbers.tex"))
        m = re.search(r"\\newcommand\{\\ScaleRelease\}\{([^}]*)\}", macros_txt)
        check(m is not None, "numbers.tex does not define ScaleRelease")
        if m:
            check(m.group(1) == rel,
                  f"ScaleRelease prints {m.group(1)!r} but provenance says {rel!r}")
        check("ScaleRelease{}" in tex, "main.tex never prints the scale model's release date")
        # The paper tells reviewers that every printed quantity is generated and appears in
        # verify_output.txt. That was false for exactly this date, which reached numbers.tex
        # straight from the provenance file and so appeared in no ground-truth block. Enforce
        # the full chain, and require it to be visible in the file a reviewer is handed.
        nums = json.load(open(os.path.join(DATA, "numbers.json"), encoding="utf-8"))
        check(nums.get("scale_model", {}).get("release") == rel,
              "numbers.json does not carry scale_model.release, so \\ScaleRelease is printed "
              "from a source no reviewer-facing ground-truth block contains")
        vpath = os.path.join(HERE, "verify_output.txt")
        if os.path.exists(vpath):
            vtxt = read(vpath)
            check(f"\\ScaleRelease" in vtxt and rel in vtxt,
                  "verify_output.txt is stale: it does not show ScaleRelease and its value, so a "
                  "numeric reviewer cannot reconcile the one date the contamination scope rests on")
            check("scale_model.release" in vtxt,
                  "verify_output.txt block 2 has no scale_model.release row, so the printed date "
                  "traces to nothing a reviewer can see")


def gate_venue(tex: str) -> None:
    for pos, negs in VENUE:
        check(re.search(pos, tex) is not None, f"venue fact missing from main.tex: /{pos}/")
        for neg in negs:
            check(neg not in tex, f"main.tex contains a wrong venue fact: {neg!r}")


# --------------------------------------------------------------------------- numbers
def gate_numbers(tex: str, macros: str) -> None:
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", macros))
    check(bool(defined), "numbers.tex defines no macros at all (the anchor found nothing)")
    # Non-numeric macros the camera-ready defines in main.tex itself (the artifact URL). Only
    # these names: a number defined in main.tex would bypass the regeneration diff below.
    local_ok = {"ArtifactURL"}
    local = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", tex)) & local_ok
    defined_all = defined | local

    body = re.sub(r"(?m)^\s*%.*$", "", tex)
    used = set(re.findall(r"\\([A-Za-z]+)\{\}", body))
    latex_builtin = {"LaTeX", "TeX", "BibTeX", "ldots", "checkmark", "textbf", "emph"}
    unknown = sorted(u for u in used - defined_all - latex_builtin if u[0].isupper())
    check(not unknown, f"main.tex uses macros that numbers.tex does not define: {unknown}")

    # Every macro must still equal what analyze.py produced. Regenerating and diffing is the only
    # way to catch a hand-edited numbers.tex, which is the failure this is here for.
    #
    # The regeneration must NOT overwrite the file it is checking. An earlier version did, which
    # made the gate fail on its own side effect the first time and pass the second, and a gate you
    # learn to run twice is a gate you have learned to ignore. Regenerate into a scratch copy.
    sys.path.insert(0, os.path.join(ROOT, "code"))
    import importlib
    import shutil
    import tempfile
    import make_macros
    importlib.reload(make_macros)
    live = os.path.join(HERE, "numbers.tex")
    before = read(live)
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy2(live, os.path.join(tmp, "numbers.tex.bak"))
        try:
            make_macros.main()
            regenerated = read(live)
        finally:
            shutil.copy2(os.path.join(tmp, "numbers.tex.bak"), live)
    check(before == regenerated,
          "paper/numbers.tex does not match what code/make_macros.py generates from "
          "data/numbers.json (someone edited a number by hand, or the analysis moved)")

    # A bare digit in the body that is not part of a macro, a citation, a label or a size is how
    # hard-coded quantities get in. Allow small structural integers and years.
    prose = re.sub(r"\\cite\{[^}]*\}|\\ref\{[^}]*\}|\\label\{[^}]*\}|\\includegraphics[^\n]*", "",
                   body)
    prose = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", "", prose, flags=re.S)
    prose = re.sub(r"\\begin\{CCSXML\}.*?\\end\{CCSXML\}", "", prose, flags=re.S)
    prose = re.sub(r"\\(acm|copyright|ccsdesc)[A-Za-z]*(\[[^\]]*\])?(\{[^}]*\})*", "", prose)
    prose = re.sub(r"\\(documentclass|usepackage|input)(\[[^\]]*\])?\{[^}]*\}", "", prose)
    prose = re.sub(r"\\AtBeginDocument\{.*?\}\}\}", "", prose, flags=re.S)   # the BibTeX kerns
    # An ORCID is an identifier, not a quantity. Only the \orcid argument is exempt, and only
    # in ORCID's own 4-4-4-4 shape, so a number cannot hide inside the command.
    prose = re.sub(r"\\orcid\{\d{4}-\d{4}-\d{4}-\d{3}[\dX]\}", "", prose)
    # Structural integers and quantities that are definitional rather than measured. Every entry
    # here is a deliberate exemption, not a convenience: 76 and 164 are quoted from cited papers,
    # 99.1 is an SEC exhibit number, 252 is trading days per year, 5 is the frozen cost in bp.
    # 50, 52, 60 and 70 are the accuracy levels of the synthetic oracle, definitional in the same
    # sense as the truncation lengths and the deadbands: they are grid levels we chose, not results.
    allowed = {"0", "1", "2", "3", "4", "5", "8", "10", "76", "95", "164", "252", "500", "1934",
               "2023", "2024", "2026", "99", "99.1", "0.10", "0.20", "0.00", "1200", "4000",
               "50", "52", "60", "70",
               # 0.5 is the score origin, a definitional constant of the pipeline in the same sense
               # as the deadband levels, and the Threats section names it as exactly that.
               "0.5"}
    # Digits glued to letters are model names such as Qwen2.5-7B, not measurements, so they are
    # exempt. The test is applied AFTER the full digit run is matched: an inline lookahead makes
    # the match backtrack instead, so "95th" was reported as a stray "9".
    stray = sorted({m.group(0).strip(".,")
                    for m in re.finditer(r"(?<![\\A-Za-z0-9.])\d[\d,.]*", prose)
                    if m.group(0).strip(".,") not in allowed
                    and not prose[m.end():m.end() + 1].isalpha()})
    check(not stray, f"main.tex contains hard-coded quantities outside the macro system: {stray}")


# --------------------------------------------------------------------------- grid
def gate_grid() -> None:
    prereg = read(os.path.join(ROOT, "notes", "preregistration.md"))
    sys.path.insert(0, os.path.join(ROOT, "code"))
    import backtest

    shipped = {
        "model": backtest.MODELS, "prompt": backtest.PROMPTS, "trunc": backtest.TRUNCS,
        "parse": backtest.PARSES, "signal": backtest.SIGNALS, "thresh": backtest.THRESHES,
        "hold": backtest.HOLDS, "entry": backtest.ENTRIES, "weight": backtest.WEIGHTS,
        "neutral": backtest.NEUTRALS,
    }
    sizes = {k: len(v) for k, v in shipped.items()}
    check(sizes == {"model": 3, "prompt": 3, "trunc": 2, "parse": 2, "signal": 3, "thresh": 3,
                    "hold": 3, "entry": 2, "weight": 2, "neutral": 2},
          f"the shipped factor grid does not match the pre-registration: {sizes}")

    n_llm = sizes["model"] * sizes["prompt"] * sizes["trunc"]
    n_post = (sizes["parse"] * sizes["signal"] * sizes["thresh"] * sizes["hold"]
              * sizes["entry"] * sizes["weight"] * sizes["neutral"])
    check(n_llm == 18 and n_post == 432 and n_llm * n_post == 7776,
          f"grid size changed: {n_llm} x {n_post} = {n_llm * n_post}, expected 18 x 432 = 7776")
    check("18 x 432 = 7,776" in prereg or "7,776" in prereg,
          "the pre-registration no longer states the grid size (anchor lost)")
    check(abs(backtest.COST - 0.0005) < 1e-12,
          f"transaction cost is {backtest.COST}, the pre-registration froze 5 bp per side")

    numbers = json.load(open(os.path.join(DATA, "numbers.json"), encoding="utf-8"))
    check(numbers["n_configs"] == 7776,
          f"numbers.json reports {numbers['n_configs']} configurations, expected 7776")


# --------------------------------------------------------------------------- corruption
# Every LaTeX control sequence used in this manuscript, minus its backslash. If one of these
# appears WITHOUT its backslash, a tool ate the escape and LaTeX will typeset the macro name as
# prose without raising an error. This gate exists because exactly that happened: a shell heredoc
# turned "\ref{fig:emax}" into a carriage return followed by "ef{fig:emax}", the build reported
# zero errors, all 79 other checks passed, and the sentence reached a rendered page as
# "Figure effig:emax shows where the curve meets the measurement".
# "item " is deliberately absent: it is an ordinary English word and fired on "the item was
# located". A corruption gate that cries wolf is a gate people learn to skim.
DEBACKSLASHED = ["ref{", "cite{", "label{", "textbf{", "emph{", "section{", "includegraphics",
                 "newcommand{", "begin{", "end{", "eqref{", "caption{", "input{"]


def gate_corruption(raw: str) -> None:
    """`raw` must be read with newline='' so line endings survive to be inspected."""
    lone_cr = [i for i, ch in enumerate(raw) if ch == "\r" and raw[i + 1:i + 2] != "\n"]
    check(not lone_cr, f"a carriage return that is not a line ending is in main.tex at {lone_cr[:3]}"
                       " (a tool ate a backslash: this is the corruption that renders as prose)")
    for ch, name in (("\t", "tab"), ("\x0b", "vertical tab"), ("\x0c", "form feed"),
                     ("\x07", "bell"), ("\x08", "backspace"), ("\x00", "null")):
        check(ch not in raw, f"a {name} character is in main.tex, which means a backslash escape "
                             f"was interpreted instead of written")
    for token in DEBACKSLASHED:
        # A de-backslashed macro is the token preceded by something that is not a backslash and
        # not a letter, so that ordinary words ending in the token cannot fire the check.
        hits = [m.start() for m in re.finditer(r"(?<![\\A-Za-z])" + re.escape(token), raw)]
        real = [h for h in hits if not raw[max(0, h - 60):h].lstrip().startswith("%")]
        check(not real, f"a LaTeX macro appears without its backslash near "
                        f"{[raw[max(0, h-40):h+20] for h in real[:2]]}")


# --------------------------------------------------------------------------- directions
def gate_directions(tex: str) -> None:
    """Gate the DIRECTION of every comparative claim, not only the values inside it.

    A number-tracing gate updates "13.4\\%" to "2.1\\%" without noticing that the sentence around it
    now says the opposite of what the data says. Each rule below pairs a sentence pattern that
    appears in the manuscript with the inequality in numbers.json that has to hold for it to be
    true. A rule whose pattern is not found FAILS, so that rewording the sentence cannot silently
    disarm the check.
    """
    n = json.load(open(os.path.join(DATA, "numbers.json"), encoding="utf-8"))

    def phrase(text: str) -> str:
        """A phrase pattern that survives LaTeX line wrapping."""
        return r"\s+".join(re.escape(w) for w in text.split())

    rules = [
        (phrase("choices account for") + r".{0,80}?"
         + phrase("of the variance and the language model for"),
         n["var_share_post_llm"] > n["var_share_model"],
         "the paper says the post-LLM choices carry more Sharpe variance than the model, "
         "but numbers.json says the opposite"),
        (phrase("re-deciding the pipeline around a fixed model moves it by"),
         n["wb"]["harness_sd"] > n["wb"]["model_sd"],
         "the paper says re-deciding the pipeline moves Sharpe more than swapping the model, "
         "but numbers.json says the opposite"),
        # Was "wrong in opposite directions by more than two orders of magnitude", which is false
        # on the single-trial side: N=1 is 1.30 orders from N_eff, not 2. The claim is now split
        # per direction, so the anchor matches the invariant part.
        (phrase("wrong in opposite directions, the Bonferroni count by more than two"),
         n["n_eff"] < n["n_configs"] / 100.0,
         "the paper claims two orders of magnitude between N_eff and the grid size, "
         "but numbers.json does not support it"),
        # Re-anchored: a compression changed "a version of the identical pipeline whose scores carry
        # no within-session information" to "the identical pipeline with scores carrying no
        # within-session information". Same claim; match the invariant tail.
        (phrase("within-session information reaches the same performance"),
         n["perm_p_best"] > 0.05,
         "the paper says the permutation benchmark matches the real pipeline, but the permutation "
         "p-value in numbers.json is significant"),
        # Re-anchored 2026-08-09: a length compression rewrote the subject from "the selected
        # configuration's apparent performance" to "its apparent performance". Same claim, same
        # number; the anchor pinned the wording and so failed on a correct edit. Match the
        # invariant fraction instead.
        (phrase("six sevenths of its apparent performance is gone"),
         (n["split"]["is_best_sharpe"] - n["split"]["oos_of_is_best"])
         / max(abs(n["split"]["is_best_sharpe"]), 1e-9) > 0.80,
         "the paper says roughly six sevenths of the selected configuration's performance is gone "
         "out of sample, but numbers.json does not support a fraction that large"),
        (phrase("It is not worthless: that still puts it at the"),
         n["split"]["oos_percentile_of_is_best"] > 50.0,
         "the paper says the selected configuration still beats the grid median out of sample, "
         "but numbers.json puts it below the median"),
        # Anchored on the SUBSTANCE, not the sentence: a length trim rewrote "On this grid, then,
        # the cheapest estimator is the one that under-corrects most." to "On this grid the
        # cheapest estimator under-corrects most." Same claim, and the old anchor turned a
        # correct compression into a gate failure. Match the invariant part.
        # Re-anchored: the sentence now names the measured count first and ends "so the cheapest
        # estimator under-corrects most here". Same directional claim.
        (phrase("The cheapest estimator under-corrects most"),
         n["n_implied"] > n["n_eff"],
         "the paper says the participation ratio under-corrected on this grid, but numbers.json "
         "has the implied trial count below it"),
    ]
    for pat, holds, msg in rules:
        found = re.search(pat, tex, re.S) is not None
        check(found, f"a gated directional claim is no longer in the manuscript, so the check "
                     f"could not run: /{pat[:60]}/")
        if found:
            check(holds, msg)

    # The pre-registered falsification condition. If it fires, the paper must say so.
    if n.get("falsified"):
        check("falsif" in tex.lower(),
              "numbers.json reports the pre-registered falsification condition as MET and the "
              "manuscript does not discuss it")


# --------------------------------------------------------------------------- process
def gate_process(tex: str) -> None:
    body_lines = [l for l in tex.split("\n") if not l.lstrip().startswith("%")]
    body = "\n".join(body_lines)
    check("[[MEASURE" not in body,
          "an unfilled [[MEASURE:...]] placeholder is still in the manuscript body")

    dec = read(os.path.join(ROOT, "DECISIONS_REDACTED.md" if RELEASED else "DECISIONS.md"))
    declared = re.findall(r"Declared amendment count(?: so far)?: \*\*(\d+)\*\*", dec)
    check(bool(declared), "DECISIONS.md declares no amendment count (anchor lost)")
    if declared:
        amendments = re.findall(r"(?m)^\*\*A-\d+ \|", dec)
        check(int(declared[-1]) == len(amendments),
              f"DECISIONS.md declares {declared[-1]} amendments but contains {len(amendments)}")
        stated = re.search(r"\\newcommand\{\\NumAmendments\}\{(\d+)\}",
                           read(os.path.join(HERE, "numbers.tex")) if os.path.exists(
                               os.path.join(HERE, "numbers.tex")) else "")
        if stated:
            check(stated.group(1) == declared[-1],
                  "the amendment count in the paper disagrees with DECISIONS.md")

    # CAMERA-READY (2026-10-17): the submission-era anonymity token check is retired, because the
    # camera-ready names its author (positive checks in VENUE). What replaces it is the artifact link:
    # the paper promises the artifact, and PROMPTOPS and RASE were lost to an unreachable one (BM11).
    check("[[ARTIFACT-URL]]" not in body,
          "the artifact link is still the [[ARTIFACT-URL]] placeholder; the public repository must "
          "exist and its real URL replace the placeholder before upload")
    m_url = re.search(r"\\newcommand\{\\ArtifactURL\}\{\\url\{(https://[^}]+)\}\}", tex)
    check(m_url is not None, "ArtifactURL is not defined as \\url{https://...}")
    check("ArtifactURL{}" in body, "the manuscript never prints the artifact link")
    for pat, why in [
        (r"accompanies the paper on publication", "submission-era promise; the camera-ready prints the link"),
        (r"nothing is attached", "submission-era wording about supplementary material"),
        (r"accepts\s+no supplementary material", "submission-era wording about supplementary material"),
        (r"Every number except those from the", "retired: the 550B numbers regenerate from stored outputs"),
    ]:
        check(re.search(pat, body) is None, f"retired availability wording is back ({why}): /{pat}/")

    # Retired claims: anything the review rounds removed must not come back. Anchored patterns,
    # never bare substrings, so that a phrase cannot fire on its own repair.
    retired = [
        (r"no published pipeline reports any of these", "overclaim retired before submission: the coding "
         "table shows several pipelines do report some items"),
        (r"prov(es|e) that LLM trading is spurious", "overclaim: the paper measures dispersion, "
         "it does not establish that any particular published result is spurious"),
        # Retired after the sol review, 2026-08-01. Each was a real overclaim; the reason is
        # recorded so that a later round cannot reinstate one by taste.
        (r"exact (within-session )?permutation null", "the permutation is a Monte Carlo benchmark "
         "enforcing within-session exchangeability, not an exact test of no model contribution"),
        (r"the language model contributes nothing\b", "the benchmark cannot reject a session-level "
         "or market-timing contribution, so this null is not the one tested"),
        (r"[Tt]he alpha is in the grid, not in the model", "attributes to the model a conclusion "
         "the within-session benchmark does not identify"),
        (r"scores contribute nothing detectable", "same overclaim in the results section"),
        (r"modal literature configuration", "the coding table records disclosure, not the levels "
         "chosen, so modality is not established"),
        # Pattern loosened 2026-08-09 after it failed in exactly the way the house rule predicts.
        # It pinned the subject "Scale", so when the retired claim came back as "The larger models
        # bought compliance, not signal" the gate saw nothing, and the sentence contradicted the
        # compliance numbers six lines above it for two review rounds. Match the CLAIM, whatever
        # its subject: the reference arm is already at 100%, so nothing on the ladder bought
        # compliance. Sibling wordings are covered too, so a rephrase does not walk it back in.
        (r"bought compliance(?!\s+nor)|buys compliance|bought (?:it )?better compliance",
         "the reference arm is already at 100% strict compliance, so no larger model bought any; "
         "and one model replacement changes family and training data too, so it does not isolate "
         "parameter count"),
        (r"one strategy read three ways", "entry timing, holding period and weighting change the "
         "trades, so these are configurations rather than readings"),
        (r"cannot generate dispersion between configurations",
         "a constant can still bear unequally on configurations that differ in turnover; the cost "
         "sweep demonstrates exactly that"),
        (r"is a floor on the multiplicity", "one calibration on one grid does not establish a "
         "general lower bound"),
    ]
    for pat, why in retired:
        check(re.search(pat, tex, re.I) is None, f"retired claim is back ({why}): /{pat}/")

    # House style.
    check("---" not in re.sub(r"(?m)^\s*%.*$", "", tex).replace("\\\\", ""),
          "an em-dash (---) is in the manuscript; house style forbids them")
    for word in ["furthermore", "moreover", "notably", "it is worth noting", "delve"]:
        check(word not in re.sub(r"(?m)^\s*%.*$", "", tex).lower(),
              f"banned filler word in the manuscript: {word!r}")


# --------------------------------------------------------------------------- pages
def gate_editcheck() -> None:
    """The second half of the double gate: no edit may pass unreviewed.

    Every other check here inspects the document as it stands, which is why a sentence that was
    CORRECT and became wrong under an edit sailed through: there was nothing to compare against.
    `code/editcheck.py` closes that by diffing against a snapshot and interrogating only what
    changed. This asserts the snapshot is CURRENT, so the gate cannot pass unless whoever edited
    ran the diff and re-snapshotted after reading it. A stale snapshot means edits exist that
    nobody looked at.
    """
    snap = os.path.join(ROOT, "notes", "editcheck_snapshot.json")
    ec = os.path.join(ROOT, "code", "editcheck.py")
    if RELEASED:
        print("  (edit-snapshot check skipped: an authoring gate; the released artifact has no notes/)")
        return
    if not check(os.path.exists(ec), "code/editcheck.py is missing; edits are unguarded"):
        return
    if not check(os.path.exists(snap),
                 "no edit snapshot: run `python code/editcheck.py snapshot`, edit, then "
                 "`python code/editcheck.py check` before building"):
        return
    sys.path.insert(0, os.path.join(ROOT, "code"))
    try:
        import editcheck  # noqa: PLC0415 - imported lazily so the gate works without it on path
    except Exception:  # noqa: BLE001
        check(False, "code/editcheck.py could not be imported")
        return
    saved = json.load(open(snap, encoding="utf-8"))["sentences"]
    current = editcheck.sentences(read(os.path.join(HERE, "main.tex")))
    check(saved == current,
          f"the edit snapshot is STALE: {len(current)} sentences now against {len(saved)} recorded, "
          f"so main.tex changed since it was last reviewed. Run "
          f"`python code/editcheck.py check`, read every flagged sentence, then re-snapshot.")


def gate_traceability() -> None:
    """The macro generator may read no source of truth but the analysis output.

    verify_output.txt tells reviewers that nothing in it was typed. Fourteen printed macros came
    from cap_probe.json, paper_coding.json, the decision ledger and model_provenance.json instead,
    so the claim was false. Checking that each printed VALUE appears somewhere in the dump does not
    catch this (with a thousand values, a fabricated one still matches something); checking where
    the generator is allowed to read does.
    """
    gen = os.path.join(ROOT, "code", "make_macros.py")
    if not os.path.exists(gen):
        check(False, "code/make_macros.py is missing; traceability cannot be checked")
        return
    src = read(gen)
    allowed = {"numbers.json", "paper_coding.json"}
    reads = set(re.findall(r'["\']([A-Za-z0-9_]+\.(?:json|jsonl|csv|md))["\']', src))
    stray = sorted(reads - allowed)
    check(not stray,
          f"make_macros.py reads {stray} in addition to the analysis output, so a printed number "
          f"could come from a second source and verify_output.txt's claim that nothing was typed "
          f"would be false again")


def gate_reviewed_macros(macros: str) -> None:
    """Every macro value the reviewers read is unchanged in the camera-ready.

    The camera-ready may add numbers (the index-membership audit adds the Mem* family) but may not
    move one the reviewers saw. notes/numbers_reviewed.tex is the submitted build's numbers.tex
    (FINAL_v439, sha256 232be12e...). notes/ does not ship, so in the released artifact this check
    has nothing to compare against and says so.
    """
    ref_path = os.path.join(ROOT, "notes", "numbers_reviewed.tex")
    if not os.path.exists(ref_path):
        print("  (reviewed-macro check skipped: notes/numbers_reviewed.tex is not in this tree)")
        return
    pat = r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}"
    old = dict(re.findall(pat, read(ref_path)))
    new = dict(re.findall(pat, macros))
    check(len(old) > 200, f"numbers_reviewed.tex parsed to only {len(old)} macros")
    moved = sorted(k for k in old if new.get(k) != old[k])
    check(not moved, f"macro values the reviewers read have changed or vanished: "
                     f"{[(k, old[k], new.get(k)) for k in moved][:8]}")
    added = sorted(k for k in new if k not in old)
    # NumPriceOK: an existing analysis output (numbers.json n_price_ok) printed at camera-ready to
    # correct the Data sentence, which described all NumEvents events as price-complete.
    stray = [k for k in added if not k.startswith("Mem") and k not in {"NumPriceOK"}]
    check(not stray, f"new macros outside the Mem* membership family: {stray}")


def gate_camera_ready_claims(tex: str) -> None:
    """The claims the camera-ready corrected, as substance with negative clauses.

    Four were provable errors in the reviewed version: reasoning was said to be disabled for the
    550B arm only (the 36B arm ran with think=False too, D-027); the score-level check was said to
    fail to "rank the next day's cross-section" (power.py computes one pooled Spearman per
    configuration); the protocol was said to cost "no additional inference" (R1's LLM-side levels
    each cost a pass); and the universe was disclosed only as missing delisted names, while 21
    joiners put look-ahead filings in the sample. The rest are reviewer-requested scoping.
    """
    body = re.sub(r"(?m)^\s*%.*$", "", tex)
    flat = re.sub(r"\s+", " ", body)
    retired = [
        (r"exception three\s+times over", "the 550B arm called the only reasoning-off arm"),
        (r"rank the next day's cross-section", "the per-day cross-section misdescription"),
        (r"no additional inference cost", "the protocol called inference-free"),
        (r"affordable whenever the grid was", "R4's cost stated as affordability"),
        (r"\(free: the same raw output", "the grid table's 'free' label"),
        (r"size is never reported", "the abstract's 'never reported'"),
        (r"a number nobody reports", "the 'nobody reports' overreach"),
        (r"stripped of any information about which filing", "the abstract overstating the null"),
        (r"no longer say which filing they belong to", "the conclusion overstating the null"),
        (r"more undisclosed moving parts than disclosed", "the unscoped conclusion"),
        (r"fixed at one fetch date", "the undated universe"),
        (r"names delisted during the window are absent", "survivorship disclosed in one direction"),
        # loopB cycle 1 (2026-10-17): each was an error in a cr2 edit or exposed by one.
        (r"each further LLM-side level", "LLM-side cost stated per level in a factorial grid"),
        (r"one inference pass per level\)", "the Table 2 label stating cost per level"),
        (r"the search alone reaches", "the abstract calling the within-session null 'the search alone'"),
        (r"and lists \\MemConstituents\{\} names", "502 called the page's count (it lists 503)"),
        (r"the study trades them", "pre-addition filings called traded (two are never held)"),
        (r"have the price\s+history the holding rules need", "all NumEvents called price-complete"),
        (r"Threats explains why", "anthropomorphised cross-reference"),
        (r"needs an API key to reproduce", "re-scoring confused with reproducing"),
        # loopB cycle 2.
        (r"bought neither compliance nor signal", "absence of signal asserted for the scale arms"),
        (r"what the grid alone produces\. The last row", "Table 4 caption overstating the null"),
        (r"accounts for \\VarShareModel\{\}\\% of it\.", "'of it' read as a share of interactions"),
        (r"configuration counts on a", "alt text calling permutation draws configurations"),
        (r"filings in the sample, \\MemPreEvents", "'sample' used for the 2,978 event set"),
        (r"Our parser missed one", "conversational wording"),
        # loopB cycle 3: the 550B arm's spread (0.95) is below the reference arm's (0.97).
        (r"dispersion does not fall either", "dispersion said not to fall, against its own numbers"),
        # External judge, 2026-10-18: the list is the 2026-07-21 revision; say so by its date.
        (r"index membership at the end of the\s+window", "membership dated to the window end"),
        (r"coded by one author", "implies co-authors in a single-author paper"),
    ]
    for pat, what in retired:
        check(re.search(pat, flat) is None, f"retired camera-ready wording is back ({what}): /{pat}/")
    required = [
        (r"Both larger models ran with their reasoning channel disabled", "both arms reasoning-off"),
        (r"In trial calls with reasoning on", "the empty-answer claim scoped to what was observed"),
        (r"has scores with a positive rank correlation with the forward return", "the pooled IC, "
                                                                                  "positive side"),
        (r"In this sample, none of the \\ICNumConfigs\{\} LLM configurations", "IC claim scoped"),
        (r"R2 to R5 below need no new model inference", "the protocol cost split"),
        (r"every combination of LLM-side levels", "R1's inference cost, factorial"),
        (r"one inference pass per combination of levels", "the Table 2 cost label, factorial"),
        (r"declare every later amendment", "R1's freeze-and-declare advice"),
        (r"On this grid, \\VarShareInteractions\{\}\\% of the variance", "the scoped conclusion"),
        (r"permuted among the filings of each session", "the within-session null"),
        (r"the grid run on permuted scores", "the abstract's null named as such"),
        (r"\\MemJoiners\{\}.{0,400}\\MemPreEvents\{\}.{0,600}\\MemRemovals\{\}",
         "the membership counts in Threats"),
        (r"No direction is established for either effect", "no direction claimed for membership"),
        (r"membership also enters the comparison between the pipeline and its null",
         "R2 W2's permutation comparison answered"),
        (r"revision \\MemRevision\{\}", "the pinned revision named"),
        (r"\\MemListed\{\} constituents.{0,80}Cboe Global Markets", "the parser's dropped row"),
        (r"\\NumPriceOK\{\} of these have every price a position needs", "price-complete count"),
        (r"which upstream provider served each call was not recorded", "the 550B provider disclosed"),
        (r"under the tag \\nolinkurl\{qwen3\.6:latest\}", "the 36B run tag, as run"),
        (r"\\Description\{", "figure alt text (ACM accessibility)"),
    ]
    for pat, what in required:
        check(re.search(pat, flat) is not None, f"camera-ready correction missing ({what}): /{pat}/")
    # The conclusion must not set ModelSD beside HarnessSD: ModelSD averages ddof=1 SDs over groups
    # of three and is biased low against HarnessSD (groups of 2,592), so the pair reads as an
    # ordering the data do not support (root-mean-variance 1.00 vs 0.98; PROVENANCE.md).
    concl = re.search(r"\\section\{Conclusion\}(.*?)\\textbf\{Artifact", flat)
    check(concl is not None, "Conclusion region not found (anchor moved?)")
    if concl:
        check("\\HarnessSD" not in concl.group(1) and "\\ModelSD" not in concl.group(1),
              "the Conclusion sets HarnessSD beside ModelSD, a group-size-biased comparison")
    # A thousands separator inside math mode typesets as "7, 776". Every \NumConfigs in math
    # must be boxed.
    for seg in re.findall(r"\$[^$]*\$", flat):
        if "\\NumConfigs{}" in seg and not re.search(r"\\mbox\{(\\mdseries)?\\NumConfigs\{\}\}", seg):
            check(False, f"\\NumConfigs in math mode without \\mbox (renders '7, 776'): {seg[:60]}")


def gate_ordinals(tex: str, macros: str) -> None:
    """A suffix typed after a macro must match its value: '72th' reached the reviewed PDF."""
    vals = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", macros))
    seen = 0
    for m in re.finditer(r"\\([A-Za-z]+)\{\}(st|nd|rd|th)\b", tex):
        v = vals.get(m.group(1), "")
        if not re.fullmatch(r"\d+", v):
            continue
        seen += 1
        n = int(v)
        want = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        check(m.group(2) == want, f"\\{m.group(1)}{{}}{m.group(2)} renders '{n}{m.group(2)}', "
                                  f"should be '{n}{want}'")
    check(seen > 0, "no macro-plus-ordinal sites found (anchor lost?)")


def gate_artifact() -> None:
    """Every data file a printed number depends on must be in the shipped artifact.

    The first pack omitted data/scores_or.jsonl and data/cap_probe.json, so five printed macros and
    the whole decode-cap ablation could not be reproduced from the release at all: one came from a
    hosted endpoint that may not exist later, the other from 1,800 local calls. The artifact's own
    reproducibility claim was therefore false for exactly the arm a reviewer is most likely to
    doubt. The packer's include list is the thing to check, since the zip is rebuilt on demand.
    """
    # The shipped ledger is generated from the working one, and was hand-written once and left to
    # rot: it declared six amendments while the paper declared seven. Regenerating is one command,
    # so a stale copy is a gate failure rather than a note.
    red = os.path.join(ROOT, "DECISIONS_REDACTED.md")
    dec = os.path.join(ROOT, "DECISIONS.md")
    if os.path.exists(red) and os.path.exists(dec):
        n_red = len(re.findall(r"(?m)^\*\*A-\d+ \|", read(red)))
        n_dec = len(re.findall(r"(?m)^\*\*A-\d+ \|", read(dec)))
        check(n_red == n_dec,
              f"DECISIONS_REDACTED.md declares {n_red} amendments and DECISIONS.md declares "
              f"{n_dec}; the artifact ships the redacted one, so a reviewer would read a different "
              f"count from the one the paper prints. Run code/make_redacted.py")
        e_red = len(re.findall(r"(?m)^\*\*[DA]-\d+ \|", read(red)))
        e_dec = len(re.findall(r"(?m)^\*\*[DA]-\d+ \|", read(dec)))
        check(e_red == e_dec,
              f"the shipped ledger has {e_red} entries and the working one has {e_dec}")

    pack = os.path.join(ROOT, "code", "pack_artifact.py")
    if RELEASED:
        src = None   # the packer does not ship; check the files themselves below
    elif not os.path.exists(pack):
        check(False, "code/pack_artifact.py is missing, so artifact completeness cannot be checked")
        return
    else:
        src = read(pack)
    needed = {
        "scores.jsonl": "the main grid's raw model outputs",
        "scores_probe.jsonl": "the mask, seed and 36B arms",
        "scores_or.jsonl": "the 550B arm, which cannot be re-derived from a local model",
        "cap_probe.json": "the decode-cap ablation, 1,800 local calls",
        "numbers.json": "every printed quantity",
        "prices.csv.gz": "the price panel",
        "event_sample.json": "the subsample draw and the pool it came from",
        "model_provenance.json": "the release dates the contamination scope rests on",
        # REPRODUCE.md advertises `harness_audit.py data/grid_returns.npy`, and figures.py and
        # robust_sharpe.py both need the grid matrices. Shipping the runsheet without the file it
        # names is an instruction that fails on first use.
        "grid_returns.npy": "the per-configuration return matrix the runsheet tells reviewers to run",
        "null_max.npy": "the permutation null draws behind Figure 1",
        "sp500_wikipedia_rev1365256480.wikitext.gz": "the pinned page revision the membership "
                                                     "audit and the universe rest on",
    }
    for f, why in sorted(needed.items()):
        present = (os.path.exists(os.path.join(ROOT, "data", f)) if RELEASED
                   else f'"{f}"' in src)
        check(present, f"the artifact does not ship data/{f} ({why}), so numbers derived "
                       f"from it cannot be reproduced from the release")


def gate_floats(tex: str) -> None:
    """Every labelled figure and table must be referenced from the body, and vice versa.

    Cutting Figure 3 for length left `fig:dispersion` labelled but unreferenced, which a reviewer
    reads as a figure the authors forgot to discuss, and which silently disarmed a self-test
    injection anchored on that reference. Neither a build log nor a number-tracing gate sees it:
    LaTeX warns about undefined refs, never about unused labels.
    """
    labels = set(re.findall(r"\\label\{((?:fig|tab):[^}]+)\}", tex))
    refs = set(re.findall(r"\\ref\{((?:fig|tab):[^}]+)\}", tex))
    orphan = sorted(labels - refs)
    dangling = sorted(refs - labels)
    check(not orphan, f"labelled but never referenced in the body: {orphan}")
    check(not dangling, f"referenced but never labelled: {dangling}")


def gate_citation_claims(tex: str) -> None:
    """Claims ABOUT other papers, which no number-tracing gate can see.

    The quotation gate checks quotation -> refs.bib note. It cannot check a CLAIM about what a
    cited paper says, and one of those drifted: the body asserted Ye et al. name researcher
    degrees of freedom as one of THREE confounders, while the abstract names five and never uses
    the phrase. The body had also drifted from its own note, which said six validity tests. These
    are negative clauses on the retired wording plus positive checks on what re-fetching confirmed.
    """
    retired = [
        (r"one of\s+three confounders", "Ye et al. name five confounders in the abstract, not three"),
        (r"names researcher degrees of freedom", "that phrase is not in the cited abstract"),
        (r"up to \d+ accuracy points on (?!one model)", "the 76-point figure is specific to one model"),
    ]
    for pat, why in retired:
        check(re.search(pat, tex) is None, f"retired citation claim is back: {why}")
    # Positive checks on figures re-fetched from the sources this round.
    if "sclar2024quantifying" in tex:
        check(re.search(r"76 accuracy points", tex) is not None,
              "the Sclar et al. figure of 76 accuracy points is no longer stated as re-fetched")
    if "menkveld2024nonstandard" in tex:
        check(re.search(r"164 teams", tex) is not None,
              "the Menkveld et al. team count of 164 is no longer stated as re-fetched")


def gate_scale_ladder(tex: str) -> None:
    """The paper asserts both large arms were released INSIDE the sample window. Check it.

    That sentence is a factual claim about two external models, of the same class as the invented
    venue string that once shipped in a submitted paper: it is typed once, prints on every build,
    and no number-tracing gate can see it. Each release date is recorded in model_provenance.json
    with a fetched source, and this asserts the claim the prose actually makes.
    """
    n = json.load(open(os.path.join(DATA, "numbers.json"), encoding="utf-8"))
    first, last = str(n["first_session"]), str(n["last_session"])
    for key, label in (("scale_model", "36B scale arm"), ("big_model", "550B scale arm")):
        rec = n.get(key)
        check(rec is not None,
              f"numbers.json has no {key} block, so the {label}'s release date traces to nothing")
        if not rec:
            continue
        check(bool(rec.get("release_source")),
              f"the {label}'s release date has no recorded source")
        # The paper says both fall inside the window; if one ever falls outside, the sentence
        # claiming neither inherits the release-date bound becomes wrong in the safe direction,
        # which is still wrong.
        check(first <= rec["release"] <= last,
              f"the {label} was released {rec['release']}, which is OUTSIDE the sample window "
              f"{first}..{last}, but the paper says both large arms were released inside it")
    # The probe section was titled "Two arms that could have broken the design" while holding
    # five. A count in a heading is a claim, and headings are exactly what a number-tracing gate
    # never looks at.
    sec = re.search(r"\\subsection\{([^}]*could have broken the design)\}", tex)
    check(sec is not None, "the probe section heading was renamed; the arm-count check lost its anchor")
    if sec:
        body = tex[sec.end():]
        nxt = body.find("\\section{")
        n_arms = len(re.findall(r"\n\\textbf\{", body[:nxt if nxt > 0 else len(body)]))
        stated = re.match(r"(Two|Three|Four|Five|Six)\s", sec.group(1))
        check(stated is None,
              f"the probe section heading states a count ({stated.group(1) if stated else ''}) but "
              f"the section contains {n_arms} arms; a heading that counts goes stale silently")

    if "\\BigParams{}" in tex:
        check("\\BigPermP{}" in tex and "\\BigCompliance{}" in tex,
              "main.tex names the 550B arm without printing both its permutation p-value and its "
              "compliance, which are the two numbers that make it an answer to the scale objection")


def gate_robust(tex: str) -> None:
    """The autocorrelation-robust figures must be comparable to the ones they are printed against.

    The first version of this measurement computed the robust standard deviation over the 6,624
    configurations that actually trade and printed it against \\SharpeSD, which is over all 7,776.
    Both numbers were individually correct and the pair was not: it reported the spread widening by
    19% when the like-for-like figure is 11%. Nothing caught it, because every number traced to a
    real key and no gate knew the two keys had different denominators.
    """
    n = json.load(open(os.path.join(DATA, "numbers.json"), encoding="utf-8"))
    rb = n.get("robust")
    check(rb is not None,
          "numbers.json has no 'robust' block, so the annualisation figures the paper prints are "
          "either stale or were deleted by a re-run of analyze.py")
    if not rb:
        return
    # The invariant: the standard sd over the SAME array the robust sd came from is what the paper
    # prints as \SharpeSD. If a future edit narrows one population, this fails.
    check(abs(rb["std_sd_all"] - float(n["sharpe_sd"])) < 5e-3,
          f"population mismatch: robust figures are computed over a set whose standard sd is "
          f"{rb['std_sd_all']:.4f}, but the paper prints \\SharpeSD = {float(n['sharpe_sd']):.4f}. "
          f"The widening would be a comparison between two different populations.")
    check(rb["n_configs"] == int(n["n_configs"]),
          f"the robust block covers {rb['n_configs']} configurations but the grid has "
          f"{int(n['n_configs'])}")
    # Negative clause: the live-subset variants exist for reference and must never be printed,
    # because they are the ones that do not match \SharpeSD.
    check("\\RobustSDLive" not in tex,
          "main.tex prints \\RobustSDLive, which is computed over the trading subset and is not "
          "comparable to the all-configuration robust figures printed beside it")
    # \StdSDLive IS printed, on purpose, as the never-trade sensitivity a reviewer asks for. It is
    # only honest in a sentence that says what was excluded, so require the exclusion to be named
    # where it appears; a bare live-subset number beside all-configuration ones is the defect.
    if "\\StdSDLive{}" in tex:
        sent = next((s for s in re.split(r"(?<=[.!?])\s+", tex) if "\\StdSDLive{}" in s), "")
        check(re.search(r"never[- ]trad|excluding", sent) is not None,
              "main.tex prints \\StdSDLive without saying in the same sentence that the "
              "never-trading configurations were excluded, so it reads as comparable to the "
              "all-configuration spread when it is not")
    # Replication count is precision, and precision can only be silently REDUCED. Re-running the
    # probe analysis with a hand-typed "200" once downgraded every arm from the 300 the manuscript
    # reports and moved every printed null mean and p-value, while every other gate passed because
    # each number still traced to a real key. Pin the floor.
    probes = n.get("probes", {})
    for arm, a in sorted(probes.items()):
        if not isinstance(a, dict) or "perm_reps" not in a:
            continue
        check(a["perm_reps"] >= 300,
              f"probe arm {arm!r} was computed at {a['perm_reps']} permutation replications, below "
              f"the 300 the manuscript reports; every p-value and null mean in that arm is at "
              f"reduced precision")
    reps_seen = {a["perm_reps"] for a in probes.values()
                 if isinstance(a, dict) and "perm_reps" in a}
    check(len(reps_seen) <= 1,
          f"probe arms were computed at differing replication counts {sorted(reps_seen)}, so their "
          f"p-values are not comparable with each other")

    check(rb.get("widening_survives_trimming") is True,
          "the robust widening no longer survives trimming and the IQR, so the paper's claim that "
          "it is not driven by a few autocorrelated cells is no longer supported")


def gate_stray_files() -> None:
    """paper/ holds the submission, not review output.

    `agy_review.py` writes its report to paper/agy_report.md by default, so every free review pass
    drops a reviewer's assessment of this paper into the directory the submission is built from.
    It does not currently reach the artifact, because the packer takes only .tex/.bib/.py from
    here, but that is one edit away from being false and the file is the kind that would be
    embarrassing to ship. Fail loudly instead of relying on remembering to move it.
    """
    allowed_ext = {".tex", ".bib", ".py", ".pdf", ".log", ".txt", ".aux", ".bbl", ".blg",
                   ".out", ".sty", ".cls", ".bst", ".png", ".pdf_tex"}
    # The edit-ledger tool keeps its record (*ledger.json) here by design. The packer takes
    # only .tex/.bib/.py/.pdf and verify_output.txt from paper/, so the ledger never ships.
    allowed_names = {n for n in os.listdir(HERE) if n.endswith("ledger.json")}
    strays = []
    for name in sorted(os.listdir(HERE)):
        path = os.path.join(HERE, name)
        if os.path.isdir(path) or name in allowed_names:
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in allowed_ext:
            strays.append(name)
    check(not strays,
          f"paper/ contains files that are not part of the submission: {strays}. Review output "
          f"belongs in notes/, not in the directory the artifact is packed from.")


def gate_power(tex: str) -> None:
    """The positive control must be reported as a rejection rate over independent oracle draws.

    An earlier draft ran ONE synthetic oracle per accuracy level and called the result a power
    analysis: it printed a single $p=0.030$ at 55% and concluded "the null above is a powered one".
    Repeating the draw showed 55% rejects in 40% of draws, so that p-value was one favourable
    realisation and the sentence built on it was wrong. This gate pins the SUBSTANCE (draws exist,
    rejection rates are printed, the reported threshold is the level that rejects in every draw)
    and, per the house rule, carries a negative clause naming the retired wording, so "claim
    weakened correctly" cannot be confused with "claim silently restored".
    """
    n = json.load(open(os.path.join(DATA, "numbers.json"), encoding="utf-8"))
    pw = n.get("power", {})
    check(int(pw.get("draws", 1)) >= 10,
          f"the positive control used {pw.get('draws')} oracle draw(s) per accuracy; one draw is a "
          "sensitivity check and may not be reported as power")
    accs = pw.get("accuracies", {})
    check(all("rejection_rate" in a for a in accs.values()),
          "some accuracy level has no rejection_rate, so its claim rests on a single realisation")
    # The 50% oracle carries no directional information: if the benchmark rejects it often, the
    # negative control is broken and every other rejection rate in the table is uninterpretable.
    if "0.50" in accs:
        check(accs["0.50"]["rejection_rate"] <= 0.10,
              f"the 50% oracle is rejected in {accs['0.50']['rejection_rate']:.0%} of draws, which "
              "means the benchmark rejects noise and the positive control proves nothing")
    strong = sorted(a for a, v in accs.items() if v["rejection_rate"] >= 0.8)
    mda = pw.get("min_detectable_accuracy")
    check(strong and mda == strong[0],
          f"min_detectable_accuracy is {mda!r} but the lowest level rejecting in at least 80% of "
          f"draws is {strong[0] if strong else None!r}")
    # Every level whose rejection rate the paper states must print its OWN macro. The 70% level
    # was briefly wired to the 60% macro, which read correctly only because both were 100.
    for tag, name in (("0.50", "Fifty"), ("0.55", "Fiftyfive"), ("0.60", "Sixty"),
                      ("0.70", "Seventy")):
        check(f"\\Oracle{name}Rej{{}}" in tex,
              f"main.tex never prints the rejection rate at {tag}, so a reader sees a p-value "
              "from one draw and no evidence of how often it replicates")
    check("\\PowerDraws{}" in tex,
          "main.tex never says how many independent oracle draws each rejection rate is over")
    # Negative clause: the retired claims, each of which the repeated draws refuted.
    for bad, why in (
        (r"null above is a powered one", "the phrase asserts power that 10 draws do not support"),
        (r"resolves a directional edge of \\MinDetectAcc", "phrased as a resolved edge, not a rate"),
        (r"at 55\\% it is detected", "55% rejects in a minority of draws; it is not a detection"),
        (r"resolves reliably", "10 draws support a count of draws, not general reliability"),
        (r"would have to be stronger than", "no experiment here estimates where that threshold is"),
    ):
        check(re.search(bad, tex) is None, f"retired positive-control claim is back: {why}")


def gate_pages() -> None:
    # main.log is pdflatex's own log and always exists after a build; b4.log was an ad hoc
    # redirect an earlier session happened to use, and deleting it silently disarmed this gate.
    candidates = [p for p in (os.path.join(HERE, "main.log"), os.path.join(HERE, "b4.log"))
                  if os.path.exists(p)]
    if RELEASED and not candidates:
        print("  (page-limit check skipped: an authoring gate; rebuild the PDF to produce main.log)")
        return
    if not candidates:
        check(False, "no build log in paper/, so the page count was never checked")
        return
    log = candidates[0]
    # A log older than the manuscript certifies the page count of a DIFFERENT paper. This gate
    # used to accept any log it found, so a stale one would have passed an over-length draft.
    tex = os.path.join(HERE, "main.tex")
    check(os.path.getmtime(log) >= os.path.getmtime(tex) - 1,
          f"{os.path.basename(log)} is older than main.tex, so its page count predates the current "
          f"text; rebuild before trusting it")
    m = re.search(r"Output written on main\.pdf \((\d+) pages", read(log))
    check(m is not None, "the build log does not report a page count (anchor lost)")
    if m:
        # Camera-ready limit: "the page limit is 9 pages including references" (ICAIF '26
        # proceedings chair, CMT email 2026-10-11). The submission limit was 8.
        check(int(m.group(1)) <= 9,
              f"the paper is {m.group(1)} pages; the ICAIF '26 camera-ready allows 9 including references")


# --------------------------------------------------------------------------- selftest
def _sample_identity_token() -> str:
    """A concrete string that the anonymity check must reject, derived from the external list."""
    path = os.path.join(ROOT, "notes", "identity_tokens.txt")
    if not os.path.exists(path):
        raise SystemExit("selftest needs notes/identity_tokens.txt to build the identity injection")
    for line in read(path).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Turn a small regex such as [Ss]imarjot into one literal string it matches.
        return re.sub(r"\[(.)[^\]]*\]", r"\1", line)
    raise SystemExit("notes/identity_tokens.txt contains no tokens")


INJECTIONS = [
    ("venue city", lambda t: t.replace("Milan, AA, Italy", "Munich, Germany")),
    # Camera-ready inversions (BO9): each must be CAUGHT in the new direction.
    ("blinding restored", lambda t: t.replace("\\documentclass[sigconf]{acmart}",
                                              "\\documentclass[sigconf,anonymous]{acmart}")),
    ("author block reverted", lambda t: re.sub(r"\\author\{[^}]*\}",
                                               lambda _m: "\\author{Anonymous Author(s)}", t, count=1)),
    ("rights block reverted", lambda t: t.replace("\\setcopyright{cc}", "\\setcopyright{rightsretained}")),
    ("DOI dropped", lambda t: t.replace("\\acmDOI{10.1145/3858063.3858869}", "")),
    ("submission-era availability wording", lambda t: t.replace(
        "\\section{Conclusion}", "\\section{Conclusion}\nThe artifact accompanies the paper on publication.")),
    # Re-anchored: fig:emax was removed when Figure 3 was cut for length, and the injection then
    # silently matched nothing, which the self-test correctly reported as a MISS rather than a pass.
    ("eaten backslash", lambda t: t.replace("Figure~" + chr(92) + "ref{fig:dispersion}",
                                            "Figure~" + chr(13) + "ef{fig:dispersion}")),
    ("tab from an escape", lambda t: t.replace("The Grid Is the Alpha",
                                               "The Grid" + chr(9) + "Is the Alpha")),
    ("model cutoff date", lambda t: t.replace("October 2023", "October 2024")),
    ("scale-arm scope claim removed", lambda t: t.replace(
        "release-date bound covers the grid, not the scale arm",
        "release-date bound: all four models predate the window")),
    ("model name drift", lambda t: t.replace("Phi-3-mini", "Phi-4-mini")),
    ("venue month", lambda t: t.replace("November 14--17", "October 14--17")),
    ("unfilled placeholder", lambda t: t.replace("\\section{Conclusion}",
                                                 "\\section{Conclusion}\n[[MEASURE:x | y | z | w]]")),
    ("hard-coded number", lambda t: t.replace("\\section{Related work}",
                                              "The Sharpe ratio ranged up to 1.97.\n"
                                              "\\section{Related work}")),
    ("em-dash", lambda t: t.replace("\\section{Related work}",
                                    "A dash---like this.\n\\section{Related work}")),
    ("filler word", lambda t: t.replace("\\section{Related work}",
                                        "Moreover, this is filler.\n\\section{Related work}")),
    ("retired overclaim", lambda t: t.replace("\\section{Related work}",
                                              "This proves that LLM trading is spurious.\n"
                                              "\\section{Related work}")),
    ("undefined macro", lambda t: t.replace("\\section{Related work}",
                                            "The value is \\NotAMacro{}.\n\\section{Related work}")),
    # The two show-stoppers found in the final review round, re-injected so the fixes are enforced
    # rather than remembered.
    ("powered-null claim restored", lambda t: t.replace(
        "The empirical null above is consequently not explained by",
        "The null above is a powered one. It is not explained by")),
    ("rejection rate dropped", lambda t: t.replace("\\OracleFiftyfiveRej{}\\%", "40\\%")),
    ("70% level rewired to the 60% macro", lambda t: t.replace(
        "at 70\\% \\OracleSeventyRej{}", "at 70\\% \\OracleSixtyRej{}")),
    ("robust sd swapped for the live-subset one", lambda t: t.replace(
        "to \\RobustSD{}", "to \\RobustSDLive{}")),
    ("reliability overclaim", lambda t: t.replace(
        "is the lowest level we tested that was rejected", "is the accuracy this pipeline "
        "resolves reliably, being rejected")),
    # The camera-ready exemptions, re-injected so neither can become a hiding place.
    ("number hidden in a malformed orcid", lambda t: t.replace(
        "\\orcid{0009-0009-4827-9884}", "\\orcid{0009-0009-4827-98845}")),
    ("macro defined in main.tex outside the allowlist", lambda t: t.replace(
        "\\section{Related work}",
        "\\newcommand{\\LocalCount}{x}The value is \\LocalCount{}.\n\\section{Related work}")),
    # The camera-ready corrections, each reverted to the reviewed wording.
    ("reasoning-off claimed for one arm only", lambda t: t.replace(
        "Both larger models ran with their reasoning channel disabled",
        "Only the larger hosted model ran with its reasoning channel disabled")),
    ("cross-section misdescription restored", lambda t: t.replace(
        "In this sample, none of the",
        "Whatever these models do, it does not rank the next day's cross-section, and none of the")),
    ("protocol called inference-free", lambda t: t.replace(
        "Items R2 to R5 below need no new model inference.",
        "All of it comes at no additional inference cost.")),
    ("membership counts dropped", lambda t: t.replace("\\MemPreEvents{} (\\MemPrePct{}", "some (")),
    ("conclusion unscoped", lambda t: t.replace(
        "computed. On this grid,", "computed. In general,")),
    # loopB cycle 1 fixes, each reverted.
    ("SD pair back in the conclusion", lambda t: t.replace(
        "The largest single choice, the language model, accounts for",
        "Re-deciding the pipeline moves Sharpe by \\HarnessSD{} against \\ModelSD{} for the model, and "
        "the largest single choice, the language model, accounts for")),
    ("per-level LLM cost restored", lambda t: t.replace(
        "every combination of LLM-side levels (model, prompt and truncation) needs its own inference "
        "pass", "each further LLM-side level (a model, prompt or truncation setting) costs one more "
        "inference pass")),
    ("thousands separator back in math", lambda t: t.replace(
        "$N=$ full grid $=\\mbox{\\NumConfigs{}}$", "$N=$ full grid $=\\NumConfigs{}$")),
    ("Cboe disclosure dropped", lambda t: t.replace(
        "One row, for Cboe Global Markets, uses a different ticker format\nand was not parsed, so the "
        "universe holds", "The universe holds")),
    ("ordinal typo restored", lambda t: t.replace("\\SplitOOSPercentile{}nd percentile",
                                                  "\\SplitOOSPercentile{}th percentile")),
    ("scale-arm absence of signal restored", lambda t: t.replace(
        "showed no better compliance than the reference arm and no detectable signal",
        "showed that scale bought neither compliance nor signal")),
    ("dispersion said not to fall", lambda t: t.replace(
        "Post-LLM dispersion does not disappear either", "Post-LLM dispersion does not fall either")),
    ("figure alt text dropped", lambda t: re.sub(r"\\Description\{.*?configuration\.\}\n", "", t,
                                                 count=1, flags=re.S)),
    ("search-alone abstract restored", lambda t: t.replace(
        "the grid run on permuted scores reaches", "the search alone reaches")),
]


def run_gate(tex: str, macros: str) -> list:
    """Return the list of failure messages, not their count.

    Comparing counts is not enough: an injected defect that lands in a check which already
    reports a list (for instance the hard-coded-number check) leaves the count unchanged, and the
    self-test then declares a MISS as a pass. The self-test compares message SETS.
    """
    FAILS.clear()
    CHECKS["n"] = 0
    gate_corruption(tex)
    gate_venue(tex)
    gate_models(tex)
    gate_numbers(tex, macros)
    gate_grid()
    gate_directions(tex)
    gate_process(tex)
    gate_power(tex)
    gate_editcheck()
    gate_traceability()
    gate_artifact()
    gate_camera_ready_claims(tex)
    gate_reviewed_macros(macros)
    gate_ordinals(tex, macros)
    gate_floats(tex)
    gate_citation_claims(tex)
    gate_scale_ladder(tex)
    gate_robust(tex)
    gate_stray_files()
    gate_pages()
    return list(FAILS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    tex_path = os.path.join(HERE, "main.tex")
    # Read WITHOUT newline translation. Python would otherwise turn every CRLF into LF and, with
    # it, hide the stray carriage return that a shell heredoc leaves behind when it eats a
    # backslash. Every other check tolerates the raw form because they match on \s.
    tex = read_raw(tex_path)
    macros = read(os.path.join(HERE, "numbers.tex"))

    if args.selftest:
        base = set(run_gate(tex, macros))
        print(f"baseline: {CHECKS['n']} checks, {len(base)} failures")
        missed = []
        for name, mutate in INJECTIONS:
            mutated = mutate(tex)
            if mutated == tex:
                # The target string moved, so nothing was injected. That is a harness failure,
                # not a pass.
                print(f"  INJECTION DID NOT APPLY  {name}")
                missed.append(name)
                continue
            after = set(run_gate(mutated, macros))
            caught = bool(after - base)
            print(f"  {'CAUGHT ' if caught else 'MISSED '} {name}")
            if not caught:
                missed.append(name)
        print()
        if missed:
            print(f"SELFTEST FAILED: {len(missed)} injected defect(s) not caught: {missed}")
            sys.exit(1)
        print(f"SELFTEST PASSED: {len(INJECTIONS)}/{len(INJECTIONS)} injected defects caught")
        return

    n = run_gate(tex, macros)
    print(f"{CHECKS['n']} checks run")
    if n:
        print(f"\nCONSISTENCY FAILED: {n} problem(s)")
        for f in FAILS:
            print("  - " + f)
        sys.exit(1)
    print("CONSISTENCY PASSED")


if __name__ == "__main__":
    main()
