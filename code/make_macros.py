"""Emit paper/numbers.tex from data/numbers.json. The manuscript may not hard-code any quantity.

Every macro defined here is traceable to one key of numbers.json, which is itself produced only by
analyze.py from the grid. `paper/consistency.py` checks both directions: every macro used in the
manuscript is defined here, and every macro defined here still matches numbers.json.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
PAPER = os.path.abspath(os.path.join(HERE, "..", "paper"))

FACTOR_LABEL = {"model": "model", "prompt": "prompt", "trunc": "truncation", "parse": "parsing",
                "signal": "signal", "thresh": "threshold", "hold": "holding", "entry": "entry",
                "weight": "weighting", "neutral": "neutralisation"}


def commafy(x: float) -> str:
    return f"{int(round(x)):,}"


def main() -> None:
    n = json.load(open(os.path.join(DATA, "numbers.json"), encoding="utf-8"))
    L = []

    def m(name: str, value: str) -> None:
        L.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    # --- scale ---------------------------------------------------------------
    m("NumEvents", commafy(n["n_events"]))
    m("NumSessions", str(n["n_sessions"]))
    m("NumConfigs", commafy(n["n_configs"]))
    m("NumLLMConfigs", str(n["n_llm_configs"]))
    m("NumPostConfigs", str(n["n_post_configs"]))
    m("NumCalls", commafy(n["n_llm_calls"]))
    m("FirstSession", n["first_session"])
    m("LastSession", n["last_session"])
    m("CostBps", f"{n['cost_bps']:.0f}")
    m("PermReps", str(n["perm_reps"]))
    m("NumPredict", str(n["num_predict"]))
    # Events with complete prices at every point a position needs (backtest price_ok). Two of the
    # NumEvents sampled events fail it and are never held; the Data section says so.
    m("NumPriceOK", commafy(n["n_price_ok"]))

    # --- index membership (code/membership_audit.py; post hoc, camera-ready) --
    if "membership" in n:
        mb = n["membership"]
        m("MemRevision", str(mb["revision"]))
        m("MemRevisionDate", mb["revision_date"])
        m("MemFetched", mb["fetched"])
        m("MemConstituents", str(mb["n_constituents"]))
        m("MemListed", str(mb["n_listed"]))
        m("MemJoiners", str(mb["n_joiners"]))
        m("MemPreEvents", str(mb["n_pre_addition_events"]))
        m("MemPrePct", f"{mb['pre_addition_pct']:.1f}")
        m("MemPreFirms", str(mb["n_pre_addition_firms"]))
        m("MemRemovals", str(mb["n_removals"]))

    # --- dispersion ----------------------------------------------------------
    for k, name in (("sharpe_min", "SharpeMin"), ("sharpe_max", "SharpeMax"),
                    ("sharpe_median", "SharpeMedian"), ("sharpe_sd", "SharpeSD"),
                    ("sharpe_p5", "SharpePfive"), ("sharpe_p95", "SharpePninetyfive"),
                    ("sharpe_p25", "SharpePtwentyfive"), ("sharpe_p75", "SharpePseventyfive"),
                    ("sharpe_iqr", "SharpeIQR"), ("sharpe_best", "SharpeBest"),
                    ("sharpe_default", "SharpeDefault")):
        m(name, f"{n[k]:.2f}")
    m("SharpeRange", f"{n['sharpe_max'] - n['sharpe_min']:.2f}")
    m("NumNeverTrades", commafy(n["n_never_trades"]))
    m("FracNeverTrades", f"{100 * n['frac_never_trades']:.1f}")
    m("FracPositive", f"{100 * n['frac_positive']:.0f}")
    m("SharpeMedianLive", f"{n['sharpe_median_live']:.2f}")
    m("DefaultPercentileLive", f"{n['default_percentile_live']:.0f}")
    m("NumLiveConfigs", commafy(n["neff"]["n_live"]))
    m("FracAboveOne", f"{100 * n['frac_above_1']:.0f}")
    m("FracBelowMinusOne", f"{100 * n['frac_below_minus1']:.0f}")
    for tag, name in (("best", "CIBest"), ("default", "CIDefault"), ("median", "CIMedian")):
        lo, hi = n[f"ci_{tag}"]
        m(name, f"[{lo:.2f}, {hi:.2f}]")

    # --- variance decomposition ----------------------------------------------
    for f, lab in FACTOR_LABEL.items():
        m("VarShare" + f.capitalize(), f"{100 * n['var_share'][f]:.1f}")
    m("VarShareInteractions", f"{100 * n['var_share']['interactions']:.1f}")
    m("VarSharePostLLM", f"{100 * n['var_share_post_llm']:.1f}")
    m("VarShareLLMSide", f"{100 * n['var_share_llm_side']:.1f}")
    # \VarShareModel is already emitted by the per-factor loop above; defining it twice makes
    # LaTeX abort with "command already defined", which is how this was caught.
    m("ModelSD", f"{n['wb']['model_sd']:.2f}")
    m("HarnessSD", f"{n['wb']['harness_sd']:.2f}")
    m("ModelRange", f"{n['wb']['model_range']:.2f}")
    m("HarnessRange", f"{n['wb']['harness_range']:.2f}")
    m("SDRatio", f"{n['wb']['ratio_sd']:.1f}")
    m("BootBetween", f"{n['boot']['between_spread_mean']:.2f}")
    m("BootWithin", f"{n['boot']['within_config_sd_mean']:.2f}")
    m("BootRatio", f"{n['boot']['ratio_between_within']:.1f}")
    m("BootFracBestAboveMedian", f"{100 * n['boot']['frac_best_above_median']:.0f}")
    m("BootGapBestMedian", f"{n['boot']['gap_best_median_mean']:.2f}")
    m("BootGapBestMedianSD", f"{n['boot']['gap_best_median_sd']:.2f}")
    if "live_only" in n:
        m("VarSharePostLLMLive", f"{100 * n['live_only']['var_share_post_llm']:.1f}")
        m("VarShareModelLive", f"{100 * n['live_only']['var_share_model']:.1f}")
    m("VarShareTrimmedModel", f"{100 * n['trimmed']['var_share_model']:.1f}")
    m("VarShareTrimmedPostLLM", f"{100 * n['trimmed']['var_share_post_llm']:.1f}")
    m("NeutralOnlySD", f"{n['neutral_only']['sharpe_sd']:.2f}")
    m("NeutralOnlyModel", f"{100 * n['neutral_only']['var_share_model']:.1f}")
    m("NeutralOnlyPostLLM", f"{100 * n['neutral_only']['var_share_post_llm']:.1f}")
    m("PostOnlySD", f"{n['post_only']['sd_mean']:.2f}")
    m("PostOnlyRange", f"{n['post_only']['range_mean']:.2f}")
    m("PostOnlyRangeMax", f"{n['post_only']['range_max']:.2f}")
    m("VarShareVolModel", f"{100 * n['var_share_vol']['model']:.1f}")
    m("VarShareMeanPostLLM", f"{100 * n['var_share_mean']['post_llm']:.1f}")
    m("VarShareVolPostLLM", f"{100 * n['var_share_vol']['post_llm']:.1f}")

    # --- multiplicity ---------------------------------------------------------
    m("Neff", f"{n['n_eff']:.0f}")
    m("NeffLiJi", f"{n['neff']['li_ji']:.0f}")
    m("NeffPct", f"{100 * n['n_eff'] / n['n_configs']:.1f}")
    m("MeanCorr", f"{n['neff']['mean_pairwise_corr']:.2f}")
    # The participation ratio squares C_ij, so it responds to the RMS correlation, not the mean.
    # Printing only the mean invited the objection that a strategy and its exact negative count as
    # one trial; printing both makes the estimator's sign-blindness explicit rather than hidden.
    m("RMSCorr", f"{n['neff']['rms_pairwise_corr']:.2f}")
    m("TopEigShare", f"{100 * n['neff']['top_eig_share']:.0f}")
    m("NImplied", ("$>$" if n.get("n_implied_saturated") else "") + commafy(n["n_implied"]))
    m("FracEventsPermutable", f"{100 * n['frac_events_permutable']:.1f}")
    m("MedianEventsPerSession", f"{n['median_events_per_session']:.0f}")
    for tag, name in (("N1", "DSRone"), ("Neff", "DSRneff"), ("Nli_ji", "DSRliji"),
                      ("Nimplied", "DSRimplied"), ("Nfull", "DSRfull")):
        m(name, f"{n['dsr'][tag]:.3f}")
    m("EmaxLiJi", f"{n['emax_pred']['Nli_ji']:.2f}")
    m("SDRatioRealNull", f"{n['sd_ratio_real_null']:.2f}")
    m("NullGridSD", f"{n['null_grid_sd_mean']:.2f}")
    m("DefaultPercentile", f"{n['default_percentile']:.0f}")
    m("PostOnlySDMax", f"{n['post_only']['sd_max']:.2f}")
    m("PermPDefaultDisp", ("$>$0.99" if n["perm_p_default"] > 0.99
                           else f"{n['perm_p_default']:.2f}"))
    m("PSRBest", f"{n['psr_best']:.3f}")
    m("NullMaxMean", f"{n['null_max_mean']:.2f}")
    m("NullMaxSD", f"{n['null_max_sd']:.2f}")
    m("NullMaxPninetyfive", f"{n['null_max_p95']:.2f}")
    m("PermPBest", f"{n['perm_p_best']:.3f}")
    m("PermPDefault", f"{n['perm_p_default']:.3f}")
    for tag, name in (("N1", "EmaxOne"), ("Neff", "EmaxNeff"), ("Nfull", "EmaxFull")):
        m(name, f"{n['emax_pred'][tag]:.2f}")
    m("EmaxMeasured", f"{n['emax_measured']:.2f}")
    sp = n["split"]
    m("SplitSessions", str(sp["n_sessions_each"]))
    m("SplitSpearman", f"{sp['spearman']:.2f}")
    m("SplitISBest", f"{sp['is_best_sharpe']:.2f}")
    m("SplitOOSofISBest", f"{sp['oos_of_is_best']:.2f}")
    m("SplitOOSPercentile", f"{sp['oos_percentile_of_is_best']:.0f}")
    m("SplitOOSTopTen", f"{sp['oos_median_of_is_top10']:.2f}")
    m("SplitOOSMedian", f"{sp['oos_median_all']:.2f}")
    tries = n.get("tries_to_sharpe_1p0")
    m("TriesToOne", "n/a" if tries is None else f"{tries:.0f}")
    pd = n.get("prudent")
    if pd:
        m("PrudentConfigs", commafy(pd["n_configs"]))
        m("PrudentSD", f"{pd['sharpe_sd']:.2f}")
        m("PrudentRange", f"{pd['range']:.2f}")
        m("PrudentMin", f"{pd['sharpe_min']:.2f}")
        m("PrudentMax", f"{pd['sharpe_max']:.2f}")
    cs = n.get("cost_sweep", {})
    if cs:
        others = [k for k in ("0bp", "10bp", "25bp") if k in cs]
        m("CostLevels", ", ".join(k.replace("bp", "") for k in others[:-1])
          + " and " + others[-1].replace("bp", ""))
    for tag, name in (("0bp", "CostZero"), ("5bp", "CostFive"), ("10bp", "CostTen"),
                      ("25bp", "CostTwentyfive")):
        if tag in cs:
            m(name + "Max", f"{cs[tag]['max']:.2f}")
            m(name + "SD", f"{cs[tag]['sd']:.2f}")
            m(name + "Median", f"{cs[tag]['median']:.2f}")
    m("BestSkew", f"{n['best_skew']:.2f}")
    m("BestKurt", f"{n['best_kurtosis']:.2f}")

    # --- factor ranges (for the prose that names the biggest levers) ----------
    ranked = sorted(FACTOR_LABEL, key=lambda f: -n["factor_effects"][f]["range"])
    for i, f in enumerate(ranked[:3]):
        m(f"TopFactor{'ABC'[i]}", FACTOR_LABEL[f])
        m(f"TopFactor{'ABC'[i]}Range", f"{n['factor_effects'][f]['range']:.2f}")
    m("ModelEffectRange", f"{n['factor_effects']['model']['range']:.2f}")

    # The declared count of deviations from the frozen pre-registration, read from the ledger
    # rather than typed, so the paper cannot drift from what the ledger records.
    import re
    # The released artifact ships the REDACTED ledger, not the working one, so reading only
    # DECISIONS.md made make_macros.py fail outright inside the artifact: a reviewer could not
    # regenerate numbers.tex at all. Both files carry the same entry and amendment counts by
    # construction (code/make_redacted.py asserts it), so either serves for counting.
    m("NumAmendments", str(n["num_amendments"]))

    # The per-model, per-prompt rate at which an answer arrives in the format the prompt asked for.
    # Generated as a table body rather than as nine macros, so the paper cannot quietly omit a cell.
    short = {"qwen2.5:7b-instruct": "Qwen2.5-7B",
             "mistral:7b-instruct-v0.2-q4_0": "Mistral-7B-v0.2",
             "phi3:mini": "Phi-3-mini"}
    prompts = ["P1", "P2", "P3"]
    lines = []
    for full, name in short.items():
        cells = []
        for p in prompts:
            v = n.get("compliance", {}).get(f"{full}|{p}")
            # One decimal, not zero. Rounding manufactured two false universals: Phi-3 on P1 is
            # 0.15% and printed as "0" under a caption saying a zero cell has no tradable events,
            # and Qwen is 99.98% on two prompts and printed as "100" under a sentence saying it
            # answers all three in the requested format. Both sentences were then literally false.
            # TWO decimals, not zero and not one. Rounding manufactured two false universals:
            # Phi-3 on P1 is 0.15% and printed as "0" under a caption saying a zero cell has no
            # tradable events, and Qwen is 99.98% on two prompts, which still prints as "100.0" at
            # one decimal, under a sentence saying it answers all three in the requested format.
            cells.append("--" if v is None else f"{100 * v:.2f}")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    # The whole tabular is generated, not just its rows. Splicing rows into a tabular with \input
    # leaves the row terminator and the file's own line endings on opposite sides of the boundary,
    # which LaTeX reports as a misplaced \noalign at \bottomrule.
    body = "\n".join([
        r"\begin{tabular}{lccc}", r"\toprule",
        r" & P1 probability & P2 label & P3 JSON \\", r"\midrule",
        *lines, r"\bottomrule", r"\end{tabular}"])
    with open(os.path.join(PAPER, "compliance.tex"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("% GENERATED by code/make_macros.py. Do not edit by hand.\n" + body + "\n")

    # Table 1 is generated from the coding file rather than typed, so the marks in the paper and the
    # coded evidence cannot drift apart. A "component" coding is rendered as a dagger, not a tick:
    # an ablation that removes an analyst is not a sensitivity analysis over implementation choices,
    # and marking it with a tick would contradict the sentence in the text that says so.
    coding = json.load(open(os.path.join(os.path.dirname(HERE), "data", "paper_coding.json"),
                            encoding="utf-8"))
    cols = ["prompt_text", "holding", "entry_timing", "pipeline_sensitivity", "costs"]
    sym = {"yes": r"\checkmark", "no": "", "component": r"$\dagger$"}
    rows = []
    for p in coding["papers"]:
        label = f"{p['short']} \\cite{{{p['key']}}}"
        if all(p[c] == "unretrieved" for c in cols):
            rows.append(label + r" & \multicolumn{5}{c}{\textit{full text not machine-readable}}")
        else:
            rows.append(label + " & " + " & ".join(sym.get(p[c], "?") for c in cols))
    table1 = "\n".join([
        r"\begin{tabular}{lccccc}", r"\toprule",
        r" & prompt & holding & entry & sensi- & costs \\",
        r" & text & period & timing & tivity & \\", r"\midrule",
        " \\\\\n".join(rows), r"\\", r"\bottomrule", r"\end{tabular}"])
    with open(os.path.join(PAPER, "coding.tex"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("% GENERATED by code/make_macros.py from data/paper_coding.json.\n" + table1 + "\n")
    # Counted in analyze.py and read back from numbers.json, so that block 2 of verify_output.txt
    # contains every printed quantity, which is what the artifact tells reviewers it contains.
    m("NumCoded", str(n["num_coded"]))
    m("NumCodedTotal", str(n["num_coded_total"]))

    # The best-of-grid gap, raw and standardised by each grid's own dispersion. A best-of-N
    # statistic scales with the spread of what it maximises over, and the null grid is wider than
    # the real one, so the raw gap overstates how much of the result is about level.
    if "gap_raw" in n:
        m("GapRaw", f"{n['gap_raw']:.2f}")
        m("GapStd", f"{n['gap_standardised']:.2f}")

    if "n_events_available" in n:
        m("NumEventsAvailable", f"{n['n_events_available']:,}")
        m("NumEventsSampled", f"{n['n_events_sampled']:,}")
        m("SamplePct", f"{n['sample_pct']:.0f}")
        m("SampleSeed", str(n["event_sample_seed"]))

    rb = n.get("robust", {})
    if rb:
        # The autocorrelation-robust annualisation (Lo 2002). Reported because holding period is a
        # factor IN the grid, so the sqrt(252) factor is wrong by an amount that differs across
        # configurations, and a reviewer is entitled to ask whether the dispersion is that scaling.
        m("RobustLags", str(rb["max_lag"]))
        m("RobustSD", f"{rb['robust_sd']:.2f}")
        m("RobustBest", f"{rb['best_config_robust']:.2f}")
        m("RobustRankCorr", f"{rb['rank_corr_with_standard']:.2f}")
        m("EtaRatioMedian", f"{rb['eta_over_sqrtq_median']:.2f}")
        m("RobustSDTrimmed", f"{rb['robust_sd_trimmed']:.2f}")
        m("StdSDTrimmed", f"{rb['std_sd_trimmed']:.2f}")
        m("RobustIQR", f"{rb['robust_iqr']:.2f}")
        # The all-configuration spread recomputed over only the configurations that trade. Printed
        # deliberately, in a sentence that names the exclusion, so a reader can see that dropping
        # the never-trading cells WIDENS the spread rather than manufacturing it.
        m("StdSDLive", f"{rb['std_sd_live']:.2f}")
        m("StdIQR", f"{rb['std_iqr']:.2f}")

    od = n.get("oracle_dispersion", {})
    if od:
        # Dispersion measured on scores that carry a KNOWN signal. Without this every dispersion
        # figure in the paper sits in a regime where the scores carry nothing, and the obvious
        # objection is that implementation choices only move a number that is already noise.
        for tag, name in (("0.55", "Fiftyfive"), ("0.60", "Sixty"), ("0.70", "Seventy")):
            if tag in od["accuracies"]:
                a = od["accuracies"][tag]
                m("OracleSD" + name, f"{a['post_llm_sd_median']:.2f}")
                m("OracleSpread" + name, f"{a['range_median']:.2f}")
        m("OracleSDRatio", f"{od['ratio_oracle60_over_real']:.2f}")
        m("OracleDispDraws", str(od["draws"]))
        # The accuracy levels themselves, so the prose naming them does not hard-code a number.
        m("OracleAccLow", f"{od['acc_low_pct']:.0f}")
        m("OracleAccHigh", f"{od['acc_high_pct']:.0f}")

    cap = n.get("cap_probe")
    if cap:
        # The decode-cap ablation. A reviewer can otherwise argue that the zero-compliance cells
        # are an artefact of the 64-token cap rather than a fact about the model-prompt pair.
        m("CapProbeN", str(cap["n_filings"]))
        m("CapProbeHigh", str(cap["cap_high"]))
        ctls = list(cap["controls"].values())
        m("CapControlLow", f"{ctls[0]['old']:.1f}")
        m("CapControlHigh", f"{ctls[0]['new']:.1f}")
        # BOTH controls, because one of them fell slightly and reporting only the one that rose
        # would be selective. The claim is that the cap changed nothing material, not that every
        # number went up.
        m("CapControlTwoLow", f"{ctls[1]['old']:.1f}")
        m("CapControlTwoHigh", f"{ctls[1]['new']:.1f}")
        # The largest movement across the FAILING cells, in percentage points. Reporting one
        # cell's new value invited the sentence "the worst cell moves from zero to 0.0%", which
        # reads as a broken macro; what the ablation actually shows is that nothing moves.
        m("CapMaxMove", f"{cap['max_move_pct_points']:.1f}")
        m("CapNumCells", str(cap["n_cells"]))

    pw = n.get("power", {})
    if pw:
        mda = pw.get("min_detectable_accuracy")
        m("MinDetectAcc", "not detected at any level tested" if mda is None
          else f"{100 * float(mda):.0f}")
        for tag, name in (("0.50", "Fifty"), ("0.52", "Fiftytwo"), ("0.55", "Fiftyfive"),
                          ("0.60", "Sixty"), ("0.70", "Seventy")):
            if tag in pw["accuracies"]:
                a = pw["accuracies"][tag]
                m("Oracle" + name + "Best", f"{a['best']:.2f}")
                m("Oracle" + name + "P", f"{a['perm_p']:.3f}")
                # The rejection rate over independent oracle draws is what makes this a power
                # estimate rather than one lucky realisation; the p-value alone is not.
                if "rejection_rate" in a:
                    m("Oracle" + name + "Rej", f"{100 * a['rejection_rate']:.0f}")
        m("PowerReps", str(pw["reps"]))
        if "draws" in pw:
            m("PowerDraws", str(pw["draws"]))
        # The highest accuracy BELOW the reliable threshold: the level the paper reports as
        # unreliable rather than as a detection. Derived here rather than typed, because typing it
        # is exactly how the previous draft came to assert a detection at a level that replicates
        # in a minority of draws.
        below = sorted(a for a in pw["accuracies"] if mda is None or float(a) < float(mda))
        if below:
            m("UnreliableAcc", f"{100 * float(below[-1]):.0f}")
        if "ic_summary" in pw:
            s = pw["ic_summary"]
            m("ICMin", f"{s['min']:+.3f}")
            m("ICMax", f"{s['max']:+.3f}")
            m("ICMean", f"{s['mean']:+.3f}")
            m("ICNumSig", str(s["n_significant_at_5pct"]))
            m("ICNumConfigs", str(s["n_configs"]))

    pr = n.get("probes", {})
    if "mask" in pr and "main" in pr:
        m("MaskCorr", f"{pr['mask']['score_corr_with_main']:.2f}")
        m("MaskShift", f"{pr['mask']['mean_abs_score_shift']:.3f}")
        m("MaskMedian", f"{pr['mask']['sharpe_median']:.2f}")
        m("MaskMax", f"{pr['mask']['sharpe_max']:.2f}")
        m("MaskSD", f"{pr['mask']['sharpe_sd']:.2f}")
        m("ArmMedian", f"{pr['main']['sharpe_median']:.2f}")
        m("ArmMax", f"{pr['main']['sharpe_max']:.2f}")
        m("ArmSD", f"{pr['main']['sharpe_sd']:.2f}")
    if "scale" in pr:
        m("ScaleMedian", f"{pr['scale']['sharpe_median']:.2f}")
        m("ScaleMax", f"{pr['scale']['sharpe_max']:.2f}")
        m("ScaleSD", f"{pr['scale']['sharpe_sd']:.2f}")
        m("ScaleNullMax", f"{pr['scale']['null_max_mean']:.2f}")
        m("ScalePermP", f"{pr['scale']['perm_p']:.2f}")
        m("ScaleCompliance", f"{100 * pr['scale']['strict_compliance']:.0f}")
    if "scale550" in pr:
        # The third rung of the scale ladder. Reported alongside the 36B arm rather than as a
        # separate probe, because the point is the monotone trend across three parameter counts,
        # not any one arm.
        b = pr["scale550"]
        m("BigMax", f"{b['sharpe_max']:.2f}")
        m("BigSD", f"{b['sharpe_sd']:.2f}")
        m("BigNullMax", f"{b['null_max_mean']:.2f}")
        m("BigPermP", f"{b['perm_p']:.2f}")
        m("BigCompliance", f"{100 * b['strict_compliance']:.1f}")
        m("BigParams", n["big_model"]["params"])
        m("ArmPermP", f"{pr['main']['perm_p']:.2f}")
        # The reference arm's own compliance, so the scale arms are compared against something
        # rather than called "strictly better" against nothing. It is 100%, which makes the 550B
        # arm marginally the worst on that axis.
        m("ArmCompliance", f"{100 * pr['main']['strict_compliance']:.0f}")
        m("ArmNullMax", f"{pr['main']['null_max_mean']:.2f}")
        m("ProbePermReps", str(pr["scale"]["perm_reps"]))
        m("ScaleParams", n["scale_model"]["params"])
        m("MainParams", n["main_model_params"])
        # The scale model's release date, taken from numbers.json like every other printed
        # quantity, because it falls INSIDE the sample window and the paper's contamination
        # scope depends on it. analyze.py copies it there from data/model_provenance.json, so it
        # reaches the reviewer-facing verify_output.txt; reading the provenance file directly
        # here left the one date in the paper that no ground-truth block contained.
        m("ScaleRelease", n["scale_model"]["release"])
    if "big_model" in n:
        m("BigRelease", n["big_model"]["release"])
    if "seed_summary" in pr and pr["seed_summary"].get("n_seeds", 0) >= 2:
        m("SeedCount", str(pr["seed_summary"]["n_seeds"]))
        m("SeedMedianRange", f"{pr['seed_summary']['median_sharpe_range']:.2f}")
        m("SeedTemp", f"{n['seed_temperature']:.1f}")

    header = ("% GENERATED by code/make_macros.py from data/numbers.json. Do not edit by hand.\n"
              "% Every quantity in the manuscript resolves through one of these macros.\n")
    with open(os.path.join(PAPER, "numbers.tex"), "w", encoding="utf-8") as fh:
        fh.write(header + "\n".join(sorted(L)) + "\n")
    print(f"wrote {len(L)} macros to paper/numbers.tex")


if __name__ == "__main__":
    main()
