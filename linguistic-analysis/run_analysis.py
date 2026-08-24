#!/usr/bin/env python3
"""End-to-end DiaLLM linguistic analysis.

Usage:
    python run_analysis.py --jsonl annotation_responses.jsonl \
        --annot-dir ./data/annotations --out ./results

Produces in --out:
    per_output_measures.csv          one row per generated output
    agg_by_condition.csv             means + bootstrap CIs per (family,stage,variant)
    reward_quality_gap.csv           explicit thread: density/diversity/stacking by method
    js_vs_standard.csv               JS divergence of each stage vs the standard pole
    bridge_summary.csv               human-preference deltas (if annotations present)
    tables.tex                       LaTeX for the two headline tables
    fig_density_vs_diversity.png, fig_stacking_by_method.png
"""
from __future__ import annotations
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diallm_analysis.analyze import (
    measure_doc, bootstrap_ci, category_distribution, js_divergence,
)
from diallm_analysis.features import FEATURE_CATEGORIES
from diallm_analysis.bridge import (
    load_annotations, build_bridge, summarise_bridge,
)

VARIANT_TO_VARIETY = {"aus": "aus", "ind": "ind", "brit": "uk",
                      "all": "all", "base": "all"}
STANDARD_STAGES = {"base", "instruct"}          # standard-English pole
EXPLICIT_VARIANTS = {"aus", "ind", "brit"}
ALIGN_STAGES = ["sft", "dpo", "grpo", "gspo"]
CORE = ["density_per1k", "diversity_types", "stacking_per_sent",
        "british_per1k", "feat_lexical", "ttr", "mtld",
        "contraction_per1k", "politeness_per1k"]


def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--annot-dir", default=None,
                    help="dir with annotations_*.csv for the human-preference bridge")
    ap.add_argument("--out", default="./results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = load_rows(args.jsonl)
    print(f"Loaded {len(rows)} outputs. Measuring (spaCy)...")

    # ---- per-output measures ----
    responses = {}            # (family,stage,variant,prompt_id) -> raw text, for bridge
    records = []
    for r in rows:
        variety = VARIANT_TO_VARIETY.get(r["variant"], "all")
        m = measure_doc(r["response"], variety)
        m.update({k: r[k] for k in ("family", "stage", "variant", "prompt_id", "domain")})
        m["variety_measured"] = variety
        records.append(m)
        responses[(r["family"], r["stage"], r["variant"], r["prompt_id"])] = r["response"]
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(args.out, "per_output_measures.csv"), index=False)
    print(f"  wrote per_output_measures.csv ({len(df)} rows)")

    # ---- aggregate by condition with bootstrap CIs on the core trio ----
    agg_rows = []
    for (fam, stg, var), g in df.groupby(["family", "stage", "variant"]):
        row = {"family": fam, "stage": stg, "variant": var, "n": len(g)}
        for col in CORE:
            mean, lo, hi = bootstrap_ci(g[col].values)
            row[col] = mean
            if col in ("density_per1k", "diversity_types", "stacking_per_sent"):
                row[f"{col}_lo"] = lo
                row[f"{col}_hi"] = hi
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows).sort_values(["family", "stage", "variant"])
    agg.to_csv(os.path.join(args.out, "agg_by_condition.csv"), index=False)
    print("  wrote agg_by_condition.csv")

    # ---- reward-quality gap table (explicit thread) ----
    rqg = []
    expl = df[df.variant.isin(EXPLICIT_VARIANTS)]
    for (fam, var), g in expl.groupby(["family", "variant"]):
        for stg in ALIGN_STAGES:
            gg = g[g.stage == stg]
            if gg.empty:
                continue
            rqg.append({
                "family": fam, "variety": var, "method": stg, "n": len(gg),
                "density_per1k": round(gg.density_per1k.mean(), 3),
                "diversity_types": round(gg.diversity_types.mean(), 3),
                "stacking_per_sent": round(gg.stacking_per_sent.mean(), 3),
                "british_per1k": round(gg.british_per1k.mean(), 3),
                "ttr": round(gg.ttr.mean(), 4),
            })
    rqg = pd.DataFrame(rqg)
    rqg.to_csv(os.path.join(args.out, "reward_quality_gap.csv"), index=False)
    print("  wrote reward_quality_gap.csv")

    # ---- JS divergence of each stage vs the standard pole (per family,variety) ----
    js_rows = []
    for fam in df.family.unique():
        fam_df = df[df.family == fam]
        std = fam_df[fam_df.stage.isin(STANDARD_STAGES)]
        if std.empty:
            continue
        std_dist = category_distribution(std.to_dict("records"))
        for var in sorted(EXPLICIT_VARIANTS):
            for stg in ALIGN_STAGES:
                cond = fam_df[(fam_df.stage == stg) & (fam_df.variant == var)]
                if cond.empty:
                    continue
                d = category_distribution(cond.to_dict("records"))
                js_rows.append({"family": fam, "variety": var, "method": stg,
                                "js_vs_standard": round(js_divergence(std_dist, d), 4)})
    pd.DataFrame(js_rows).to_csv(os.path.join(args.out, "js_vs_standard.csv"), index=False)
    print("  wrote js_vs_standard.csv")

    # ---- human-preference bridge ----
    if args.annot_dir and os.path.isdir(args.annot_dir):
        annot = load_annotations(args.annot_dir)
        if not annot.empty:
            bridge = build_bridge(annot, responses, family="llama")
            summ = summarise_bridge(bridge)
            summ.to_csv(os.path.join(args.out, "bridge_summary.csv"), index=False)
            bridge.to_csv(os.path.join(args.out, "bridge_pairs.csv"), index=False)
            print(f"  wrote bridge_summary.csv ({len(bridge)} pairwise trials used)")

    # ---- LaTeX for the headline reward-quality-gap table (Llama, en-IN as example) ----
    _write_latex(rqg, os.path.join(args.out, "tables.tex"))

    # ---- plots ----
    _plot_density_vs_diversity(expl, os.path.join(args.out, "fig_density_vs_diversity.png"))
    _plot_stacking(rqg, os.path.join(args.out, "fig_stacking_by_method.png"))

    _print_headlines(rqg)
    print(f"\nDone. Results in {args.out}/")


def _write_latex(rqg, path):
    if rqg.empty:
        open(path, "w").write("% no data\n")
        return
    lines = [r"\begin{table}[t]\centering\small",
             r"\begin{tabular}{lll rrr}", r"\toprule",
             r"Family & Variety & Method & Density & Diversity & Stacking \\",
             r"\midrule"]
    for fam in rqg.family.unique():
        for var in sorted(rqg[rqg.family == fam].variety.unique()):
            sub = rqg[(rqg.family == fam) & (rqg.variety == var)]
            best = sub.density_per1k.max()
            for _, r in sub.iterrows():
                dens = f"\\textbf{{{r.density_per1k:.2f}}}" if r.density_per1k == best else f"{r.density_per1k:.2f}"
                lines.append(f"{fam} & {var} & {r.method.upper()} & {dens} & "
                             f"{r.diversity_types:.2f} & {r.stacking_per_sent:.2f} \\\\")
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Independent linguistic measures on the explicit thread. "
              r"Density = dialectal features /1k tokens (what the reward maximises); "
              r"Diversity = distinct feature types; Stacking = features per sentence. "
              r"GRPO maximising density while diversity falls / stacking rises is the "
              r"linguistic signature of the reward--quality gap.}",
              r"\label{tab:ling}", r"\end{table}"]
    open(path, "w").write("\n".join(lines))


def _plot_density_vs_diversity(expl, path):
    if expl.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {"sft": "#4C72B0", "dpo": "#55A868", "grpo": "#C44E52", "gspo": "#8172B3"}
    for stg in ALIGN_STAGES:
        g = expl[expl.stage == stg]
        if g.empty:
            continue
        ax.scatter(g.density_per1k, g.diversity_types, s=14, alpha=0.5,
                   label=stg.upper(), color=colors.get(stg))
    ax.set_xlabel("Dialectal feature density (/1k tokens)")
    ax.set_ylabel("Feature-type diversity")
    ax.set_title("Density vs diversity (explicit thread, all families)")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _plot_stacking(rqg, path):
    if rqg.empty:
        return
    piv = rqg.groupby("method")["stacking_per_sent"].mean().reindex(ALIGN_STAGES)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([m.upper() for m in piv.index], piv.values,
           color=["#4C72B0", "#55A868", "#C44E52", "#8172B3"])
    ax.set_ylabel("Mean features per sentence (stacking)")
    ax.set_title("Feature stacking by alignment method")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _print_headlines(rqg):
    if rqg.empty:
        return
    print("\n=== Headline: mean density / diversity / stacking by method (explicit thread) ===")
    g = rqg.groupby("method")[["density_per1k", "diversity_types", "stacking_per_sent"]].mean()
    print(g.reindex(ALIGN_STAGES).round(3).to_string())


if __name__ == "__main__":
    main()
