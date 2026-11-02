#!/usr/bin/env python
"""E1 figure: accuracy against the encrypted fraction, one panel per placement, one row per model.
Series: redaction (orange) and one-way attention (blue); solid = placeholder token, dotted = token
dropped from attention; dashed gray = the full model. Palette: dataviz reference slots 1 and 2."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, INK, INK2, SURFACE, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e1"
PLACEMENTS = ["random", "box", "salient", "nonsalient"]
TITLES = {"random": "Random tokens", "box": "Contiguous box / span", "salient": "Most informative tokens",
          "nonsalient": "Least informative tokens"}
SHORT = {"google/vit-base-patch16-224": "ViT-B/16, Tiny-ImageNet val (182 classes)",
         "textattack/bert-base-uncased-SST-2": "BERT-base, SST-2 val"}


def load(path):
    d = json.load(open(path))
    rows = d["rows"]
    for r in rows:
        if r["placement"] == "span":
            r["placement"] = "box"
    return d["model"], rows


def main(paths, out, ft_paths=()):
    data = [load(p) for p in paths]
    ft = {load(p)[0]: load(p)[1] for p in ft_paths}
    fig, axes = plt.subplots(len(data), 4, figsize=(13, 3.1 * len(data)), squeeze=False, sharey="row")
    fig.patch.set_facecolor(SURFACE)
    for (model, rows), axrow in zip(data, axes):
        acc_full = rows[0]["acc_full"]
        for ax, pl in zip(axrow, PLACEMENTS):
            ax.set_facecolor(SURFACE)
            ax.axhline(acc_full, color=INK2, lw=1.2, ls="--", zorder=1)
            phs = sorted({r["placeholder"] for r in rows}, key=lambda p: p == "drop")
            for ph in phs:
                sub = sorted([r for r in rows if r["placement"] == pl and r["placeholder"] == ph], key=lambda r: r["frac_enc"])
                x = [r["frac_enc"] for r in sub]
                ls = ":" if ph == "drop" else "-"
                ax.plot(x, [r["acc_redact"] for r in sub], color=ORANGE, lw=1.6, ls=ls, marker="o", ms=4.5, zorder=3)
                ax.plot(x, [r["acc_oneway"] for r in sub], color=BLUE, lw=1.6, ls=ls, marker="o", ms=4.5, zorder=4)
            fmodel = next((m for m in ft if ("vit" in m.lower()) == ("vit" in model.lower())), None)
            frows = ft.get(fmodel, []) if fmodel else []
            best = "drop" if "vit" in model else "mask"
            fsub = sorted([r for r in frows if r["placement"] == pl and r["placeholder"] == best], key=lambda r: r["frac_enc"])
            if fsub:
                ax.plot([r["frac_enc"] for r in fsub], [r["acc_oneway"] for r in fsub], color=AQUA, lw=1.6, ls="-",
                        marker="o", ms=4.5, zorder=5)
                ax.axhline(fsub[0]["acc_full"], color=AQUA, lw=1.0, ls="--", zorder=1)
            ax.set_title(TITLES[pl], fontsize=10.5, color=INK, loc="left", pad=6)
            ax.set_xlim(0, 1.02)
            ax.grid(True, color=GRID, lw=0.8, zorder=0)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)
            ax.tick_params(colors=INK2, labelsize=9)
            ax.set_xlabel("Fraction of tokens encrypted", fontsize=9, color=INK2)
        axrow[0].set_ylabel("Top-1 accuracy", fontsize=9.5, color=INK2)
        axrow[0].text(0.02, 0.04, SHORT.get(model, model), transform=axrow[0].transAxes, fontsize=9, color=INK2)
        # direct labels on the first panel, at the last solid point
        sub = sorted([r for r in rows if r["placement"] == PLACEMENTS[2] and r["placeholder"] != "drop"], key=lambda r: r["frac_enc"])
        mid = sub[len(sub) // 2]
        axrow[2].annotate("one-way", (mid["frac_enc"], mid["acc_oneway"]), xytext=(4, 6), textcoords="offset points",
                          fontsize=9, color=INK, zorder=5)
        axrow[2].annotate("redaction", (mid["frac_enc"], mid["acc_redact"]), xytext=(4, -12), textcoords="offset points",
                          fontsize=9, color=INK, zorder=5)
        axrow[2].annotate("full model", (0.03, acc_full), xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=INK2)
    handles = [plt.Line2D([], [], color=BLUE, lw=1.6, marker="o", ms=4.5, label="One-way attention (ours)"),
               plt.Line2D([], [], color=ORANGE, lw=1.6, marker="o", ms=4.5, label="Redaction (placeholder only)"),
               plt.Line2D([], [], color=AQUA, lw=1.6, marker="o", ms=4.5, label="One-way after fine-tune (best placeholder)"),
               plt.Line2D([], [], color=AQUA, lw=1.0, ls="--", label="Fine-tuned model, nothing private"),
               plt.Line2D([], [], color=INK, lw=1.6, ls="-", label="Placeholder token"),
               plt.Line2D([], [], color=INK, lw=1.6, ls=":", label="Token dropped from attention"),
               plt.Line2D([], [], color=INK2, lw=1.2, ls="--", label="Full model (full FHE)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Accuracy when only the sensitive tokens are encrypted: one-way attention keeps the full model's accuracy, "
                 "redaction does not", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    # usage: e1_plot.py <e1 jsons...> [--ft <fine-tuned jsons...>] [--out png]
    args = sys.argv[1:]
    out = os.path.expanduser("~/data/selective-fhe/e1/e1_oneway.png")
    if "--out" in args:
        i = args.index("--out"); out = args[i + 1]; args = args[:i] + args[i + 2:]
    ft_paths = []
    if "--ft" in args:
        i = args.index("--ft"); ft_paths = args[i + 1:]; args = args[:i]
    main([p for p in args if os.path.exists(p)], out, [p for p in ft_paths if os.path.exists(p)])
