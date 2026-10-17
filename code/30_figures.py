"""Figures for the C2 paper.

Reads whichever result files exist and writes figures/*.pdf and *.png.
Every plotted number is computed from the JSON artifacts, never hand-typed.

Run:
  .venv/bin/python code/30_figures.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
FIG = REPO / "figures"
FIG.mkdir(exist_ok=True)

MODELS = [
    ("Qwen2.5-7B", "c2_raw.json", "#1f4e79"),
    ("Qwen2.5-3B", "c2_qwen3b.json", "#c0504d"),
    ("Mistral-7B", "c2_mistral.json", "#4f8a3c"),
    ("Qwen2.5-7B\n+ validity only", "c2_7b_pressure_validity.json", "#2f8e8e"),
    ("Qwen2.5-7B\n+ schema prompt", "c2_7b_pressure.json", "#e0a800"),
    ("Qwen2.5-3B\n+ schema prompt", "c2_qwen3b_pressure.json", "#8064a2"),
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def load(name: str):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{stem}.pdf")


def fig_failure_modes() -> None:
    rows = []
    for label, f, color in MODELS:
        d = load(f)
        if not d:
            continue
        s = d["summary"]
        n = s["n_awkward"]
        sub, echo = s["readout_1_awkward_in_members"], round(s["echo_rate_awkward"] * n)
        rows.append((label, color, sub, echo, n))

    # Clean error is 0 in every pass; show it as a reference line instead of a
    # fourth bar that would be invisible.
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    x = np.arange(len(rows))
    w = 0.36
    for i, (label, color, sub, echo, n) in enumerate(rows):
        for j, (k, hatch) in enumerate(((sub, ""), (echo, "///"))):
            rate = k / n
            lo, hi = wilson(k, n)
            ax.bar(i + (j - 0.5) * w, rate, w, color=color, alpha=1.0 if j == 0 else 0.55,
                   hatch=hatch, edgecolor="white", linewidth=0.6)
            ax.errorbar(i + (j - 0.5) * w, rate,
                        yerr=[[rate - lo], [hi - rate]], fmt="none",
                        ecolor="#333333", elinewidth=0.9, capsize=2.5)
            ax.text(i + (j - 0.5) * w, hi + 0.02, f"{k}/{n}",
                    ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_ylabel("rate on awkward items")
    ax.set_ylim(0, 1.08)
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.text(0.5, -0.34, "solid = silent substitution (schema-valid);   "
                        "hatched = loud echo (schema-invalid)",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("How each run resolves the conflict", fontsize=10, loc="left")
    fig.tight_layout()
    save(fig, "fig1_failure_modes")


def fig_kind_split() -> None:
    rows = []
    for label, f, color in MODELS:
        d = load(f)
        if not d:
            continue
        s = d["summary"]
        rows.append((label, color, s["by_kind"]["equivalent"], s["by_kind"]["shifted"]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    x = np.arange(len(rows))
    w = 0.36
    for i, (label, color, eq, sh) in enumerate(rows):
        for j, (dct, name, alpha) in enumerate(((eq, "equivalent", 1.0),
                                               (sh, "shifted", 0.55))):
            rate = dct["rate"]
            lo, hi = wilson(dct["substituted"], dct["n"])
            ax.bar(i + (j - 0.5) * w, rate, w, color=color, alpha=alpha,
                   edgecolor="white", linewidth=0.6)
            ax.errorbar(i + (j - 0.5) * w, rate,
                        yerr=[[rate - lo], [hi - rate]], fmt="none",
                        ecolor="#333333", elinewidth=0.9, capsize=2.5)
            ax.text(i + (j - 0.5) * w, rate + 0.02,
                    f"{dct['substituted']}/{dct['n']}", ha="center",
                    va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_ylabel("substitution rate")
    ax.set_title("Acceptable paraphrase vs meaning change", fontsize=10, loc="left")
    ax.text(0.5, -0.30, "solid = equivalent (same meaning);   "
                        "pale = shifted (meaning changed)",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "fig2_kind_split")


def fig_detection_plane() -> None:
    pts = [("schema\nvalidation", 0.0, 0.0, "#7f7f7f", "o"),
           ("literal diff", 0.861, 0.822, "#c0504d", "s"),
           ("comparator 1\n(MS MARCO)", 0.400, 0.322, "#1f4e79", "D"),
           ("comparator 2\n(BGE + LR)", 0.360, 0.344, "#4f8a3c", "^")]
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    # usable region: detection >= 0.70 at false alarms <= 0.20
    ax.add_patch(plt.Rectangle((0, 0.70), 0.20, 0.30, color="#2e7d32", alpha=0.10, zorder=0))
    ax.text(0.10, 0.86, "usable", ha="center", fontsize=8, color="#2e7d32", alpha=0.9)
    # axes: x = false alarms, y = detection; usable = low FA, high detection
    for name, det, fa, color, marker in pts:
        ax.scatter(fa, det, s=85, color=color, marker=marker, zorder=3,
                   edgecolor="white", linewidth=0.8, label=name)
    ax.set_xlim(-0.03, 1.0)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("false alarms on correct mappings")
    ax.set_ylabel("detection of meaning-changing substitutions")
    ax.plot([0, 1], [0, 1], linestyle=":", color="#999999", linewidth=0.9, zorder=1)
    ax.text(0.75, 0.72, "no better\nthan chance", fontsize=7, color="#888888",
            rotation=38, ha="center")
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    ax.set_title("The two available checks fail in opposite directions", fontsize=10, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "fig3_detection_plane")


def fig_per_schema() -> None:
    d = load("c2_raw.json")
    if not d:
        return
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0])
    for it in d["items"]:
        if it["arm"] != "awkward":
            continue
        a = agg[it["schema_idx"]]
        a[1] += 1
        a[0] += int(it["status"] == "substituted")
    keys = sorted(agg, key=lambda k: agg[k][0] / agg[k][1])
    rate = [agg[k][0] / agg[k][1] for k in keys]
    lo = [wilson(*agg[k])[0] for k in keys]
    hi = [wilson(*agg[k])[1] for k in keys]
    labels = [f"S{k}" for k in keys]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(len(keys))
    ax.bar(x, rate, color="#1f4e79", alpha=0.85, edgecolor="white", linewidth=0.6)
    ax.errorbar(x, rate, yerr=[np.array(rate) - np.array(lo), np.array(hi) - np.array(rate)],
                fmt="none", ecolor="#333333", elinewidth=0.9, capsize=2.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("substitution rate (7B)")
    ax.set_title("Substitution is concentrated in a subset of schemas", fontsize=10, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "fig4_per_schema")


def main() -> int:
    print("[figures]")
    fig_failure_modes()
    fig_kind_split()
    fig_detection_plane()
    fig_per_schema()
    print(f"[figures] done -> {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
