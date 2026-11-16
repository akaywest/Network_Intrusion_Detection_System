"""Plots for Chapter Four (M8, implementation plan §8.5).

Colour: confusion matrices are magnitude data, so they use one sequential hue
(blue, light -> dark); cell text switches to white on dark cells for contrast.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f7fafe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
INK, MUTED = "#1f1f1d", "#6b6a66"


def plot_confusion(cm: np.ndarray, names: list[str], n_seeds: int, title: str,
                   path: Path) -> None:
    """Row-normalised confusion matrix (cell = recall share of the true class), annotated
    with the percentage and the mean count per seed. cm holds counts summed over seeds."""
    cm = np.asarray(cm, dtype=float)
    share = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    mean_count = cm / n_seeds
    n = len(names)
    fig, ax = plt.subplots(figsize=(8.2, 7.0), dpi=150)
    im = ax.imshow(share, cmap=SEQ_BLUE, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            v = share[i, j]
            ax.text(j, i, f"{v * 100:.1f}%\n({mean_count[i, j]:.1f})", ha="center",
                    va="center", fontsize=7.5, color="white" if v >= 0.55 else INK)
    ax.set_xticks(range(n), names, rotation=35, ha="right", fontsize=9, color=INK)
    ax.set_yticks(range(n), names, fontsize=9, color=INK)
    ax.set_xlabel("Predicted class", color=INK)
    ax.set_ylabel("True class", color=INK)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_xticks(np.arange(-.5, n), minor=True)
    ax.set_yticks(np.arange(-.5, n), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Share of true class (row %)", color=MUTED)
    cb.outline.set_visible(False)
    fig.text(0.01, 0.01, f"Cells: row % (mean count per seed over {n_seeds} seeds)",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- system metrics (§8.5)
# Model identity: reference categorical slots 1-3 (validated all-pairs, so lines and
# small multiples may share an axis); every series is also direct-labelled or in a
# legend, so identity is never colour alone. Latency components: slots 5-8, validated
# as adjacent pairs for a stack, with value labels (magenta is below 3:1 on white).
MODEL_COLOURS = {"iforest": "#2a78d6", "rf": "#eb6834", "lstm": "#1baf7a"}
MODEL_LABELS = {"iforest": "Isolation Forest", "rf": "Random Forest", "lstm": "LSTM"}
COMPONENT_COLOURS = {"t_tx_ms": "#e87ba4", "t_pre_ms": "#008300", "t_inf_ms": "#6250d6",
                     "t_ser_ms": "#e34948"}
COMPONENT_LABELS = {"t_tx_ms": "t_tx transfer", "t_pre_ms": "t_pre preprocess",
                    "t_inf_ms": "t_inf inference", "t_ser_ms": "t_ser store"}
SURFACE, GRID = "#fcfcfb", "#e4e3df"
# where each CDF line is labelled (quantile, side) so labels sit in empty space
CDF_LABEL_AT = {"lstm": (0.75, "left"), "iforest": (0.3, "right"),
                "rf": (0.6, "right")}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def _save(fig, path: Path) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_latency_cdf(L_by_model: dict, path: Path, target_ms: float) -> None:
    """Empirical CDF of L per model (S1, post-warm-up records pooled over seeds)."""
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=150)
    _style(ax)
    for m, L in L_by_model.items():
        if len(L) == 0:
            continue
        x = np.sort(L)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, color=MODEL_COLOURS[m], linewidth=2, label=MODEL_LABELS[m])
        med = float(np.median(x))
        q, side = CDF_LABEL_AT[m]
        ax.annotate(f"{MODEL_LABELS[m]}  (p50 {med:.0f} ms)", (np.quantile(x, q), q),
                    xytext=(-8 if side == "left" else 8, 0), textcoords="offset points",
                    ha="right" if side == "left" else "left", va="center", fontsize=8,
                    color=INK)
    ax.axvline(target_ms, color=MUTED, linestyle="--", linewidth=1.2)
    ax.text(target_ms, 0.03, f"  NFR1 target {target_ms:.0f} ms", color=MUTED, fontsize=8)
    ax.axhline(0.95, color=GRID, linestyle=":", linewidth=1)
    ax.text(0.005, 0.955, "p95", color=MUTED, fontsize=8, transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("End-to-end latency L (ms, log scale)", color=INK)
    ax.set_ylabel("Share of records", color=INK)
    ax.set_title("S1 latency distribution per model (Δt = 100 ms, 5 seeds pooled)",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _save(fig, path)


def plot_components(means: dict, path: Path) -> None:
    """Stacked mean latency components per model (S1)."""
    models = [m for m in ("iforest", "rf", "lstm") if m in means][::-1]
    fig, ax = plt.subplots(figsize=(7.6, 3.4), dpi=150)
    _style(ax)
    ax.grid(False, axis="y")
    total_max = max(sum(means[m].values()) for m in models)
    for yi, m in enumerate(models):
        left = 0.0
        for c, col in COMPONENT_COLOURS.items():
            v = means[m][c]
            ax.barh(yi, v, left=left, height=0.55, color=col, edgecolor=SURFACE,
                    linewidth=2, label=COMPONENT_LABELS[c] if yi == 0 else None)
            if v >= 0.07 * total_max:
                ax.text(left + v / 2, yi, f"{v:.1f}", ha="center", va="center",
                        fontsize=8, color="white")
            left += v
        ax.text(left + total_max * 0.01, yi, f"{left:.1f} ms", va="center", fontsize=8.5,
                color=INK)
    ax.set_yticks(range(len(models)), [MODEL_LABELS[m] for m in models], color=INK)
    ax.set_xlim(0, total_max * 1.15)
    ax.set_xlabel("Mean latency per record (ms)", color=INK)
    ax.set_title("S1 latency components (mean over 5 seeds, warm-up excluded)",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.28))
    _save(fig, path)


def plot_rate_search(runs, path: Path) -> None:
    """S3 rate search: p95(L) and throughput against Δt (mean ± SD over seeds)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.3), dpi=150)
    for ax in (a1, a2):
        _style(ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
    dts = np.array(sorted(runs.dt_ms.unique()))
    a1.plot(dts, dts, color=MUTED, linestyle="--", linewidth=1.2)
    a1.text(dts[-1], dts[-1], " p95 = Δt", color=MUTED, fontsize=8, va="bottom")
    a2.plot(dts, 1000 / dts, color=MUTED, linestyle="--", linewidth=1.2)
    a2.text(dts[0], 1000 / dts[0], " offered 1000/Δt", color=MUTED, fontsize=8,
            va="bottom")
    for m in ("iforest", "rf", "lstm"):
        d = runs[runs.model == m].groupby("dt_ms")
        if not len(d):
            continue
        p95 = d.L_p95_ms.agg(["mean", "std", "min"])
        thr = d.throughput_rps.agg(["mean", "std", "min"])
        for ax, s in ((a1, p95), (a2, thr)):
            sd = s["std"].fillna(0)
            # log axis: the lower bar stops at the lowest seed instead of going below 0
            lo = s["mean"] - np.maximum(s["mean"] - sd, s["min"])
            ax.errorbar(s.index, s["mean"], yerr=[lo, sd], color=MODEL_COLOURS[m],
                        linewidth=2, marker="o", markersize=6, capsize=3,
                        markeredgecolor=SURFACE, markeredgewidth=1.5,
                        label=MODEL_LABELS[m])
            ax.annotate(MODEL_LABELS[m], (s.index[0], s["mean"].iloc[0]),
                        xytext=(-6, 0), textcoords="offset points", ha="right",
                        va="center", fontsize=8, color=INK)
    a1.set_xlabel("Emission interval Δt (ms, log)", color=INK)
    a1.set_ylabel("p95 of L (ms, log)", color=INK)
    a1.set_title("p95 latency vs Δt", loc="left", fontsize=10.5, color=INK)
    a2.set_xlabel("Emission interval Δt (ms, log)", color=INK)
    a2.set_ylabel("Throughput, Eq. 3.7 (records/s, log)", color=INK)
    a2.set_title("Throughput vs Δt", loc="left", fontsize=10.5, color=INK)
    for ax in (a1, a2):
        ax.set_xlim(dts[0] / 2.2, dts[-1] * 1.6)
    a2.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle("S3 rate search (S1 and S2 supply Δt = 100 and 10 ms; mean ± SD over 5 "
                 "seeds, lower bar clipped at the lowest seed)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    _save(fig, path)


def plot_burst(recs, lead: int, block: int, path: Path, recovery_ms: float,
               notes: dict | None = None) -> None:
    """S4: L over time per model (small multiples, one line per seed). notes[model] is
    appended to the panel title (e.g. the recovery result)."""
    models = [m for m in ("iforest", "rf", "lstm") if m in set(recs.model)]
    fig, axes = plt.subplots(1, len(models), figsize=(11.5, 3.9), dpi=150, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        _style(ax)
        d = recs[recs.model == m]
        for _, r in d.groupby("seed"):
            r = r.sort_values("seq_no")
            ax.plot(r.t_rel_s, r.L_ms, color=MODEL_COLOURS[m], linewidth=1, alpha=0.55)
        if len(d):
            t_burst = d[d.seq_no == lead].t_rel_s.median()
            ax.axvline(t_burst, color=MUTED, linestyle=":", linewidth=1.1, zorder=0)
        ax.axhline(recovery_ms, color=MUTED, linestyle="--", linewidth=1.1)
        ax.set_yscale("log")
        note = (notes or {}).get(m)
        ax.set_title(f"{MODEL_LABELS[m]} ({d.seed.nunique()} seeds)"
                     + (f"\n{note}" if note else ""), loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel("Seconds since first emission", color=INK)
    axes[0].set_ylabel("L (ms, log)", color=INK)
    axes[0].text(0.01, recovery_ms * 1.15, f"{recovery_ms:.0f} ms", color=MUTED,
                 fontsize=8, transform=axes[0].get_yaxis_transform())
    fig.suptitle(f"S4 burst: {lead} records @ 100 ms, {block} back-to-back (dotted line), "
                 f"then the rest @ 100 ms; dashed = {recovery_ms:.0f} ms recovery threshold",
                 x=0.01, ha="left", fontsize=11, color=INK)
    _save(fig, path)
