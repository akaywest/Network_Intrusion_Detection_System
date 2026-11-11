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
