"""Generate the README figures from the recorded training history and benchmarks.

Reads ``results/training_history.json`` and ``results/benchmarks.json`` and
writes PNGs into ``docs/``. Nothing here invents numbers - if a results file is
missing, the corresponding figure is skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Validated categorical palette (light surface). Slots are assigned in fixed
# order and never cycled.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
S1 = "#2a78d6"   # blue
S2 = "#eb6834"   # orange
S3 = "#1baf7a"   # aqua

RESULTS = Path("results")
DOCS = Path("docs")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "savefig.facecolor": SURFACE,
    "savefig.bbox": "tight",
    "savefig.dpi": 160,
})


def _clean(ax, xlabel: str = "", ylabel: str = "") -> None:
    """Recessive chrome: no top/right spines, hairline remainder."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)


# --------------------------------------------------------------------- #
def figure_loss() -> None:
    path = RESULTS / "training_history.json"
    if not path.exists():
        print("skip loss figure: no training_history.json")
        return
    hist = json.loads(path.read_text())
    it = [h["iter"] for h in hist]
    tr = [h["train"] for h in hist]
    va = [h["val"] for h in hist]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(it, tr, color=S1, linewidth=2, label="train")
    ax.plot(it, va, color=S2, linewidth=2, label="validation")

    # Direct labels at the line ends - identity is never carried by color alone.
    ax.annotate("train", (it[-1], tr[-1]), xytext=(6, 0), textcoords="offset points",
                color=S1, fontweight="bold", va="center", fontsize=9.5)
    ax.annotate("validation", (it[-1], va[-1]), xytext=(6, 0), textcoords="offset points",
                color=S2, fontweight="bold", va="center", fontsize=9.5)

    best = min(va)
    # Stated as a caption rather than an arrow annotation: at this scale the
    # curves converge into the bottom-right corner and a leader line lands on
    # top of the direct labels.
    ax.text(0.97, 0.62, f"best validation loss  {best:.3f}", transform=ax.transAxes,
            ha="right", color=INK_2, fontsize=9.5)

    ax.set_title("Training and validation loss")
    _clean(ax, "iteration", "cross-entropy loss (nats/token)")
    ax.set_xlim(0, it[-1] * 1.14)
    ax.legend(loc="upper right", labelcolor=INK_2)
    fig.savefig(DOCS / "loss_curve.png")
    plt.close(fig)
    print("wrote docs/loss_curve.png")


def figure_kv_cache() -> None:
    path = RESULTS / "benchmarks.json"
    if not path.exists():
        print("skip kv figure: no benchmarks.json")
        return
    rows = json.loads(path.read_text())["kv_cache"]
    n = [r["tokens"] for r in rows]
    naive = [r["naive_s"] for r in rows]
    cached = [r["cached_s"] for r in rows]
    speed = [r["speedup"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # Left: wall-clock time. Separate subplots rather than a second y-axis.
    ax1.plot(n, naive, color=S2, linewidth=2, marker="o", markersize=5,
             markeredgecolor=SURFACE, markeredgewidth=1.5, label="no cache")
    ax1.plot(n, cached, color=S1, linewidth=2, marker="o", markersize=5,
             markeredgecolor=SURFACE, markeredgewidth=1.5, label="KV cache")
    ax1.annotate("no cache", (n[-1], naive[-1]), xytext=(-4, 10),
                 textcoords="offset points", color=S2, fontweight="bold",
                 ha="right", fontsize=9.5)
    ax1.annotate("KV cache", (n[-1], cached[-1]), xytext=(-4, 10),
                 textcoords="offset points", color=S1, fontweight="bold",
                 ha="right", fontsize=9.5)
    ax1.set_title("Generation time")
    _clean(ax1, "tokens generated", "seconds")
    ax1.legend(loc="upper left", labelcolor=INK_2)

    # Right: the speedup ratio.
    bars = ax2.bar([str(x) for x in n], speed, color=S1, width=0.62)
    for rect, s in zip(bars, speed):
        ax2.annotate(f"{s:.1f}x", (rect.get_x() + rect.get_width() / 2, s),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", color=INK_2, fontsize=9, fontweight="bold")
    ax2.axhline(1.0, color=AXIS, linewidth=1, linestyle="--")
    ax2.set_title("Speedup from KV caching")
    _clean(ax2, "tokens generated", "x faster than no cache")
    ax2.set_ylim(0, max(speed) * 1.2)

    fig.tight_layout()
    fig.savefig(DOCS / "kv_cache_speedup.png")
    plt.close(fig)
    print("wrote docs/kv_cache_speedup.png")


def figure_quantization() -> None:
    path = RESULTS / "benchmarks.json"
    if not path.exists():
        print("skip quant figure: no benchmarks.json")
        return
    q = json.loads(path.read_text())["quantization"]
    labels = ["fp32", "int8\n(from scratch,\nweight-only)", "int8\n(torch dynamic)"]
    keys = list(q.keys())
    sizes = [q[k]["size_mb"] for k in keys]
    tps = [q[k]["tok_per_s"] for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    b1 = ax1.bar(labels, sizes, color=[S1, S3, S2], width=0.6)
    for rect, v in zip(b1, sizes):
        ax1.annotate(f"{v:.2f} MB", (rect.get_x() + rect.get_width() / 2, v),
                     xytext=(0, 4), textcoords="offset points", ha="center",
                     color=INK_2, fontsize=9, fontweight="bold")
    ax1.set_title("Model size in memory")
    _clean(ax1, "", "megabytes")
    ax1.set_ylim(0, max(sizes) * 1.18)

    b2 = ax2.bar(labels, tps, color=[S1, S3, S2], width=0.6)
    for rect, v in zip(b2, tps):
        ax2.annotate(f"{v:.0f}", (rect.get_x() + rect.get_width() / 2, v),
                     xytext=(0, 4), textcoords="offset points", ha="center",
                     color=INK_2, fontsize=9, fontweight="bold")
    ax2.set_title("Decoding throughput")
    _clean(ax2, "", "tokens / second")
    ax2.set_ylim(0, max(tps) * 1.18)

    fig.tight_layout()
    fig.savefig(DOCS / "quantization.png")
    plt.close(fig)
    print("wrote docs/quantization.png")


def figure_window_policy() -> None:
    path = RESULTS / "benchmarks.json"
    if not path.exists():
        return
    wp = json.loads(path.read_text()).get("window_policy")
    if not wp:
        return

    order = ["no cache (reference)", "reprefill", "evict"]
    labels = ["no cache\n(reference)", "reprefill\n(default)", "evict\n(slide window)"]
    vals = [wp[k]["val_loss_past_window"] for k in order]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bars = ax.bar(labels, vals, color=[S1, S3, S2], width=0.58)
    for rect, v, key in zip(bars, vals, order):
        d = wp[key]["delta_vs_reference"]
        note = f"{v:.3f}" if key == order[0] else f"{v:.3f}  ({d:+.3f})"
        ax.annotate(note, (rect.get_x() + rect.get_width() / 2, v),
                    xytext=(0, 5), textcoords="offset points", ha="center",
                    color=INK_2, fontsize=9.5, fontweight="bold")

    ax.axhline(vals[0], color=AXIS, linewidth=1, linestyle="--")
    ax.set_title("Cost of each long-context cache policy")
    _clean(ax, "", "loss on held-out text past the window\n(nats/token, lower is better)")
    ax.set_ylim(0, max(vals) * 1.2)
    fig.savefig(DOCS / "window_policy.png")
    plt.close(fig)
    print("wrote docs/window_policy.png")


def figure_kv_memory() -> None:
    path = RESULTS / "benchmarks.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text())["kv_memory"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    colors = {1: S1, 8: S2, 32: S3}
    for batch in (1, 8, 32):
        sub = [r for r in rows if r["batch"] == batch]
        xs = [r["seq_len"] for r in sub]
        ys = [r["kv_cache_mb"] for r in sub]
        ax.plot(xs, ys, color=colors[batch], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"batch {batch}")
        ax.annotate(f"batch {batch}", (xs[-1], ys[-1]), xytext=(6, 0),
                    textcoords="offset points", color=colors[batch],
                    fontweight="bold", va="center", fontsize=9.5)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_title("KV cache memory grows linearly with context and batch")
    _clean(ax, "context length (tokens)", "KV cache size (MB, log scale)")
    ax.set_xlim(right=8192 * 3)
    ax.legend(loc="upper left", labelcolor=INK_2)
    fig.savefig(DOCS / "kv_memory.png")
    plt.close(fig)
    print("wrote docs/kv_memory.png")


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    figure_loss()
    figure_kv_cache()
    figure_quantization()
    figure_window_policy()
    figure_kv_memory()


if __name__ == "__main__":
    main()
