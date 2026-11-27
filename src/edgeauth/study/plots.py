"""Static report figures (PNG) for reports/findings.md. Aggregates only - no face images."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

# Fixed categorical order (validated palette slots 1-3); colour follows the model, never rank.
COLOURS = {"dlib-legacy": "#257de0", "sface-fp32": "#f47230", "sface-int8": "#00b894"}
LABELS = {
    "dlib-legacy": "dlib (2022 legacy)",
    "sface-fp32": "SFace FP32",
    "sface-int8": "SFace INT8",
}
MARKERS = {"dlib-legacy": "o", "sface-fp32": "s", "sface-int8": "D"}
LABEL_OFFSETS = {"dlib-legacy": (8, 4), "sface-fp32": (8, -12), "sface-int8": (8, 6)}
TEXT, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def det(results: dict, path: Path) -> None:
    """DET curve on normal-deviate axes: the standard biometric trade-off plot."""
    ticks = np.array([0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4])
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=150)
    for model, entry in results["models"].items():
        far = np.clip(np.array(entry["curve"]["far"]), 1e-4, 1 - 1e-4)
        frr = np.clip(np.array(entry["curve"]["frr"]), 1e-4, 1 - 1e-4)
        ax.plot(
            norm.ppf(far),
            norm.ppf(frr),
            color=COLOURS[model],
            linewidth=2,
            label=f"{LABELS[model]}  (EER {entry['eer'][0]:.1%})",
        )
        point = entry["operating_points"]["cost_10to1"]
        ax.plot(
            norm.ppf(max(point["far"][0], 5e-4)),
            norm.ppf(max(point["frr"][0], 5e-4)),
            MARKERS[model],
            color=COLOURS[model],
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )
    ax.plot(norm.ppf(ticks), norm.ppf(ticks), color=MUTED, linestyle=":", linewidth=1)
    ax.set_xticks(norm.ppf(ticks), [f"{t:.1%}".replace(".0%", "%") for t in ticks])
    ax.set_yticks(norm.ppf(ticks), [f"{t:.1%}".replace(".0%", "%") for t in ticks])
    ax.set_xlim(norm.ppf(0.0004), norm.ppf(0.4))
    ax.set_ylim(norm.ppf(0.0004), norm.ppf(0.4))
    ax.set_xlabel("False accept rate (stranger let in)", color=TEXT)
    ax.set_ylabel("False reject rate (owner locked out)", color=TEXT)
    ax.set_title(
        "DET curves on LFW (lower-left is better)\nmarkers: cost-based operating point",
        color=TEXT,
        fontsize=11,
        loc="left",
    )
    _style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def operating_points(results: dict, path: Path) -> None:
    policies = [("far_1pct", "FAR ≤ 1%"), ("cost_10to1", "Cost 10:1"), ("eer", "EER point")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=150, sharey=True)
    models = list(results["models"])
    width = 0.8 / len(models)
    for ax, (metric, title) in zip(
        axes, (("far", "False accept rate"), ("frr", "False reject rate"))
    ):
        for m, model in enumerate(models):
            ops = results["models"][model]["operating_points"]
            names = [p for p, _ in policies]
            values = np.array([ops[p][metric][0] for p in names])
            low = values - np.array([ops[p][metric][1] for p in names])
            high = np.array([ops[p][metric][2] for p in names]) - values
            y = np.arange(len(names)) + (m - (len(models) - 1) / 2) * width
            ax.barh(y, values, height=width - 0.04, color=COLOURS[model], label=LABELS[model])
            ax.errorbar(
                values, y, xerr=[low, high], fmt="none", ecolor=TEXT, elinewidth=1, capsize=2
            )
        ax.set_yticks(range(len(policies)), [label for _, label in policies])
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=2))
        ax.set_title(title, color=TEXT, fontsize=10, loc="left")
        _style(ax)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(
        "Out-of-sample error rates by threshold policy (95% bootstrap CI)",
        color=TEXT,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _field(key: str, metric: str) -> str:
    return "far_matched" if metric == "far" and key in ("sex", "ethnicity") else metric


def _count(key: str, metric: str, default: str) -> str:
    return "matched_impostor_pairs" if _field(key, metric) == "far_matched" else default


def subgroups(results: dict, path: Path, metric: str) -> None:
    """Forest plot: one row per subgroup, one dot per model, CI whiskers."""
    order = [
        ("sex", "Sex"),
        ("ethnicity", "Ethnicity (est.)"),
        ("pose", "Pose"),
        ("eyeglasses", "Eyeglasses"),
        ("harsh_lighting", "Harsh lighting"),
        ("blurry", "Blurry"),
    ]
    models = list(results["models"])
    first = results["models"][models[0]]["subgroups"]
    rows = [(key, title, value) for key, title in order if key in first for value in first[key]]
    count = "genuine_pairs" if metric == "frr" else "impostor_pairs"
    fig, ax = plt.subplots(figsize=(7.4, 0.36 * len(rows) + 1.4), dpi=150)
    for m, model in enumerate(models):
        groups = results["models"][model]["subgroups"]
        for r, (key, _, value) in enumerate(rows):
            cell = groups[key][value]
            point, low, high = cell[_field(key, metric)]
            if point is None:
                continue
            y = r + (m - (len(models) - 1) / 2) * 0.22
            ax.errorbar(
                point,
                y,
                xerr=[[point - low], [high - point]],
                fmt=MARKERS[model],
                color=COLOURS[model],
                markersize=6,
                elinewidth=1.2,
                capsize=0,
                markeredgecolor="white",
                label=LABELS[model] if r == 0 else None,
            )
    labels = [
        f"{title}: {value}  (n={first[key][value][_count(key, metric, count)]:,})"
        for key, title, value in rows
    ]
    ax.set_yticks(range(len(rows)), labels)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    name = (
        "False reject rate (genuine pairs)"
        if metric == "frr"
        else ("False accept rate (same-group stranger pairs)")
    )
    ax.set_xlabel(name, color=TEXT)
    ax.set_title(
        f"{name.split(' (')[0]} by subgroup\n"
        "cost-based threshold, 95% bootstrap CI, n = pairs in group",
        color=TEXT,
        fontsize=10,
        loc="left",
    )
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.4, -0.09), ncol=3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def tradeoff(results: dict, bench: dict, path: Path) -> None:
    """Accuracy vs latency and vs memory: two panels, never a dual axis."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=150, sharey=True)
    for ax, (key, xlabel) in zip(
        axes,
        (
            ("latency_ms_p50", "Median latency per probe (ms)"),
            ("model_bytes", "Model size on disk (MB)"),
        ),
    ):
        for model, entry in results["models"].items():
            if model not in bench["models"]:
                continue
            x = bench["models"][model][key]
            x = x / 1e6 if key == "model_bytes" else x
            point, low, high = entry["eer"]
            ax.errorbar(
                x,
                point,
                yerr=[[point - low], [high - point]],
                fmt=MARKERS[model],
                color=COLOURS[model],
                markersize=9,
                markeredgecolor="white",
                elinewidth=1.2,
                label=LABELS[model],
            )
            ax.annotate(
                LABELS[model],
                (x, point),
                xytext=LABEL_OFFSETS[model],
                textcoords="offset points",
                fontsize=8,
                color=TEXT,
            )
        ax.set_xlabel(xlabel, color=TEXT)
        ax.set_xlim(left=0)
        ax.margins(x=0.25)
        _style(ax)
    axes[0].set_ylabel("Equal error rate (lower is better)", color=TEXT)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    board = bench["device"].get("board") or bench["device"].get("cpu", bench["device"]["machine"])
    fig.suptitle(
        f"Accuracy vs edge cost - measured on {board}", color=TEXT, fontsize=11, x=0.01, ha="left"
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
