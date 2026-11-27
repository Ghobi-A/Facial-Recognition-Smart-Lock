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
    worst = max(min(e["curve"]["frr"]) for e in results["models"].values())
    top = 0.95 if worst > 0.3 else 0.4  # widen when a model never gets below 30% FRR
    ticks = np.array([0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4])
    if top > 0.4:
        ticks = np.array([0.001, 0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95])
    labels = [f"{t * 100:g}%" for t in ticks]
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
    ax.set_xticks(norm.ppf(ticks), labels)
    ax.set_yticks(norm.ppf(ticks), labels)
    ax.set_xlim(norm.ppf(0.0004), norm.ppf(top))
    ax.set_ylim(norm.ppf(0.0004), norm.ppf(top))
    ax.set_xlabel("False accept rate (stranger let in)", color=TEXT)
    ax.set_ylabel("False reject rate (owner locked out)", color=TEXT)
    ax.set_title(
        f"DET curves on {results['dataset'].upper()} (lower-left is better)\n"
        "curves: protocol pairs; markers: cost-based point, FAR from all stranger pairs",
        color=TEXT,
        fontsize=11,
        loc="left",
    )
    _style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


POLICY_LABELS = {
    "eer": "EER point",
    "cost_10to1": "Cost 10:1",
    "far_1e-02": "FAR = 1%",
    "far_1e-03": "FAR = 0.1%",
    "far_1e-04": "FAR = 0.01%",
    "legacy_default_0.6": "2022 default",
}
SUBGROUP_ROWS = [
    ("sex", "Sex"),
    ("ethnicity", "Ethnicity (est.)"),
    ("pose_turned", "Turned head"),
    ("profile", "Profile view"),
    ("eyeglasses", "Eyeglasses"),
    ("harsh_lighting", "Harsh lighting"),
    ("blurry", "Blurry"),
    ("young", "Young"),
]


def _bars(ax, results, policies, metric, models) -> None:
    width = 0.8 / len(models)
    for m, model in enumerate(models):
        ops = results["models"][model]["operating_points"]
        names = [p for p in policies if p in ops]
        values = np.array([ops[p][metric][0] for p in names])
        low = values - np.array([ops[p][metric][1] for p in names])
        high = np.array([ops[p][metric][2] for p in names]) - values
        y = np.array([policies.index(p) for p in names]) + (m - (len(models) - 1) / 2) * width
        ax.barh(y, values, height=width - 0.04, color=COLOURS[model], label=LABELS[model])
        ax.errorbar(values, y, xerr=[low, high], fmt="none", ecolor=TEXT, elinewidth=1, capsize=2)
    ax.set_yticks(range(len(policies)), [POLICY_LABELS[p] for p in policies])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    _style(ax)
    ax.grid(axis="y", visible=False)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()


def operating_points(results: dict, path: Path) -> None:
    """FAR and FRR at the pair-calibrated policies (EER point and cost 10:1)."""
    policies = ["eer", "cost_10to1"]
    models = list(results["models"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), dpi=150, sharey=True)
    for ax, (metric, title) in zip(
        axes, (("far", "False accept rate (all stranger pairs)"), ("frr", "False reject rate"))
    ):
        _bars(ax, results, policies, metric, models)
        ax.set_title(title, color=TEXT, fontsize=10, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=3)
    fig.suptitle(
        "Out-of-sample error rates by threshold policy (95% person-level CI)",
        color=TEXT,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def frr_at_far(results: dict, path: Path) -> None:
    """Security-constrained comparison: owner lock-out rate at fixed stranger-accept rates."""
    policies = ["far_1e-02", "far_1e-03", "far_1e-04"]
    models = list(results["models"])
    fig, ax = plt.subplots(figsize=(7.4, 3.5), dpi=150)
    _bars(ax, results, policies, "frr", models)
    ax.set_xlabel("False reject rate (owner locked out)", color=TEXT)
    ax.set_title(
        "False rejects at a fixed false accept rate\n"
        "threshold set on the other half's dense stranger pairs; 95% person-level CI",
        color=TEXT,
        fontsize=10,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def subgroups(results: dict, path: Path, metric: str, policy: str = "cost_10to1") -> None:
    """Forest plot: one row per subgroup, one marker per model, CI whiskers.

    FAR uses every same-group stranger pair (dense sets), on a log axis because it spans
    orders of magnitude between groups. FRR uses genuine pairs, on a linear axis.
    """
    models = list(results["models"])
    first = results["models"][models[0]]["subgroups"][policy]
    rows = [
        (key, title, value)
        for key, title in SUBGROUP_ROWS
        if key in first
        for value in first[key]
        if first[key][value].get(metric) is not None and first[key][value][metric][0] is not None
    ]
    fig, ax = plt.subplots(figsize=(7.6, 0.36 * len(rows) + 1.6), dpi=150)
    log = metric == "far"
    floor = 1e-5
    for m, model in enumerate(models):
        groups = results["models"][model]["subgroups"][policy]
        for r, (key, _, value) in enumerate(rows):
            point, low, high = groups[key][value][metric]
            if point is None or low is None:
                continue
            if log:
                point, low, high = max(point, floor), max(low, floor), max(high, floor)
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

    def label(key, title, value):
        cell = first[key][value]
        if key in ("sex", "ethnicity"):
            return f"{title}: {value}  ({cell['people']:,} people)"
        shown = {"yes": "yes", "no": "no"}[value]
        return f"{title}: {shown}"

    ax.set_yticks(range(len(rows)), [label(*row) for row in rows])
    ax.invert_yaxis()
    if log:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.3g}%")
        )
    else:
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    name = (
        "False reject rate (genuine pairs)"
        if metric == "frr"
        else "False accept rate (same-group stranger pairs, log scale)"
    )
    ax.set_xlabel(name, color=TEXT)
    ax.set_title(
        f"{name.split(' (')[0]} by subgroup at {POLICY_LABELS[policy]} "
        f"({results['dataset'].upper()})\n95% person-level bootstrap CI",
        color=TEXT,
        fontsize=10,
        loc="left",
    )
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.4, -0.1), ncol=3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def thermal(bench: dict, path: Path) -> None:
    """Sustained load: throughput and SoC temperature over time, two stacked panels."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.4, 4.6), dpi=150, sharex=True)
    for model, entry in bench["models"].items():
        series = entry.get("sustained")
        if not series:
            continue
        t = np.array(series["t"])
        top.plot(t, series["fps"], color=COLOURS[model], linewidth=2, label=LABELS[model])
        if any(v is not None for v in series["temp_c"]):
            temps = np.array([np.nan if v is None else v for v in series["temp_c"]], dtype=float)
            bottom.plot(t, temps, color=COLOURS[model], linewidth=2)
    if not bottom.lines:
        bottom.text(
            0.5,
            0.5,
            "no temperature sensor on this machine",
            ha="center",
            va="center",
            color=MUTED,
            transform=bottom.transAxes,
        )
        bottom.set_yticks([])
    top.set_ylabel("Probes / s", color=TEXT)
    bottom.set_ylabel("SoC temperature (°C)", color=TEXT)
    bottom.set_xlabel("Seconds under continuous load", color=TEXT)
    for ax in (top, bottom):
        _style(ax)
    top.legend(frameon=False, fontsize=8)
    board = bench["device"].get("board") or bench["device"].get("cpu", "")
    top.set_title(f"Sustained load on {board}", color=TEXT, fontsize=10, loc="left")
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
