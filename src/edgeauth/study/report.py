"""Render every results table as Markdown straight from the JSON (reports/tables.md).

The prose in reports/findings.md quotes these tables; regenerating them after a re-run keeps
the numbers and the text from drifting apart.
"""

import json
from pathlib import Path

MODEL_NAMES = {"dlib-legacy": "dlib (2022)", "sface-fp32": "SFace FP32", "sface-int8": "SFace INT8"}
DATASET_ORDER = ("lfw", "cfp-ff", "cfp-fp", "celeba", "rfw", "bfw")


def pct(cell, digits: int = 2, ci: bool = True) -> str:
    """Format (point, low, high) rates as '1.23% (0.98–1.50)'."""
    if cell is None or cell[0] is None:
        return "–"
    text = f"{cell[0] * 100:.{digits}f}%"
    if ci and len(cell) > 2 and cell[1] is not None:
        text += f" ({cell[1] * 100:.{digits}f}–{cell[2] * 100:.{digits}f})"
    return text


def ratio(a, b) -> str:
    if not a or not b or a[0] is None or b[0] in (None, 0):
        return "–"
    return f"{a[0] / b[0]:.1f}×"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def overall(results: dict) -> str:
    rows = []
    for model, entry in results["models"].items():
        ops = entry["operating_points"]
        rows.append(
            [
                MODEL_NAMES[model],
                pct(entry["eer"]),
                f"{entry['auc'][0]:.4f}",
                pct((entry["failure_to_acquire"],), ci=False),
                *(pct(ops[p]["frr"]) for p in ("far_1e-02", "far_1e-03", "far_1e-04")),
            ]
        )
    return _table(
        ["Model", "EER", "AUC", "FTA", "FRR @ FAR 1%", "FRR @ FAR 0.1%", "FRR @ FAR 0.01%"], rows
    )


def operating_points(results: dict) -> str:
    rows = []
    for model, entry in results["models"].items():
        for policy, op in entry["operating_points"].items():
            rows.append(
                [
                    MODEL_NAMES[model],
                    policy,
                    pct(op["far"], 3),
                    pct(op["frr"]),
                    pct(op["frr_pair_ci"]),
                ]
            )
    return _table(
        ["Model", "Policy", "FAR (dense, out-of-sample)", "FRR (person CI)", "FRR (pair CI)"],
        rows,
    )


def subgroup_table(results: dict, policy: str, key: str, metric: str) -> str:
    models = list(results["models"])
    first = results["models"][models[0]]["subgroups"].get(policy, {}).get(key)
    if not first:
        return ""
    rows = []
    for value, cell in first.items():
        label = f"{value} ({cell['people']:,} people)" if "people" in cell else value
        rows.append(
            [label]
            + [
                pct(results["models"][m]["subgroups"][policy][key][value].get(metric), 3)
                for m in models
            ]
        )
    return _table([key.replace("_", " ").title()] + [MODEL_NAMES[m] for m in models], rows)


def cross_dataset(all_results: dict[str, dict]) -> str:
    rows = []
    for name in DATASET_ORDER:
        results = all_results.get(name)
        if not results:
            continue
        for model, entry in results["models"].items():
            sex = entry["subgroups"]["far_1e-03"].get("sex", {})
            female, male = sex.get("female", {}), sex.get("male", {})
            rows.append(
                [
                    name.upper(),
                    MODEL_NAMES[model],
                    f"{results['people']:,}",
                    pct(entry["eer"]),
                    pct(entry["operating_points"]["far_1e-03"]["frr"]),
                    ratio(female.get("frr"), male.get("frr")),
                    ratio(female.get("far"), male.get("far")),
                ]
            )
    return _table(
        [
            "Dataset",
            "Model",
            "People",
            "EER",
            "FRR @ FAR 0.1%",
            "FRR female/male",
            "FAR female/male",
        ],
        rows,
    )


def ci_widening(results: dict) -> str:
    rows = []
    for model, entry in results["models"].items():
        for policy in ("cost_10to1", "far_1e-03"):
            op = entry["operating_points"][policy]
            person = op["frr"][2] - op["frr"][1]
            pair = op["frr_pair_ci"][2] - op["frr_pair_ci"][1]
            rows.append(
                [
                    MODEL_NAMES[model],
                    policy,
                    f"{pair * 100:.2f} pp",
                    f"{person * 100:.2f} pp",
                    f"{person / pair:.2f}×",
                ]
            )
    return _table(["Model", "Policy", "FRR CI width, pairs", "FRR CI width, people", "Ratio"], rows)


def render(results_dir: Path) -> str:
    loaded = {
        path.stem.removesuffix("_verification"): json.loads(path.read_text())
        for path in sorted(results_dir.glob("*_verification.json"))
    }
    parts = [
        "# Results tables",
        "",
        "*Generated by `edgeauth-study report` from `reports/results/`. Do not edit by hand.*",
        "",
        "Rates are percentages with 95% person-level bootstrap intervals.",
        "",
    ]
    if len(loaded) > 1:
        parts += ["## Across datasets (FAR fixed at 0.1%)", "", cross_dataset(loaded), ""]
    for name in DATASET_ORDER:
        results = loaded.get(name)
        if not results:
            continue
        parts += [
            f"## {name.upper()}",
            "",
            results.get("description") or "",
            "",
            f"{results['people']:,} people, {results['pairs']:,} pairs "
            f"({results['genuine_pairs']:,} genuine). Labels: {results.get('labels')}.",
            "",
            "### Overall",
            "",
            overall(results),
            "",
            "### Operating points",
            "",
            operating_points(results),
            "",
        ]
        if name == "lfw":
            parts += ["### Person-level vs pair-level intervals", "", ci_widening(results), ""]
        for policy, label in (("far_1e-03", "FAR fixed at 0.1%"), ("cost_10to1", "cost 10:1")):
            for key in (
                "sex",
                "ethnicity",
                "pose_turned",
                "profile",
                "eyeglasses",
                "harsh_lighting",
                "blurry",
                "young",
            ):
                for metric, what in (
                    ("far", "False accepts, same-group strangers"),
                    ("frr", "False rejects"),
                ):
                    table = subgroup_table(results, policy, key, metric)
                    if table:
                        parts += [f"### {what} by {key.replace('_', ' ')} ({label})", "", table, ""]
    return "\n".join(parts)
