"""edgeauth-study: reproducible face-verification benchmarking pipeline.

    edgeauth-study run                       # LFW: prepare -> embed -> analyse -> benchmark -> figures
    edgeauth-study --dataset cfp-fp run      # same pipeline on another dataset
    edgeauth-study --dataset rfw --source /data/RFW/test prepare   # datasets needing access
    edgeauth-study report                    # regenerate reports/tables.md from all results

Datasets: lfw, cfp-ff, cfp-fp, celeba (downloaded automatically); rfw, bfw (need --source).
Every stage caches its output under --work (default .edgeauth-study/), so re-running only
recomputes what is missing. Results land in reports/results/, figures in reports/figures/.
If MLflow is installed, each analyse/benchmark run is also logged as an MLflow run.
"""

import argparse
import json
import sys
from pathlib import Path

from . import embedders
from .datasets import load
from .datasets.builders import BUILDERS


def _paths(args, dataset: str | None = None) -> dict[str, Path]:
    name = dataset or args.dataset
    return {
        "dataset": args.work / "datasets" / name,
        "embeddings": args.work / "embeddings" / name,
        "models": args.work / "models",
        "results": args.reports / "results",
        "figures": args.reports / "figures",
    }


def _write(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(path)


def _mlflow(run_name: str, params: dict, metrics: dict, artifacts: list[Path]) -> None:
    try:
        import mlflow
    except ImportError:
        return
    try:
        mlflow.set_experiment("edgeauth-study")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
            for artifact in artifacts:
                mlflow.log_artifact(str(artifact))
    except Exception as error:  # tracking is optional; never lose a finished analysis to it
        print(f"MLflow logging skipped: {error}", file=sys.stderr)


def prepare(args, dataset: str | None = None) -> None:
    paths = _paths(args, dataset)
    if (paths["dataset"] / "dataset.json").exists():
        return
    name = dataset or args.dataset
    BUILDERS[name](paths["dataset"], args.source)
    print(paths["dataset"])


def embed(args) -> None:
    paths = _paths(args)
    data = load(paths["dataset"])
    images = sorted({r.image_a for r in data.rows} | {r.image_b for r in data.rows})
    kinds = {"sface": ("sface-fp32", "sface-int8"), "dlib": ("dlib-legacy",)}
    for kind, models in kinds.items():
        if kind not in args.kinds or (kind == "dlib" and not _has_dlib()):
            continue
        if all((paths["embeddings"] / f"{m}.npz").exists() for m in models):
            continue
        print(f"embedding {len(images)} {data.name} images with {kind} ...", file=sys.stderr)
        results = embedders.embed_images(
            kind, data.image_root, images, paths["models"], args.workers
        )
        for model in models:
            embedders.save(paths["embeddings"] / f"{model}.npz", images, results, model)


def analyse(args) -> None:
    from .analysis import analyse as run

    paths = _paths(args)
    data = load(paths["dataset"])
    results = run(data, paths["embeddings"])
    out = paths["results"] / f"{data.name}_verification.json"
    _write(results, out)
    for model, entry in results["models"].items():
        ops = entry["operating_points"]
        _mlflow(
            f"{data.name}-{model}",
            {
                "dataset": data.name,
                "model": model,
                "seed": results["seed"],
                "manifest_sha256": results["manifest_sha256"],
            },
            {
                "eer": entry["eer"][0],
                "auc": entry["auc"][0],
                "fta": entry["failure_to_acquire"],
                **{f"{p}_far": ops[p]["far"][0] for p in ops},
                **{f"{p}_frr": ops[p]["frr"][0] for p in ops},
            },
            [out],
        )


def benchmark(args) -> None:
    from .bench import benchmark as run

    prepare(args, "lfw")
    data = load(_paths(args, "lfw")["dataset"])
    images = sorted({r.image_a for r in data.rows})[: args.probes + 10]
    available = [m for m in embedders.MODELS if m != "dlib-legacy" or _has_dlib()]
    paths = _paths(args)
    result = run(
        [data.image_root / i for i in images],
        paths["models"],
        available,
        duration=args.duration,
    )
    name = args.label or result["device"]["machine"]
    out = paths["results"] / f"benchmark_{name}.json"
    _write(result, out)
    for model, entry in result["models"].items():
        _mlflow(
            f"bench-{name}-{model}",
            {"model": model, "device": name},
            {k: v for k, v in entry.items() if isinstance(v, (int, float))},
            [out],
        )


def figures(args) -> None:
    from . import plots

    paths = _paths(args)
    name = args.dataset
    source = paths["results"] / f"{name}_verification.json"
    results = json.loads(source.read_text())
    out = paths["figures"]
    out.mkdir(parents=True, exist_ok=True)
    prefix = "" if name == "lfw" else f"{name}_"
    plots.det(results, out / f"{prefix}det_curves.png")
    plots.frr_at_far(results, out / f"{prefix}frr_at_far.png")
    if name == "lfw":
        plots.operating_points(results, out / "operating_points.png")
    policies = ("cost_10to1", "far_1e-03") if name == "lfw" else ("far_1e-03",)
    for policy in policies:
        for metric in ("frr", "far"):
            tag = "" if policy == "cost_10to1" else "_far1e-3"
            plots.subgroups(results, out / f"{prefix}subgroups_{metric}{tag}.png", metric, policy)
    if name == "lfw":
        for bench in sorted(paths["results"].glob("benchmark_*.json")):
            label = bench.stem.removeprefix("benchmark_")
            data = json.loads(bench.read_text())
            plots.tradeoff(results, data, out / f"tradeoff_{label}.png")
            if any("sustained" in m for m in data["models"].values()):
                plots.thermal(data, out / f"thermal_{label}.png")
    print(out)


def report(args) -> None:
    from .report import render

    paths = _paths(args)
    out = args.reports / "tables.md"
    out.write_text(render(paths["results"]), encoding="utf-8")
    print(out)


def _has_dlib() -> bool:
    try:
        import face_recognition  # noqa: F401
    except ImportError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    cli.add_argument("--work", type=Path, default=Path(".edgeauth-study"))
    cli.add_argument("--reports", type=Path, default=Path("reports"))
    cli.add_argument("--dataset", default="lfw", choices=sorted(BUILDERS))
    cli.add_argument("--source", type=Path, help="local copy of the dataset (rfw, bfw)")
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="download / convert the dataset to the common format")
    e = commands.add_parser("embed", help="cache embeddings for every manifest image")
    e.add_argument("--kinds", nargs="+", default=["sface", "dlib"], choices=["sface", "dlib"])
    e.add_argument("--workers", type=int, default=4)
    commands.add_parser("analyse", help="metrics, CIs, operating points and subgroups")
    b = commands.add_parser("benchmark", help="latency / memory / power / thermals on this device")
    b.add_argument("--probes", type=int, default=200)
    b.add_argument("--duration", type=float, default=0, help="sustained-load seconds (e.g. 300)")
    b.add_argument("--label", help="results file label, e.g. pi5 (default: CPU architecture)")
    commands.add_parser("figures", help="render report figures for --dataset")
    commands.add_parser("report", help="regenerate reports/tables.md from every results file")
    r = commands.add_parser("run", help="prepare, embed, analyse, figures (+ benchmark on lfw)")
    r.add_argument("--kinds", nargs="+", default=["sface", "dlib"], choices=["sface", "dlib"])
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--probes", type=int, default=200)
    r.add_argument("--duration", type=float, default=0)
    r.add_argument("--label")
    args = cli.parse_args(argv)
    run_stages = [prepare, embed, analyse, figures]
    if args.dataset == "lfw":
        run_stages.insert(3, benchmark)
    stages = {
        "prepare": [prepare],
        "embed": [embed],
        "analyse": [analyse],
        "benchmark": [benchmark],
        "figures": [figures],
        "report": [report],
        "run": run_stages + [report],
    }
    for stage in stages[args.command]:
        stage(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
