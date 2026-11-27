"""edgeauth-study: reproducible LFW benchmarking pipeline.

    edgeauth-study run            # fetch -> manifest -> embed -> analyse -> benchmark -> figures

Every stage caches its output under --work (default .edgeauth-study/), so re-running only
recomputes what is missing. Results land in reports/results/, figures in reports/figures/.
If MLflow is installed, each analyse/benchmark run is also logged as an MLflow run.
"""

import argparse
import json
import sys
from pathlib import Path

from . import embedders, lfw


def _paths(args) -> dict[str, Path]:
    work = args.work
    return {
        "data": work / "lfw",
        "manifest": work / "lfw" / "pairs_manifest.csv",
        "embeddings": work / "embeddings",
        "models": work / "models",
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
        mlflow.set_experiment("edgeauth-lfw-study")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
            for artifact in artifacts:
                mlflow.log_artifact(str(artifact))
    except Exception as error:  # tracking is optional; never lose a finished analysis to it
        print(f"MLflow logging skipped: {error}", file=sys.stderr)


def fetch(args) -> None:
    lfw.fetch(_paths(args)["data"])


def manifest(args) -> None:
    paths = _paths(args)
    rows = lfw.parse_pairs(paths["data"] / "pairs.txt")
    lfw.write_manifest(rows, paths["manifest"])
    print(
        f"{paths['manifest']}  ({len(rows)} pairs, sha256 {lfw.manifest_digest(paths['manifest'])[:12]})"
    )


def embed(args) -> None:
    paths = _paths(args)
    rows = lfw.read_manifest(paths["manifest"])
    images = sorted({r.image_a for r in rows} | {r.image_b for r in rows})
    root = paths["data"] / "lfw"
    kinds = {"sface": ("sface-fp32", "sface-int8"), "dlib": ("dlib-legacy",)}
    for kind, models in kinds.items():
        if kind not in args.kinds:
            continue
        if all((paths["embeddings"] / f"{m}.npz").exists() for m in models):
            continue
        print(f"embedding {len(images)} images with {kind} ...", file=sys.stderr)
        results = embedders.embed_images(kind, root, images, paths["models"], args.workers)
        for model in models:
            embedders.save(paths["embeddings"] / f"{model}.npz", images, results, model)


def analyse(args) -> None:
    from .analysis import analyse as run

    paths = _paths(args)
    results = run(paths["manifest"], paths["data"] / "lfw_attributes.txt", paths["embeddings"])
    out = paths["results"] / "lfw_verification.json"
    _write(results, out)
    for model, entry in results["models"].items():
        ops = entry["operating_points"]
        _mlflow(
            f"lfw-{model}",
            {
                "model": model,
                "protocol": results["protocol"],
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

    paths = _paths(args)
    rows = lfw.read_manifest(paths["manifest"])
    images = sorted({r.image_a for r in rows})[: args.probes + 10]
    root = paths["data"] / "lfw"
    available = [m for m in embedders.MODELS if m != "dlib-legacy" or _has_dlib()]
    result = run([root / i for i in images], paths["models"], available)
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
    results = json.loads((paths["results"] / "lfw_verification.json").read_text())
    paths["figures"].mkdir(parents=True, exist_ok=True)
    plots.det(results, paths["figures"] / "det_curves.png")
    plots.operating_points(results, paths["figures"] / "operating_points.png")
    plots.subgroups(results, paths["figures"] / "subgroups_frr.png", "frr")
    plots.subgroups(results, paths["figures"] / "subgroups_far.png", "far")
    for bench in sorted(paths["results"].glob("benchmark_*.json")):
        label = bench.stem.removeprefix("benchmark_")
        plots.tradeoff(
            results, json.loads(bench.read_text()), paths["figures"] / f"tradeoff_{label}.png"
        )
    print(paths["figures"])


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
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch", help="download LFW, pairs.txt and attribute labels")
    commands.add_parser("manifest", help="write the 6,000-pair manifest CSV")
    e = commands.add_parser("embed", help="cache embeddings for every manifest image")
    e.add_argument("--kinds", nargs="+", default=["sface", "dlib"], choices=["sface", "dlib"])
    e.add_argument("--workers", type=int, default=4)
    commands.add_parser("analyse", help="metrics, CIs, operating points and subgroups")
    b = commands.add_parser("benchmark", help="latency / memory / power on this device")
    b.add_argument("--probes", type=int, default=200)
    b.add_argument("--label", help="results file label, e.g. pi5 (default: CPU architecture)")
    commands.add_parser("figures", help="render report figures")
    r = commands.add_parser("run", help="every stage in order")
    r.add_argument("--kinds", nargs="+", default=["sface", "dlib"], choices=["sface", "dlib"])
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--probes", type=int, default=200)
    r.add_argument("--label")
    args = cli.parse_args(argv)
    stages = {
        "fetch": [fetch],
        "manifest": [manifest],
        "embed": [embed],
        "analyse": [analyse],
        "benchmark": [benchmark],
        "figures": [figures],
        "run": [fetch, manifest, embed, analyse, benchmark, figures],
    }
    for stage in stages[args.command]:
        stage(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
