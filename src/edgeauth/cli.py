"""Local CLI for the verification lab."""

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import cv2

from .benchmark import benchmark
from .evaluation import evaluate, evaluate_pad, load_pairs, synthetic_report
from .models import PAD, RECOGNIZERS, YUNET, download_model
from .vault import TemplateVault, create_key, subject_id
from .verification import DEFAULT_PAD_CUTOFF, Screen, build_template, verification_decision
from .vision import CaptureRejected, FacePipeline, PresentationDetector, read_image


def output(data: dict, output_path: Path | None = None) -> None:
    payload = json.dumps(data, indent=2, allow_nan=False) + "\n"
    if output_path is None:
        sys.stdout.write(payload)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8") as destination:
            destination.write(payload)


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--state",
        type=Path,
        default=Path(".edgeauth"),
        help="local state directory (default: .edgeauth)",
    )
    cli.add_argument("--models-dir", type=Path, help="override the state models directory")
    cli.add_argument("--key-file", type=Path, help="override the state vault key file")
    commands = cli.add_subparsers(dest="command", required=True)

    models = commands.add_parser("download-models", help="fetch and verify pinned weights")
    models.add_argument("--variant", choices=(*RECOGNIZERS, "all"), default="all")
    commands.add_parser("init", help="initialise a local verification vault")

    def inference_command(name: str, help_text: str) -> argparse.ArgumentParser:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--variant", choices=RECOGNIZERS, default="sface-fp32")
        command.add_argument("--pad-cutoff", type=float, default=DEFAULT_PAD_CUTOFF)
        return command

    enrol = inference_command("enrol", "enrol a pseudonymous subject from three JPEG/PNG captures")
    enrol.add_argument("subject", type=subject_id)
    enrol.add_argument("captures", type=Path, nargs=3)
    enrol.add_argument("--replace", action="store_true")

    verify = inference_command("verify", "evaluate a still capture against a claimed identity")
    verify.add_argument("subject", type=subject_id)
    verify.add_argument("capture", type=Path)
    decision = verify.add_mutually_exclusive_group(required=True)
    decision.add_argument("--policy", type=Path, help="threshold policy exported from calibration")
    decision.add_argument("--threshold", type=float, help="explicit experimental cosine threshold")

    delete = commands.add_parser("delete", help="delete a subject's local template")
    delete.add_argument("subject", type=subject_id)

    evaluation = inference_command(
        "evaluate", "calibrate on one identity split and test on another"
    )
    evaluation.add_argument("manifest", type=Path)
    evaluation.add_argument("--target-fmr", type=float, default=0.01)
    evaluation.add_argument("--robustness", action="store_true")
    evaluation.add_argument("--output", type=Path)
    evaluation.add_argument("--policy-out", type=Path)

    pad = inference_command("evaluate-pad", "score a labelled PAD manifest")
    pad.add_argument("manifest", type=Path)
    pad.add_argument("--output", type=Path)

    latency = commands.add_parser("benchmark", help="compare FP32 and INT8 inference on the CPU")
    latency.add_argument("capture", type=Path)
    latency.add_argument("--runs", type=int, default=30)
    latency.add_argument("--warmup", type=int, default=5)
    latency.add_argument("--output", type=Path)

    synthetic = commands.add_parser("demo-report", help="emit synthetic dashboard example scores")
    synthetic.add_argument("--output", type=Path)
    return cli


def run(args: argparse.Namespace) -> int:
    models_dir = args.models_dir or args.state / "models"
    key_path = args.key_file or args.state / "vault.key"
    database = args.state / "templates.sqlite3"
    if args.command == "download-models":
        variants = RECOGNIZERS if args.variant == "all" else (args.variant,)
        for model in (YUNET, *(RECOGNIZERS[variant] for variant in variants), PAD):
            print(download_model(models_dir, model))
        return 0
    if args.command == "init":
        if not key_path.exists():
            if database.exists():
                raise ValueError("existing vault requires its original key")
            create_key(key_path)
        TemplateVault(database, key_path, initialise=True)
        output({"status": "ready"})
        return 0
    if args.command == "delete":
        removed = TemplateVault(database, key_path).delete(args.subject)
        output({"deleted": removed})
        return 0 if removed else 1
    if args.command == "demo-report":
        output(synthetic_report(), args.output)
        return 0
    if args.command == "benchmark":
        output(benchmark(models_dir, args.capture, args.runs, args.warmup), args.output)
        return 0

    if args.command == "verify":
        try:
            image = read_image(args.capture)
        except CaptureRejected:
            output({"accepted": False})
            return 1

    screen = Screen(
        FacePipeline(models_dir, args.variant), PresentationDetector(models_dir), args.pad_cutoff
    )
    if args.command == "enrol":
        vault = TemplateVault(database, key_path)
        images = [read_image(path) for path in args.captures]
        fingerprints = {hashlib.sha256(image.tobytes()).digest() for image in images}
        if len(fingerprints) != 3:
            raise ValueError("enrolment requires three distinct captures")
        template = build_template([screen.embed(image) for image in images])
        vault.put(args.subject, screen.matcher.model_id, template, replace=args.replace)
        output({"enrolled": True})
        return 0
    if args.command == "verify":
        vault = TemplateVault(database, key_path)
        threshold = screen.load_threshold(args.policy) if args.policy else args.threshold
        accepted = verification_decision(vault, screen, args.subject, image, threshold)
        output({"accepted": accepted})
        return 0 if accepted else 1
    if args.command == "evaluate":
        if args.output and args.policy_out and args.output.resolve() == args.policy_out.resolve():
            raise ValueError("report and policy must use different files")
        report = evaluate(
            load_pairs(args.manifest), screen, args.target_fmr, robustness=args.robustness
        )
        output(report, args.output)
        if args.policy_out:
            point = report["calibration"]["operating_point"]
            if point is None:
                raise ValueError("calibration found no usable threshold; report is available")
            output(screen.policy(point["threshold"], args.target_fmr), args.policy_out)
        return 0
    if args.command == "evaluate-pad":
        output(evaluate_pad(args.manifest, screen), args.output)
        return 0
    raise RuntimeError(f"unhandled command: {args.command}")


def main() -> int:
    cli = parser()
    try:
        return run(cli.parse_args())
    except (OSError, ValueError, cv2.error, sqlite3.Error, csv.Error) as error:
        print(f"edgeauth: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
