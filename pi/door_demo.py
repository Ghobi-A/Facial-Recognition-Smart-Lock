#!/usr/bin/env python3
"""Live Raspberry Pi demo: camera -> EdgeAuth 1:1 verification -> GPIO relay.

This is a demonstration, not an access-control product. It fixes the 2022 design flaws:
  * 1:1 verification against a claimed subject (2022 unlocked for *any* detected face);
  * presentation screening and quality gates on every frame;
  * a calibrated threshold policy instead of the library default;
  * N consecutive accepted frames before the relay fires, and a cooldown afterwards;
  * the relay is always released on exit.

Usage (after `edgeauth init`, `edgeauth enrol ...` and calibrating a policy):
    python pi/door_demo.py alice --policy policy.json [--pin 23] [--frames 3]
Without RPi.GPIO installed (e.g. on a laptop) the relay is simulated and printed.
"""

import argparse
import time
from pathlib import Path

import cv2

from edgeauth.vault import TemplateVault
from edgeauth.verification import DEFAULT_PAD_CUTOFF, Screen, verification_decision
from edgeauth.vision import FacePipeline, PresentationDetector


class Relay:
    def __init__(self, pin: int):
        self.pin = pin
        try:
            import RPi.GPIO as GPIO

            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
            self.gpio = GPIO
        except (ImportError, RuntimeError):
            self.gpio = None
            print("RPi.GPIO unavailable: simulating the relay")

    def set(self, on: bool) -> None:
        if self.gpio:
            self.gpio.output(self.pin, self.gpio.HIGH if on else self.gpio.LOW)
        else:
            print("RELAY", "OPEN" if on else "closed")

    def close(self) -> None:
        self.set(False)
        if self.gpio:
            self.gpio.cleanup(self.pin)


def main() -> int:
    cli = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    cli.add_argument("subject")
    cli.add_argument("--policy", type=Path, required=True)
    cli.add_argument("--state", type=Path, default=Path(".edgeauth"))
    cli.add_argument("--variant", default="sface-int8")
    cli.add_argument("--pad-cutoff", type=float, default=DEFAULT_PAD_CUTOFF)
    cli.add_argument("--camera", type=int, default=0)
    cli.add_argument("--pin", type=int, default=23)
    cli.add_argument("--frames", type=int, default=3, help="consecutive accepts required")
    cli.add_argument("--open-seconds", type=float, default=5.0)
    cli.add_argument("--timeout", type=float, default=15.0, help="give up after this long")
    args = cli.parse_args()

    models = args.state / "models"
    screen = Screen(
        FacePipeline(models, args.variant), PresentationDetector(models), args.pad_cutoff
    )
    threshold = screen.load_threshold(args.policy)
    vault = TemplateVault(args.state / "templates.sqlite3", args.state / "vault.key")
    camera = cv2.VideoCapture(args.camera)
    relay = Relay(args.pin)
    try:
        streak, deadline = 0, time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            ok, frame = camera.read()
            if not ok:
                continue
            accepted = verification_decision(vault, screen, args.subject, frame, threshold)
            streak = streak + 1 if accepted else 0
            print(f"frame accepted={accepted} streak={streak}/{args.frames}")
            if streak >= args.frames:
                relay.set(True)
                time.sleep(args.open_seconds)
                relay.set(False)
                return 0
        print("not verified before timeout")
        return 1
    finally:
        relay.close()
        camera.release()


if __name__ == "__main__":
    raise SystemExit(main())
