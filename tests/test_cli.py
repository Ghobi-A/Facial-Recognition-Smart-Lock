import json

import numpy as np
from PIL import Image
from test_verification import PAD, Matcher

from edgeauth.cli import parser, run
from edgeauth.verification import Screen


def test_cli_runs_local_vault_workflow(tmp_path, monkeypatch, capsys):
    images = []
    for index in (1, 2, 3):
        path = tmp_path / f"capture{index}.png"
        Image.new("RGB", (64, 64), (index, 20, 50)).save(path)
        images.append(str(path))

    class TestMatcher(Matcher):
        def analyse(self, image):
            return super().analyse(np.array([1.0, float(image[0, 0, 2]) / 100]))

    monkeypatch.setattr("edgeauth.cli.FacePipeline", lambda *_: TestMatcher())
    monkeypatch.setattr("edgeauth.cli.PresentationDetector", lambda *_: PAD(0.99))
    policy = tmp_path / "threshold.json"
    policy.write_text(
        json.dumps(Screen(TestMatcher(), PAD(0.99)).policy(0.7, 0.01)), encoding="utf-8"
    )

    def command(*args):
        return run(parser().parse_args(["--state", str(tmp_path / "state"), *args]))

    assert command("init") == 0
    assert command("enrol", "sample", *images) == 0
    capsys.readouterr()
    assert command("verify", "sample", images[0], "--policy", str(policy)) == 0
    assert json.loads(capsys.readouterr().out) == {
        "accepted": True,
    }
    assert command("delete", "sample") == 0
    assert command("verify", "sample", images[0], "--threshold", ".7") == 1
