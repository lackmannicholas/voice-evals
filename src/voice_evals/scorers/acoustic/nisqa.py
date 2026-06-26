"""NISQA — multidimensional speech-quality predictor (design spec §8.2).

Outputs overall MOS plus four dimensions: Noisiness, Coloration, Discontinuity,
Loudness. The **Discontinuity** dimension is the primary glitch/dropout/choppy-audio
detector — directly relevant to "weird pause"/artifact complaints and especially
regression-sensitive.

NISQA has no clean PyPI package, so we use its own supported CLI
(``run_predict.py``) against a vendored copy of ``gabrielmittag/NISQA`` + the
pretrained ``nisqa.tar`` weights. ``scripts/fetch_models.py nisqa`` clones the
repo and downloads weights into ``.cache/models/``. The repo location can also be
set via the ``NISQA_DIR`` env var.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from ...config import Config
from ...models import AudioClip
from ..base import BaseScorer, ScorerError, register
from .._audio import pick_acoustic_path

_COLUMN_MAP = {
    "mos_pred": "nisqa_mos",
    "noi_pred": "nisqa_noisiness",
    "col_pred": "nisqa_coloration",
    "dis_pred": "nisqa_discontinuity",
    "loud_pred": "nisqa_loudness",
}


def _find_repo() -> Optional[Path]:
    candidates = []
    env = os.environ.get("NISQA_DIR")
    if env:
        candidates.append(Path(env))
    candidates += [
        Path.cwd() / ".cache" / "models" / "NISQA",
        Path.cwd() / "third_party" / "NISQA",
    ]
    for c in candidates:
        if c.exists() and (c / "run_predict.py").exists():
            return c
    return None


def _find_weights(repo: Path) -> Optional[Path]:
    for p in [
        repo / "weights" / "nisqa.tar",
        Path.cwd() / ".cache" / "models" / "nisqa" / "nisqa.tar",
        repo / "nisqa.tar",
    ]:
        if p.exists():
            return p
    return None


class NisqaScorer(BaseScorer):
    name = "nisqa"
    layer = "acoustic"
    version = "1"

    def config_dict(self) -> dict[str, Any]:
        repo = _find_repo()
        weights = _find_weights(repo) if repo else None
        return {
            "target_sr": self.config.ingest.target_sr,
            "weights": weights.name if weights else "nisqa.tar",
        }

    def _compute(self, clip: AudioClip) -> tuple[dict[str, float], dict[str, Any]]:
        repo = _find_repo()
        if repo is None:
            raise ScorerError(
                "NISQA repo not found. Run `python scripts/fetch_models.py nisqa` to vendor "
                "gabrielmittag/NISQA + weights into .cache/models/, or set NISQA_DIR."
            )
        weights = _find_weights(repo)
        if weights is None:
            raise ScorerError(
                f"NISQA weights (nisqa.tar) not found near {repo}. "
                "Run `python scripts/fetch_models.py nisqa`."
            )

        path, channel = pick_acoustic_path(clip)
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            cmd = [
                sys.executable,
                str(repo / "run_predict.py"),
                "--mode",
                "predict_file",
                "--pretrained_model",
                str(weights),
                "--deg",
                str(path),
                "--output_dir",
                str(out_dir),
            ]
            cp = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(repo), check=False
            )
            if cp.returncode != 0:
                raise ScorerError(f"NISQA run_predict failed: {cp.stderr.strip()[-500:]}")
            row = self._read_first_csv_row(out_dir)
        metrics: dict[str, float] = {}
        for col, metric in _COLUMN_MAP.items():
            if col in row and row[col] not in (None, ""):
                metrics[metric] = float(row[col])
        if "nisqa_mos" not in metrics:
            raise ScorerError(f"NISQA output missing mos_pred; columns={list(row)}")
        return metrics, {"input_channel": channel}

    @staticmethod
    def _read_first_csv_row(out_dir: Path) -> dict[str, str]:
        csvs = sorted(out_dir.glob("*.csv"))
        if not csvs:
            raise ScorerError("NISQA produced no CSV output")
        with csvs[0].open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                return row
        raise ScorerError("NISQA CSV had no data rows")


@register("nisqa", "acoustic")
def _build(config: Config) -> NisqaScorer:
    return NisqaScorer(config)
