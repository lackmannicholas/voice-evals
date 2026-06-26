#!/usr/bin/env python3
"""Fetch / vendor model weights into ``.cache/models/`` (design spec §19).

Most backends self-fetch on first use:
  * DNSMOS  — bundled in the ``speechmos`` pip package (no fetch needed).
  * UTMOS   — torch.hub downloads ``tarepan/SpeechMOS`` weights on first run.
  * SQUIM   — torchaudio downloads the SQUIM_OBJECTIVE bundle on first run.
  * silero-vad — downloads on first use.

NISQA has no clean PyPI package, so it must be vendored explicitly:

    python scripts/fetch_models.py nisqa

clones ``gabrielmittag/NISQA`` into ``.cache/models/NISQA`` and downloads the
pretrained ``nisqa.tar`` weights into ``.cache/models/nisqa/``. The NISQA scorer
then invokes the repo's own ``run_predict.py`` against those weights.
"""

from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path

NISQA_REPO = "https://github.com/gabrielmittag/NISQA.git"
# Weights live in the repo under weights/nisqa.tar after clone; we also expose a
# stable copy under .cache/models/nisqa/nisqa.tar.
CACHE = Path.cwd() / ".cache" / "models"


def fetch_nisqa() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    repo = CACHE / "NISQA"
    if repo.exists():
        print(f"[nisqa] repo already present at {repo}")
    else:
        print(f"[nisqa] cloning {NISQA_REPO} -> {repo}")
        cp = subprocess.run(["git", "clone", "--depth", "1", NISQA_REPO, str(repo)])
        if cp.returncode != 0:
            print("[nisqa] git clone failed", file=sys.stderr)
            return 1
    # The repo ships weights under weights/nisqa.tar. Mirror to .cache/models/nisqa.
    repo_weights = repo / "weights" / "nisqa.tar"
    dest_dir = CACHE / "nisqa"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "nisqa.tar"
    if repo_weights.exists() and not dest.exists():
        dest.write_bytes(repo_weights.read_bytes())
        print(f"[nisqa] copied weights -> {dest}")
    if not (repo_weights.exists() or dest.exists()):
        print(
            "[nisqa] WARNING: nisqa.tar not found in the cloned repo. Check the "
            "NISQA README for the current weights download location and place it "
            f"at {dest}.",
            file=sys.stderr,
        )
        return 2
    print("[nisqa] ready. The NISQA scorer will use this on the next run.")
    print(f"        (export NISQA_DIR={repo} to override the search path.)")
    return 0


def main(argv: list[str]) -> int:
    targets = argv or ["nisqa"]
    rc = 0
    for t in targets:
        if t == "nisqa":
            rc |= fetch_nisqa()
        elif t in {"dnsmos", "utmos", "squim", "silero"}:
            print(f"[{t}] self-fetches on first use; nothing to do.")
        else:
            print(f"unknown target {t!r}; known: nisqa, dnsmos, utmos, squim, silero", file=sys.stderr)
            rc |= 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
