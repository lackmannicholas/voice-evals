"""Persistent UTMOS worker (run as a subprocess).

UTMOS22 (``tarepan/SpeechMOS``) ships a top-level package literally named
``speechmos`` that collides with the *Microsoft* ``speechmos`` pip package used
by the DNSMOS scorer. Once the Microsoft package is in ``sys.modules`` (which
happens in any normal run), torch.hub's UTMOS import fails. Isolating UTMOS in a
dedicated interpreter that never imports the Microsoft package removes the
collision entirely, and a *persistent* worker amortizes the one-time model load
across a whole corpus.

Protocol (newline-delimited JSON over stdio):
  stdin : {"path": "<wav>"} per line  (or {"cmd": "ping"})
  stdout: {"score": <float>} | {"error": "<msg>"} | {"ready": true} per line
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    try:
        import numpy as np
        import soundfile as sf
        import torch

        model = torch.hub.load("tarepan/SpeechMOS", "utmos22_strong", trust_repo=True)
        model.eval()
        backend = "utmos22_strong"
    except Exception as e:  # noqa: BLE001
        sys.stdout.write(json.dumps({"fatal": f"{type(e).__name__}: {e}"}) + "\n")
        sys.stdout.flush()
        return 1

    sys.stdout.write(json.dumps({"ready": True, "backend": backend}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("cmd") == "quit":
                break
            path = req["path"]
            window_s = float(req.get("window_s", 20.0))
            min_window_s = float(req.get("min_window_s", 3.0))
            how = req.get("aggregate", "median")
            wav, sr = sf.read(path, dtype="float32", always_2d=False)
            if wav.ndim == 2:
                wav = wav.mean(axis=1)
            wav = np.ascontiguousarray(wav, dtype="float32")
            # window long clips (UTMOS is trained on short utterances)
            n = int(window_s * sr)
            chunks = [wav] if (n <= 0 or wav.shape[0] <= n) else [
                wav[i : i + n] for i in range(0, wav.shape[0], n)
                if wav[i : i + n].shape[0] >= int(min_window_s * sr) or i == 0
            ]
            scores = []
            for w in chunks:
                t = torch.from_numpy(w).unsqueeze(0)
                with torch.no_grad():
                    scores.append(float(model(t, sr).squeeze().item()))
            agg = float(np.median(scores) if how == "median" else np.mean(scores))
            out = {"score": agg, "n_windows": len(scores)}
        except Exception as e:  # noqa: BLE001
            out = {"error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
