"""Ingestion: the only component that touches raw files (design spec §6).

For each source file it:
  1. locates an optional ``<stem>.json`` sidecar and parses ClipMetadata,
  2. decodes to 16 kHz mono 16-bit PCM wav (canonical MOS input),
  3. preserves channels in a stereo wav when the source has >=2 channels, and
     isolates agent/caller channels when a channel_map is known,
  4. computes ``clip_id`` from the decoded mono PCM bytes (NOT the source file),
  5. caches all decoded artifacts under ``.cache/audio/<clip_id>/`` keyed by a
     source-content hash so decode happens once.

Everything downstream operates on the resulting ``AudioClip``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from .cache import ResultCache
from .config import Config
from .logging_util import get_logger
from .models import AudioClip, ChannelMap, ClipMetadata

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# ffmpeg / ffprobe discovery
# --------------------------------------------------------------------------- #
def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
        raise RuntimeError(
            "ffmpeg not found. Install system ffmpeg or `pip install imageio-ffmpeg`."
        )


def _ffprobe_exe() -> Optional[str]:
    return shutil.which("ffprobe")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


# --------------------------------------------------------------------------- #
# probing
# --------------------------------------------------------------------------- #
def _probe_channels(src: Path) -> int:
    """Return the number of audio channels in the source (best effort)."""
    probe = _ffprobe_exe()
    if probe:
        cp = _run(
            [
                probe,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=channels",
                "-of",
                "json",
                str(src),
            ]
        )
        if cp.returncode == 0:
            try:
                streams = json.loads(cp.stdout).get("streams", [])
                if streams and streams[0].get("channels"):
                    return int(streams[0]["channels"])
            except Exception:
                pass
    # fallback: decode a short native wav and read its header
    try:
        info = sf.info(str(src))
        return int(info.channels)
    except Exception:
        # last resort: decode native via ffmpeg then read
        tmp = src.parent / f".probe_{src.stem}.wav"
        try:
            _decode(src, tmp, channels=None, rate=None)
            n = int(sf.info(str(tmp)).channels)
            return n
        finally:
            tmp.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #
def _decode(
    src: Path,
    dst: Path,
    channels: Optional[int],
    rate: Optional[int],
    pan_channel: Optional[int] = None,
) -> None:
    """Decode ``src`` to a pcm_s16le wav at ``dst``.

    channels/rate None => preserve source. ``pan_channel`` extracts a single
    source channel to mono (used for agent/caller isolation).
    """
    ff = _ffmpeg_exe()
    cmd = [ff, "-nostdin", "-y", "-i", str(src)]
    if pan_channel is not None:
        cmd += ["-af", f"pan=mono|c0=c{pan_channel}"]
    elif channels is not None:
        cmd += ["-ac", str(channels)]
    if rate is not None:
        cmd += ["-ar", str(rate)]
    cmd += ["-c:a", "pcm_s16le", "-f", "wav", str(dst)]
    cp = _run(cmd)
    if cp.returncode != 0 or not dst.exists():
        raise RuntimeError(f"ffmpeg decode failed for {src}: {cp.stderr.strip()[-500:]}")


def _pcm_clip_id(mono16k_path: Path, salt: Optional[str] = None) -> str:
    """blake2b of the decoded mono 16-bit PCM samples (re-encode-stable identity).

    ``salt`` (e.g. a sidecar call_id) makes otherwise-identical audio resolve to a
    distinct clip — used for simulated corpora so repeated/identical runs aren't
    collapsed by content dedup."""
    data, _ = sf.read(str(mono16k_path), dtype="int16", always_2d=False)
    h = hashlib.blake2b(np.ascontiguousarray(data).tobytes(), digest_size=20)
    if salt:
        h.update(b"\x00")
        h.update(salt.encode())
    return h.hexdigest()


def _source_hash(src: Path) -> str:
    h = hashlib.sha256()
    with src.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _apply_normalization(path: Path, target_dbfs: float) -> None:
    """Peak-normalize a wav to ``target_dbfs`` in place (no-op on silence)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak <= 1e-9:
        return
    current_dbfs = 20.0 * np.log10(peak)
    gain = 10.0 ** ((target_dbfs - current_dbfs) / 20.0)
    out = np.clip(data * gain, -1.0, 1.0)
    sf.write(str(path), out, sr, subtype="PCM_16")


# --------------------------------------------------------------------------- #
# sidecar
# --------------------------------------------------------------------------- #
def _load_sidecar(src: Path) -> ClipMetadata:
    sidecar = src.with_suffix(".json")
    if not sidecar.exists():
        return ClipMetadata()
    try:
        data = json.loads(sidecar.read_text())
        return ClipMetadata.from_dict(data)
    except Exception as e:
        log.warning("failed to parse sidecar %s: %s; ignoring", sidecar.name, e)
        return ClipMetadata()


# --------------------------------------------------------------------------- #
# Ingestor
# --------------------------------------------------------------------------- #
class Ingestor:
    def __init__(self, config: Config, cache: ResultCache):
        self.config = config
        self.cache = cache

    def discover(self, corpus_dir: Path) -> list[Path]:
        exts = {e.lower() for e in self.config.ingest.accept_extensions}
        files = [
            p
            for p in sorted(corpus_dir.rglob("*"))
            if p.is_file() and p.suffix.lower() in exts
        ]
        return files

    def ingest_dir(self, corpus_dir: Path) -> list[AudioClip]:
        corpus_dir = Path(corpus_dir)
        if not corpus_dir.exists():
            raise FileNotFoundError(f"corpus dir not found: {corpus_dir}")
        clips: list[AudioClip] = []
        seen: set[str] = set()
        for f in self.discover(corpus_dir):
            try:
                clip = self.ingest_file(f)
            except Exception as e:  # noqa: BLE001 - one bad file shouldn't abort
                log.error("failed to ingest %s: %s", f, e)
                continue
            if clip.clip_id in seen:
                log.info("duplicate content (clip_id=%s) from %s; skipping", clip.clip_id, f)
                continue
            seen.add(clip.clip_id)
            clips.append(clip)
        return clips

    # ------------------------------------------------------------------ #
    def ingest_file(self, src: Path) -> AudioClip:
        src = Path(src)
        meta = _load_sidecar(src)

        # decode cache keyed by source content hash + the ingest settings that
        # change the decoded bytes (target_sr, normalize_dbfs), so changing
        # normalization re-decodes instead of serving stale artifacts.
        src_hash = _source_hash(src)
        norm = self.config.ingest.normalize_dbfs
        decode_tag = f"sr{self.config.ingest.target_sr}_norm{'none' if norm is None else norm}"
        by_source = self.cache.audio_dir / "_by_source"
        by_source.mkdir(parents=True, exist_ok=True)
        mapping = by_source / f"{src_hash}.{decode_tag}.json"
        if mapping.exists():
            try:
                cached = json.loads(mapping.read_text())
                clip = self._clip_from_cache(cached, src, meta)
                if clip is not None:
                    return clip
            except Exception:
                pass  # fall through to fresh decode

        n_channels = _probe_channels(src)

        # decode the canonical mono16k into a temp dir. Apply normalization BEFORE
        # hashing so clip_id identifies the exact bytes every scorer reads — any
        # change to normalize_dbfs then changes clip_id and invalidates all caches
        # (acoustic, dynamics-VAD, and judge), not just a subset.
        tmp_dir = self.cache.audio_dir / "_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_mono = tmp_dir / f"{src_hash}.mono16k.wav"
        _decode(src, tmp_mono, channels=1, rate=self.config.ingest.target_sr)
        if norm is not None:
            _apply_normalization(tmp_mono, norm)
        salt = meta.call_id if (self.config.ingest.identity_from_call_id and meta.call_id) else None
        clip_id = _pcm_clip_id(tmp_mono, salt=salt)

        out_dir = self.cache.audio_dir_for(clip_id)
        mono16k_path = out_dir / "mono16k.wav"
        shutil.move(str(tmp_mono), str(mono16k_path))

        stereo_path: Optional[Path] = None
        agent_path: Optional[Path] = None
        caller_path: Optional[Path] = None

        if n_channels >= 2:
            stereo_path = out_dir / "stereo.wav"
            _decode(src, stereo_path, channels=None, rate=None)
            cmap: Optional[ChannelMap] = meta.channel_map
            if cmap is not None:
                if cmap.agent_channel is not None and cmap.agent_channel < n_channels:
                    agent_path = out_dir / "agent_only16k.wav"
                    _decode(
                        src,
                        agent_path,
                        channels=None,
                        rate=self.config.ingest.target_sr,
                        pan_channel=cmap.agent_channel,
                    )
                if cmap.caller_channel is not None and cmap.caller_channel < n_channels:
                    caller_path = out_dir / "caller_only16k.wav"
                    _decode(
                        src,
                        caller_path,
                        channels=None,
                        rate=self.config.ingest.target_sr,
                        pan_channel=cmap.caller_channel,
                    )

        # mono16k was already normalized before hashing; normalize the isolated
        # channels too (they are derived independently and don't feed clip_id).
        if norm is not None:
            for p in [agent_path, caller_path]:
                if p is not None:
                    _apply_normalization(p, norm)

        info = sf.info(str(mono16k_path))
        duration_s = info.frames / float(info.samplerate)
        self._validate_duration(src, duration_s)

        clip = AudioClip(
            clip_id=clip_id,
            source_path=src,
            mono16k_path=mono16k_path,
            stereo_path=stereo_path,
            agent_only_path=agent_path,
            caller_only_path=caller_path,
            sample_rate=int(info.samplerate),
            duration_s=duration_s,
            n_source_channels=n_channels,
            metadata=meta,
        )

        # persist the decode mapping (paths only; metadata re-parsed from sidecar)
        mapping.write_text(
            json.dumps(
                {
                    "clip_id": clip_id,
                    "mono16k_path": str(mono16k_path),
                    "stereo_path": str(stereo_path) if stereo_path else None,
                    "agent_only_path": str(agent_path) if agent_path else None,
                    "caller_only_path": str(caller_path) if caller_path else None,
                    "sample_rate": int(info.samplerate),
                    "duration_s": duration_s,
                    "n_source_channels": n_channels,
                },
                indent=2,
            )
        )
        return clip

    # ------------------------------------------------------------------ #
    def _clip_from_cache(
        self, cached: dict, src: Path, meta: ClipMetadata
    ) -> Optional[AudioClip]:
        mono = Path(cached["mono16k_path"])
        if not mono.exists():
            return None

        def _opt(key: str) -> Optional[Path]:
            v = cached.get(key)
            if not v:
                return None
            p = Path(v)
            return p if p.exists() else None

        return AudioClip(
            clip_id=cached["clip_id"],
            source_path=src,
            mono16k_path=mono,
            stereo_path=_opt("stereo_path"),
            agent_only_path=_opt("agent_only_path"),
            caller_only_path=_opt("caller_only_path"),
            sample_rate=int(cached["sample_rate"]),
            duration_s=float(cached["duration_s"]),
            n_source_channels=int(cached["n_source_channels"]),
            metadata=meta,
        )

    def _validate_duration(self, src: Path, duration_s: float) -> None:
        if duration_s < self.config.ingest.min_duration_s:
            log.warning(
                "%s is very short (%.2fs); some scorers may be unreliable", src.name, duration_s
            )
        if duration_s > self.config.ingest.max_duration_s:
            log.warning(
                "%s is very long (%.0fs); some scorers may be slow or fail",
                src.name,
                duration_s,
            )
