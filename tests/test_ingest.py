"""Ingestion: decode/hash/sidecar/channel-split (design spec §6). Requires ffmpeg."""

from __future__ import annotations

import shutil

import pytest
import soundfile as sf

from voice_evals.cache import ResultCache
from voice_evals.ingest import Ingestor

pytestmark = pytest.mark.local


def test_decodes_to_16k_mono(config, cache, tone_wav, tmp_path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    shutil.copy(tone_wav, corpus / "a.wav")
    clip = Ingestor(config, cache).ingest_dir(corpus)[0]
    assert clip.sample_rate == 16000
    assert clip.n_source_channels == 1
    assert clip.mono16k_path.exists()
    assert sf.info(str(clip.mono16k_path)).samplerate == 16000
    assert 1.8 < clip.duration_s < 2.2


def test_clip_id_stable_and_content_addressed(config, cache, tone_wav, tmp_path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    shutil.copy(tone_wav, corpus / "a.wav")
    shutil.copy(tone_wav, corpus / "b_copy.wav")  # identical content
    ing = Ingestor(config, cache)
    clips = ing.ingest_dir(corpus)
    assert len(clips) == 1  # identical content deduped
    first = clips[0].clip_id
    # re-ingest same file => identical clip_id (decode is deterministic)
    again = ing.ingest_file(corpus / "a.wav")
    assert again.clip_id == first


def test_stereo_split_and_channel_isolation(config, cache, stereo_conversation, tmp_path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    shutil.copy(stereo_conversation, corpus / "conv.wav")
    shutil.copy(stereo_conversation.with_suffix(".json"), corpus / "conv.json")
    clip = Ingestor(config, cache).ingest_dir(corpus)[0]
    assert clip.n_source_channels == 2
    assert clip.stereo_path.exists()
    assert clip.agent_only_path.exists() and clip.caller_only_path.exists()
    assert clip.metadata.channel_map.agent_channel == 0
    # isolated channels differ
    ag, _ = sf.read(str(clip.agent_only_path))
    ca, _ = sf.read(str(clip.caller_only_path))
    # agent speaks 3-5s, caller 0-2s => early samples differ markedly
    assert abs(float(ag[: 16000].std()) - float(ca[: 16000].std())) > 1e-3


def test_sidecar_parsed(config, cache, stereo_conversation, tmp_path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    shutil.copy(stereo_conversation, corpus / "conv.wav")
    shutil.copy(stereo_conversation.with_suffix(".json"), corpus / "conv.json")
    clip = Ingestor(config, cache).ingest_dir(corpus)[0]
    assert clip.metadata.agent_version == "v1"


def test_normalize_dbfs_changes_clip_id(config, cache, tone_wav, tmp_path):
    # changing normalize_dbfs changes the bytes scorers read, so it must change
    # clip_id (and thus invalidate every scorer's cache).
    corpus = tmp_path / "c"
    corpus.mkdir()
    shutil.copy(tone_wav, corpus / "a.wav")
    config.ingest.normalize_dbfs = None
    id_none = Ingestor(config, cache).ingest_file(corpus / "a.wav").clip_id
    config.ingest.normalize_dbfs = -3.0
    id_norm = Ingestor(config, cache).ingest_file(corpus / "a.wav").clip_id
    assert id_none != id_norm


def test_missing_corpus_raises(config, cache, tmp_path):
    with pytest.raises(FileNotFoundError):
        Ingestor(config, cache).ingest_dir(tmp_path / "does_not_exist")
