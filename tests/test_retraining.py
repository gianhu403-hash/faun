"""Retraining loop tests — pass WITHOUT TensorFlow.

Covers the provenance negative gate, CV metrics on precomputed embeddings,
save/load round-trip (incl. YAMNetAdapter pickle path), the zero-ground-truth
refusal, and CLI dispatch for ``retrain`` / ``export-clips``.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from faun import retraining
from faun.cli import main

SR = 16_000


def _label(source: str, status: str, species: str = "Turdus merula", **extra) -> dict:
    return {"source": source, "status": status, "species": species, **extra}


def test_provenance_negative_gate_drops_model_labels():
    labels = [
        _label("model:perch", "pseudo"),
        _label("expert:ornithologist", "confirmed", "Erithacus rubecula"),
        _label("operator:ranger", "corrected", "Parus major"),
        _label("model:yamnet-probe", "pseudo"),
    ]

    kept = retraining.filter_ground_truth(labels)

    assert len(kept) == 2
    sources = {item["source"] for item in kept}
    assert sources == {"expert:ornithologist", "operator:ranger"}
    assert all(not item["source"].startswith("model:") for item in kept)


def test_is_ground_truth_accepts_objects_and_rejects_unconfirmed():
    class L:
        def __init__(self, source, status):
            self.source = source
            self.status = status

    assert retraining.is_ground_truth(L("expert:ornithologist", "confirmed"))
    assert retraining.is_ground_truth(L("operator:ranger", "corrected"))
    # right source, wrong status
    assert not retraining.is_ground_truth(L("expert:ornithologist", "pseudo"))
    # model source can never be ground truth
    assert not retraining.is_ground_truth(L("model:perch", "confirmed"))


def test_train_probe_cv_small_n_flags_unreliable_ci_and_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    n, dim = 40, 16
    X = rng.standard_normal((n, dim))
    # Two separable-ish classes.
    y = np.array(["a"] * (n // 2) + ["b"] * (n // 2))
    X[: n // 2] += 1.5

    clf, metrics = retraining.train_probe_cv(X, y, seed=42)

    assert hasattr(clf, "predict_proba")
    assert metrics["n"] == n
    assert metrics["n_classes"] == 2
    assert metrics["metric"] == "roc_auc"
    assert metrics["ci_low"] is None and metrics["ci_high"] is None
    assert "CI unreliable" in metrics["note"]
    assert f"n={n}" in metrics["note"]

    out = tmp_path / "probe.pkl"
    retraining.save_probe(clf, out)
    loaded = retraining.load_probe(out)
    proba = loaded.predict_proba(X[:3])
    assert proba.shape == (3, 2)


def test_train_probe_cv_large_n_reports_ci():
    rng = np.random.default_rng(1)
    n, dim = 400, 16
    X = rng.standard_normal((n, dim))
    y = np.array(["a"] * (n // 2) + ["b"] * (n // 2))
    X[: n // 2] += 1.2

    _clf, metrics = retraining.train_probe_cv(X, y, seed=42)

    assert metrics["n"] == n
    assert metrics["ci_low"] is not None and metrics["ci_high"] is not None
    assert metrics["ci_low"] <= metrics["value"] <= metrics["ci_high"]
    assert metrics["note"] == ""


def test_train_probe_cv_single_class_never_crashes():
    X = np.random.default_rng(2).standard_normal((10, 16))
    y = np.array(["a"] * 10)

    _clf, metrics = retraining.train_probe_cv(X, y)

    assert metrics["n_classes"] == 1
    assert metrics["ci_low"] is None
    assert "CI unreliable" in metrics["note"]


def test_saved_pickle_loads_via_yamnet_adapter(tmp_path):
    """The probe file is loadable by YAMNetAdapter without touching TF."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((30, 16))
    y = np.array(["a"] * 15 + ["b"] * 15)
    clf, _ = retraining.train_probe_cv(X, y)

    out = tmp_path / "probe.pkl"
    retraining.save_probe(clf, out)

    from faun.classification.yamnet import YAMNetAdapter

    adapter = YAMNetAdapter(probe_path=str(out))
    probe = adapter._load_probe()
    assert probe is not None
    assert hasattr(probe, "predict_proba")


def test_retrain_zero_ground_truth_refuses_before_embed(tmp_path):
    class ExplodingModel:
        def embed(self, waveform, sr):  # pragma: no cover - must never run
            raise AssertionError("model.embed must not be called without ground truth")

    labels = [
        _label("model:perch", "pseudo"),
        _label("model:yamnet-probe", "pseudo"),
    ]

    with pytest.raises(ValueError, match="no ground-truth"):
        retraining.retrain_from_labels(
            labels,
            audio_dir=tmp_path,
            model=ExplodingModel(),
            out_path=tmp_path / "p.pkl",
        )


def test_cli_retrain_dispatches_to_retrain_from_labels(tmp_path, monkeypatch):
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text(
        "species,source,status,segment_path\n"
        "Turdus merula,expert:ornithologist,confirmed,clip1.wav\n",
        encoding="utf-8",
    )
    out = tmp_path / "probe.pkl"

    seen = {}

    def fake_retrain(labels, audio_dir, model, out_path):
        seen["labels"] = labels
        seen["audio_dir"] = audio_dir
        seen["model"] = model
        seen["out_path"] = out_path
        return {"n": 1, "value": 1.0}

    monkeypatch.setattr(retraining, "retrain_from_labels", fake_retrain)

    rc = main(
        ["retrain", "--labels", str(labels_csv), "--model", "yamnet", "--out", str(out)]
    )
    assert rc == 0

    assert len(seen["labels"]) == 1
    assert seen["labels"][0]["species"] == "Turdus merula"
    assert seen["labels"][0]["source"] == "expert:ornithologist"
    # audio-dir defaults to the labels CSV parent
    assert seen["audio_dir"] == tmp_path
    assert seen["out_path"] == out
    # The yamnet embedder is constructed and handed over.
    assert seen["model"].__class__.__name__ == "YAMNetAdapter"


def test_cli_retrain_honours_audio_dir(tmp_path, monkeypatch):
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text(
        "species,source,status,segment_path\n"
        "Parus major,operator:ranger,corrected,clip.wav\n",
        encoding="utf-8",
    )
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    seen = {}
    monkeypatch.setattr(
        retraining,
        "retrain_from_labels",
        lambda labels, audio_dir, model, out_path: (
            seen.update(audio_dir=audio_dir) or {}
        ),
    )

    rc = main(
        [
            "retrain",
            "--labels",
            str(labels_csv),
            "--out",
            str(tmp_path / "p.pkl"),
            "--audio-dir",
            str(audio_dir),
        ]
    )
    assert rc == 0
    assert seen["audio_dir"] == audio_dir


def test_cli_export_clips_builds_zip(tmp_path, capsys):
    job = tmp_path / "job"
    seg_dir = job / "segments"
    seg_dir.mkdir(parents=True)

    # Two real clips written via soundfile.
    sig = np.zeros(SR // 2, dtype=np.float32)
    clip1 = "segments/det1.wav"
    clip2 = "segments/det2.wav"
    sf.write(job / clip1, sig, SR)
    sf.write(job / clip2, sig, SR)

    detections = [
        {
            "detection_id": "det1",
            "trap_id": "A1",
            "source_file": "REC_20260610_213000.wav",
            "segment": {"start_s": 12.5, "duration_s": 3.0},
            "segment_path": clip1,
            "labels": [
                {
                    "species": "Turdus merula",
                    "probability": 0.91,
                    "source": "model:perch",
                    "status": "pseudo",
                    "ts": "2026-06-10T21:30:00Z",
                }
            ],
        },
        {
            "detection_id": "det2",
            "trap_id": "A1",
            "source_file": "REC_20260610_213000.wav",
            "segment": {"start_s": 40.0, "duration_s": 2.0},
            "segment_path": clip2,
            "labels": [],
        },
    ]
    jsonl = job / "detections.jsonl"
    jsonl.write_text(
        "\n".join(json.dumps(d) for d in detections) + "\n", encoding="utf-8"
    )

    out_zip = tmp_path / "clips.zip"
    rc = main(["export-clips", "--job", str(job), "--out", str(out_zip)])
    assert rc == 0
    assert Path(capsys.readouterr().out.strip()) == out_zip

    with zipfile.ZipFile(out_zip) as zf:
        names = set(zf.namelist())
        assert "clips_index.csv" in names
        assert clip1 in names
        assert clip2 in names

        index = zf.read("clips_index.csv").decode("utf-8")

    header = index.splitlines()[0]
    assert header.split(",") == [
        "detection_id",
        "trap_id",
        "source_file",
        "start_sec",
        "duration_sec",
        "suggested_species",
        "suggested_source",
        "suggested_probability",
    ]
    assert "Turdus merula" in index
    assert "model:perch" in index
    assert "det1" in index and "det2" in index


@pytest.mark.requires_tf
def test_real_embed_probe_adapter_roundtrip_cluster(tmp_path):
    """7b (cluster-only): real YAMNet embed -> probe -> save -> reload via adapter.

    SKIPS (never passes) without TensorFlow/tensorflow_hub — it exercises the
    full deploy path only on the cluster (faun-ml-cpu/torch): human ground-truth
    labels -> real YAMNet embeddings -> trained probe -> save -> reload through
    YAMNetAdapter(probe_path=...) (the YAMNET_PROBE_PATH path) -> classify.
    """
    pytest.importorskip("tensorflow")
    pytest.importorskip("tensorflow_hub")
    from faun.classification import YAMNetAdapter

    audio_dir = tmp_path
    labels = []
    rng = np.random.default_rng(0)
    for species, freq in (("species_a", 1000.0), ("species_b", 3000.0)):
        for i in range(2):
            t = np.arange(SR) / SR
            wav = (
                0.3 * np.sin(2 * np.pi * freq * t) + 0.01 * rng.standard_normal(SR)
            ).astype(np.float32)
            name = f"{species}_{i}.wav"
            sf.write(audio_dir / name, wav, SR)
            labels.append(
                _label(
                    "expert:ornithologist",
                    "confirmed",
                    species=species,
                    source_file=name,
                )
            )

    out = tmp_path / "probe.pkl"
    metrics = retraining.retrain_from_labels(labels, audio_dir, YAMNetAdapter(), out)
    assert out.exists()
    assert metrics["n"] == 4 and metrics["n_classes"] == 2

    # Reload through the adapter and classify; labels = the probe's own classes.
    probe = retraining.load_probe(out)
    adapter = YAMNetAdapter(probe_path=str(out), labels=list(probe.classes_))
    t = np.arange(SR) / SR
    probe_wav = (0.3 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    preds = adapter.classify(probe_wav, SR)
    assert preds and preds[0].species in {"species_a", "species_b"}


# ---------------------------------------------------------------------------
# FR-006: temperature calibration (ADR-0005)
# ---------------------------------------------------------------------------


def _synthetic_logits(seed: int = 0, n: int = 400, c: int = 4, margin: float = 2.5):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, c, n)
    logits = rng.standard_normal((n, c))
    logits[np.arange(n), y] += margin  # true class scores higher
    return logits, y


def test_apply_calibration_identity_passthrough() -> None:
    logits, _ = _synthetic_logits()
    out = retraining.apply_calibration(None, logits)
    assert np.allclose(out, logits)  # no calibrator -> raw scores unchanged


def test_temperature_calibrator_outputs_probabilities() -> None:
    logits, y = _synthetic_logits()
    cal = retraining.fit_temperature(logits, y)
    probs = retraining.apply_calibration(cal, logits)
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert cal.temperature > 0.0


def test_temperature_reduces_ece_on_miscalibrated_logits() -> None:
    """Over-confident logits (x4) are mis-calibrated; fitting T lowers ECE."""
    logits, y = _synthetic_logits(seed=1, margin=1.5)
    miscal = logits * 4.0  # inflate -> over-confident softmax
    ece_raw = retraining.expected_calibration_error(retraining._softmax(miscal), y)
    cal = retraining.fit_temperature(miscal, y)
    ece_cal = retraining.expected_calibration_error(
        retraining.apply_calibration(cal, miscal), y
    )
    assert ece_cal <= ece_raw  # calibration never worsens ECE here
    assert cal.temperature > 1.0  # cooling an over-confident model


def test_fit_temperature_single_class_falls_back_to_one() -> None:
    logits = np.array([[2.0], [3.0], [1.5]])
    cal = retraining.fit_temperature(logits, [0, 0, 0])
    assert cal.temperature == 1.0  # no calibration possible with one class


def test_fit_temperature_with_string_classes() -> None:
    logits, yi = _synthetic_logits(seed=2, c=3)
    classes = ["Turdus merula", "Parus major", "Sitta europaea"]
    y = [classes[i] for i in yi]
    cal = retraining.fit_temperature(logits, y, classes=classes)
    assert cal.classes == classes
    probs = retraining.apply_calibration(cal, logits)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_fit_temperature_rejects_non_2d_logits() -> None:
    with pytest.raises(ValueError, match="2-D"):
        retraining.fit_temperature(np.array([1.0, 2.0, 3.0]), [0, 1, 2])
