"""Тесты батч-разметки — проходят БЕЗ TensorFlow.

Покрывают: сканирование архива -> сегменты -> псевдо-метки от каждой модели
(source=model:perch / model:birdnet, status=pseudo), консенсус Perch∩BirdNET,
экспорт эмбеддингов, режимы отказа и — главный merge-blocker — лицензионный
гейт BirdNET (CC BY-NC-SA): метки model:birdnet НИКОГДА не попадают в
training_candidates коммерческой модели.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from faun.classification import Prediction
from faun.detections import (
    SOURCE_BIRDNET,
    SOURCE_PERCH,
    STATUS_PSEUDO,
    Detection,
    Label,
    read_detections,
)
from faun.labeling import batch_label, training_candidates
from faun.segmentation import Segment


# ---------------------------------------------------------------------------
# Фикстуры: крошечный архив ловушек + управляемые модели
# ---------------------------------------------------------------------------


_INFO_HEADER = (
    "date,time,long,lat,battery,temp,humidity,filename,sample_rate,gain,channel\n"
)


def _write_trap(root, trap_id, fname, *, sr=16000, seconds=2.0, with_event=True):
    """Создать папку ловушки с одним WAV и info.txt.

    ``with_event``: вставить громкий импульс, чтобы onset-детектор нашёл
    хотя бы один сегмент.
    """
    folder = root / trap_id
    folder.mkdir(parents=True, exist_ok=True)
    n = int(sr * seconds)
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(n) * 0.01).astype(np.float32)
    if with_event:
        # Резкий всплеск энергии в середине -> onset.
        mid = n // 2
        audio[mid : mid + sr // 2] += 0.8
    sf.write(str(folder / fname), audio, sr)
    (folder / "info.txt").write_text(
        _INFO_HEADER
        + f"2026-01-15,08:30:00,44.1,57.4,85,12.5,60,{fname},{sr},auto,mono\n",
        encoding="utf-8",
    )
    return folder


class FixedModel:
    """Классификатор, возвращающий заданный список предсказаний на любой сегмент."""

    def __init__(self, predictions):
        self._predictions = list(predictions)

    def classify(self, segment, sr):
        return list(self._predictions)


_TRAPS_MINI = Path(__file__).parent / "fixtures" / "traps_mini"


# ---------------------------------------------------------------------------
# Готовая on-disk фикстура traps_mini (две ловушки A1/A2)
# ---------------------------------------------------------------------------


class TestTrapsMiniFixture:
    def test_batch_label_over_committed_fixture(self, tmp_path):
        assert _TRAPS_MINI.is_dir(), "фикстура traps_mini отсутствует"
        out = tmp_path / "detections.jsonl"
        models = {
            "perch": FixedModel([Prediction("Turdus merula", 0.9)]),
            "birdnet": FixedModel([Prediction("Turdus merula", 0.8)]),
        }
        summary = batch_label(_TRAPS_MINI, models, out)
        dets = read_detections(out)
        # Обе ловушки дают события -> >=2 детекции, обе trap_id присутствуют.
        assert summary["n_detections"] >= 2
        assert {d.trap_id for d in dets} == {"A1", "A2"}
        assert summary["n_consensus"] == summary["n_detections"]


# ---------------------------------------------------------------------------
# batch_label — happy path + консенсус
# ---------------------------------------------------------------------------


class TestBatchLabel:
    def test_writes_detections_with_per_model_pseudo_labels(self, tmp_path):
        archive = tmp_path / "archive"
        _write_trap(archive, "A1", "20260115_083000.wav")
        out = tmp_path / "detections.jsonl"

        models = {
            "perch": FixedModel([Prediction("Turdus merula", 0.9)]),
            "birdnet": FixedModel([Prediction("Turdus merula", 0.8)]),
        }
        summary = batch_label(archive, models, out)

        assert out.exists()
        dets = read_detections(out)
        assert len(dets) >= 1
        det = dets[0]
        sources = {lbl.source for lbl in det.labels}
        assert SOURCE_PERCH in sources
        assert SOURCE_BIRDNET in sources
        assert all(lbl.status == STATUS_PSEUDO for lbl in det.labels)

        assert summary["n_detections"] == len(dets)
        assert summary["counts"]["perch"] >= 1
        assert summary["counts"]["birdnet"] >= 1
        assert str(out) == summary["paths"]["detections"]

    def test_consensus_counts_species_predicted_by_both(self, tmp_path):
        archive = tmp_path / "archive"
        _write_trap(archive, "A1", "20260115_083000.wav")
        out = tmp_path / "detections.jsonl"

        # Оба согласны на Turdus merula -> консенсус.
        models = {
            "perch": FixedModel([Prediction("Turdus merula", 0.9)]),
            "birdnet": FixedModel([Prediction("Turdus merula", 0.7)]),
        }
        summary = batch_label(archive, models, out)
        assert summary["n_consensus"] >= 1

    def test_consensus_empty_when_models_disagree(self, tmp_path):
        archive = tmp_path / "archive"
        _write_trap(archive, "A1", "20260115_083000.wav")
        out = tmp_path / "detections.jsonl"

        models = {
            "perch": FixedModel([Prediction("Turdus merula", 0.9)]),
            "birdnet": FixedModel([Prediction("Parus major", 0.7)]),
        }
        summary = batch_label(archive, models, out)
        assert summary["n_consensus"] == 0

    def test_emb_out_writes_npz_one_row_per_detection(self, tmp_path):
        archive = tmp_path / "archive"
        _write_trap(archive, "A1", "20260115_083000.wav")
        out = tmp_path / "detections.jsonl"
        emb_out = tmp_path / "embeddings.npz"

        class FakeEmbedder:
            DIM = 8

            def embed(self, waveform, sr):
                return np.full(8, float(np.mean(waveform)), dtype=np.float32)

        models = {"perch": FixedModel([Prediction("Turdus merula", 0.9)])}
        summary = batch_label(
            archive, models, out, emb_out=emb_out, embedder=FakeEmbedder()
        )

        assert emb_out.exists()
        from faun.embeddings import EmbeddingCache

        cache = EmbeddingCache.load(emb_out)
        n_dets = summary["n_detections"]
        assert cache.embeddings.shape == (n_dets, 8)
        # ids привязаны к detection_id.
        assert cache.ids is not None and len(cache.ids) == n_dets
        assert summary["paths"]["embeddings"] == str(emb_out)


# ---------------------------------------------------------------------------
# Режимы отказа
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_empty_archive_writes_empty_detections_no_crash(self, tmp_path):
        archive = tmp_path / "empty"
        archive.mkdir()
        out = tmp_path / "detections.jsonl"
        models = {"perch": FixedModel([Prediction("Turdus merula", 0.9)])}
        summary = batch_label(archive, models, out)
        assert out.exists()
        assert read_detections(out) == []
        assert summary["n_detections"] == 0
        assert summary["n_consensus"] == 0

    def test_model_returning_no_predictions(self, tmp_path):
        archive = tmp_path / "archive"
        _write_trap(archive, "A1", "20260115_083000.wav")
        out = tmp_path / "detections.jsonl"
        models = {
            "perch": FixedModel([]),  # ничего не предсказывает
            "birdnet": FixedModel([Prediction("Turdus merula", 0.8)]),
        }
        summary = batch_label(archive, models, out)
        # Детекции есть (сегменты найдены), но perch не дал меток.
        assert summary["counts"]["perch"] == 0
        # Консенсус невозможен, если одна из моделей молчит.
        assert summary["n_consensus"] == 0

    def test_emb_out_without_embedder_skips_npz(self, tmp_path):
        archive = tmp_path / "archive"
        _write_trap(archive, "A1", "20260115_083000.wav")
        out = tmp_path / "detections.jsonl"
        emb_out = tmp_path / "embeddings.npz"
        models = {"perch": FixedModel([Prediction("Turdus merula", 0.9)])}
        # emb_out задан, но embedder=None -> npz не пишем, без падения.
        summary = batch_label(archive, models, out, emb_out=emb_out, embedder=None)
        assert not emb_out.exists()
        assert "embeddings" not in summary["paths"]


# ---------------------------------------------------------------------------
# ЛИЦЕНЗИОННЫЙ ГЕЙТ BirdNET (merge-blocker)
# ---------------------------------------------------------------------------


class TestBirdnetLicenseGate:
    def _det_with(self, *labels) -> Detection:
        return Detection.new(
            trap_id="A1",
            source_file="rec.wav",
            segment=Segment(start_s=0.0, duration_s=1.0),
            labels=list(labels),
        )

    def test_training_candidates_excludes_birdnet_labels(self):
        det = self._det_with(
            Label.now("Turdus merula", 0.9, SOURCE_PERCH, STATUS_PSEUDO),
            Label.now("Turdus merula", 0.8, SOURCE_BIRDNET, STATUS_PSEUDO),
        )
        candidates = training_candidates([det])

        # В кандидатах не должно быть НИ ОДНОЙ метки model:birdnet.
        for cand in candidates:
            labels = cand.labels if isinstance(cand, Detection) else cand
            for lbl in labels:
                src = lbl.source if isinstance(lbl, Label) else lbl["source"]
                assert src != SOURCE_BIRDNET, "BirdNET (NC+SA) попал в обучение!"

    def test_birdnet_only_detection_yields_no_candidate_labels(self):
        det = self._det_with(
            Label.now("Parus major", 0.7, SOURCE_BIRDNET, STATUS_PSEUDO),
        )
        candidates = training_candidates([det])
        # Все метки этой детекции — birdnet -> ни одной обучающей метки.
        for cand in candidates:
            labels = cand.labels if isinstance(cand, Detection) else cand
            for lbl in labels:
                src = lbl.source if isinstance(lbl, Label) else lbl["source"]
                assert src != SOURCE_BIRDNET

    def test_perch_labels_survive_the_gate(self):
        det = self._det_with(
            Label.now("Turdus merula", 0.9, SOURCE_PERCH, STATUS_PSEUDO),
            Label.now("Turdus merula", 0.8, SOURCE_BIRDNET, STATUS_PSEUDO),
        )
        candidates = training_candidates([det])
        all_sources = []
        for cand in candidates:
            labels = cand.labels if isinstance(cand, Detection) else cand
            for lbl in labels:
                all_sources.append(
                    lbl.source if isinstance(lbl, Label) else lbl["source"]
                )
        assert SOURCE_PERCH in all_sources
