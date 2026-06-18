"""Тесты адаптеров эмбеддингов — проходят БЕЗ TensorFlow.

Покрывают: батч-эмбеддинг через фейковый эмбеддер (реальный прогон + npz
round-trip на диске), препроцессинг PerchEmbedder/YamnetEmbedder (resample +
pad/truncate до нужной формы) через монкипатч тяжёлой TF-функции на stub,
который ASSERT-ит полученную форму/частоту, и режимы отказа.

Любое число, полученное из синтетических эмбеддингов, помечается как
``SYNTHETIC — not a species metric`` — это не видовая метрика.
"""

from __future__ import annotations

import numpy as np
import pytest

from faun.embeddings import (
    EmbeddingCache,
    Perch2Embedder,
    PerchEmbedder,
    YamnetEmbedder,
    embed_batch,
)


# ---------------------------------------------------------------------------
# Фейковый эмбеддер — детерминированный, без ML
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Детерминированный эмбеддер фиксированной размерности (без ML).

    Возвращает вектор [dim], завязанный на длину/среднее сигнала, чтобы
    батч-прогон давал воспроизводимые, но различимые строки.
    """

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls: list[tuple[int, int]] = []

    def embed(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        self.calls.append((len(np.asarray(waveform)), sr))
        base = float(np.asarray(waveform, dtype=np.float64).mean())
        return np.full(self.dim, base, dtype=np.float32) + np.arange(
            self.dim, dtype=np.float32
        )


# ---------------------------------------------------------------------------
# embed_batch + EmbeddingCache (реальный прогон, TF-free)
# ---------------------------------------------------------------------------


class TestEmbedBatch:
    def test_batch_stacks_per_clip_embeddings(self):
        emb = FakeEmbedder(dim=8)
        clips = [
            (np.full(100, 0.1, dtype=np.float32), 16000),
            (np.full(200, 0.5, dtype=np.float32), 16000),
            (np.full(50, -0.2, dtype=np.float32), 16000),
        ]
        out = embed_batch(clips, emb)
        assert out.shape == (3, 8)
        assert out.dtype == np.float32
        # Эмбеддер реально вызван на каждый клип.
        assert len(emb.calls) == 3
        assert emb.calls[1] == (200, 16000)
        # Строки различимы (зависят от среднего сигнала).
        assert not np.allclose(out[0], out[1])

    def test_empty_batch_returns_empty_2d(self):
        emb = FakeEmbedder(dim=8)
        out = embed_batch([], emb)
        assert out.shape == (0, 8)
        assert out.ndim == 2

    def test_empty_batch_unknown_dim_returns_zero_zero(self):
        # Эмбеддер без атрибута dim и пустой вход -> (0, 0), без падения.
        class Anon:
            def embed(self, waveform, sr):  # pragma: no cover - не вызывается
                return np.zeros(4)

        out = embed_batch([], Anon())
        assert out.shape == (0, 0)


class TestEmbeddingCache:
    def test_npz_roundtrip_on_disk(self, tmp_path):
        emb = FakeEmbedder(dim=8)
        clips = [
            (np.full(100, 0.1, dtype=np.float32), 16000),
            (np.full(200, 0.5, dtype=np.float32), 16000),
        ]
        arr = embed_batch(clips, emb)
        cache = EmbeddingCache(embeddings=arr, ids=["d0", "d1"], labels=["a", "b"])

        path = tmp_path / "emb.npz"
        cache.save(path)
        assert path.exists()

        loaded = EmbeddingCache.load(path)
        # SYNTHETIC — not a species metric: сравниваем сами векторы, не качество.
        assert np.array_equal(loaded.embeddings, arr)
        assert loaded.ids == ["d0", "d1"]
        assert loaded.labels == ["a", "b"]

    def test_roundtrip_without_ids_or_labels(self, tmp_path):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        cache = EmbeddingCache(embeddings=arr)
        path = tmp_path / "emb.npz"
        cache.save(path)
        loaded = EmbeddingCache.load(path)
        assert np.array_equal(loaded.embeddings, arr)
        assert loaded.ids is None
        assert loaded.labels is None

    def test_len_matches_rows(self):
        arr = np.zeros((5, 8), dtype=np.float32)
        assert len(EmbeddingCache(embeddings=arr)) == 5


# ---------------------------------------------------------------------------
# PerchEmbedder — препроцессинг через монкипатч (TF-free)
# ---------------------------------------------------------------------------


class TestPerchEmbedderPreprocessing:
    def test_resamples_and_pads_to_window_shape(self, monkeypatch):
        import experiments.wrappers.perch as perch

        seen = {}

        def fake_embed(windows_32k, batch_size=8):
            w = np.asarray(windows_32k)
            seen["shape"] = w.shape
            seen["dtype"] = w.dtype
            # Контракт реального perch.embed: вход [N, 160000].
            assert w.ndim == 2, f"expected 2-D, got {w.shape}"
            assert w.shape[1] == perch.WIN_SAMPLES, w.shape
            return np.ones((w.shape[0], 1280), dtype=np.float32), None

        monkeypatch.setattr(perch, "embed", fake_embed)

        # Короткий сигнал @16k -> ресемпл до 32k -> pad до 160000.
        wav = np.full(16000, 0.3, dtype=np.float32)
        out = PerchEmbedder().embed(wav, 16000)

        assert seen["shape"] == (1, perch.WIN_SAMPLES)
        assert out.shape == (1280,)
        assert out.dtype == np.float32

    def test_truncates_overlong_signal(self, monkeypatch):
        import experiments.wrappers.perch as perch

        captured = {}

        def fake_embed(windows_32k, batch_size=8):
            captured["len"] = np.asarray(windows_32k).shape[1]
            assert captured["len"] == perch.WIN_SAMPLES
            return np.zeros((1, 1280), dtype=np.float32), None

        monkeypatch.setattr(perch, "embed", fake_embed)
        # 10 c @ 32k = 320000 -> truncate до 160000.
        wav = np.zeros(320000, dtype=np.float32)
        PerchEmbedder().embed(wav, 32000)
        assert captured["len"] == perch.WIN_SAMPLES

    def test_downmixes_stereo(self, monkeypatch):
        import experiments.wrappers.perch as perch

        def fake_embed(windows_32k, batch_size=8):
            w = np.asarray(windows_32k)
            assert w.ndim == 2 and w.shape[1] == perch.WIN_SAMPLES
            return np.zeros((1, 1280), dtype=np.float32), None

        monkeypatch.setattr(perch, "embed", fake_embed)
        stereo = np.zeros((perch.WIN_SAMPLES, 2), dtype=np.float32)
        out = PerchEmbedder().embed(stereo, 32000)
        assert out.shape == (1280,)

    def test_dim_property(self):
        assert PerchEmbedder.DIM == 1280


# ---------------------------------------------------------------------------
# Perch2Embedder — препроцессинг через монкипатч (TF-free); DIM=1536
# ---------------------------------------------------------------------------


class TestPerch2EmbedderPreprocessing:
    def test_resamples_and_pads_to_window_shape_1536(self, monkeypatch):
        import experiments.wrappers.perch_v2 as perch_v2

        seen = {}

        def fake_embed(windows_32k, batch_size=8):
            w = np.asarray(windows_32k)
            seen["shape"] = w.shape
            seen["dtype"] = w.dtype
            # Контракт experiments.wrappers.perch_v2.embed: вход [N, 160000] @32k.
            assert w.ndim == 2, f"expected 2-D, got {w.shape}"
            assert w.shape[1] == perch_v2.WIN_SAMPLES, w.shape
            return np.ones((w.shape[0], perch_v2.DIM), dtype=np.float32), None

        monkeypatch.setattr(perch_v2, "embed", fake_embed)

        # 48k stereo -> downmix -> ресемпл 32k -> pad/truncate до 160000.
        stereo = np.zeros((48000, 2), dtype=np.float32)
        out = Perch2Embedder().embed(stereo, 48000)

        assert seen["shape"] == (1, perch_v2.WIN_SAMPLES)
        assert out.shape == (1536,)
        assert out.dtype == np.float32

    def test_dim_property_is_1536_not_1280(self):
        assert Perch2Embedder.DIM == 1536
        assert Perch2Embedder.DIM != PerchEmbedder.DIM  # distinct embedding spaces


# ---------------------------------------------------------------------------
# YamnetEmbedder — препроцессинг через монкипатч (TF-free)
# ---------------------------------------------------------------------------


class TestYamnetEmbedderPreprocessing:
    def test_resamples_to_16k_and_pools_concat_mean_max(self, monkeypatch):
        import experiments.wrappers.yamnet_probe as yp

        seen = {}

        def fake_embed_waveform(x_16k):
            x = np.asarray(x_16k)
            seen["ndim"] = x.ndim
            assert x.ndim == 1, f"expected mono 1-D, got {x.shape}"
            # 5 фреймов по 1024 — как реальный YAMNet.
            return np.ones((5, 1024), dtype=np.float32)

        monkeypatch.setattr(yp, "embed_waveform", fake_embed_waveform)

        # Сигнал @48k -> ресемпл до 16k.
        wav = np.full(48000, 0.2, dtype=np.float32)
        out = YamnetEmbedder().embed(wav, 48000)

        assert seen["ndim"] == 1
        assert out.shape == (2048,)  # concat(mean=1024, max=1024)
        assert out.dtype == np.float32

    def test_empty_frames_returns_zero_vector(self, monkeypatch):
        import experiments.wrappers.yamnet_probe as yp

        def fake_embed_waveform(x_16k):
            return np.zeros((0, 1024), dtype=np.float32)

        monkeypatch.setattr(yp, "embed_waveform", fake_embed_waveform)
        out = YamnetEmbedder().embed(np.zeros(16000, dtype=np.float32), 16000)
        assert out.shape == (2048,)
        assert np.array_equal(out, np.zeros(2048, dtype=np.float32))

    def test_downmixes_stereo(self, monkeypatch):
        import experiments.wrappers.yamnet_probe as yp

        def fake_embed_waveform(x_16k):
            assert np.asarray(x_16k).ndim == 1
            return np.ones((3, 1024), dtype=np.float32)

        monkeypatch.setattr(yp, "embed_waveform", fake_embed_waveform)
        stereo = np.zeros((16000, 2), dtype=np.float32)
        out = YamnetEmbedder().embed(stereo, 16000)
        assert out.shape == (2048,)

    def test_dim_property(self):
        assert YamnetEmbedder.DIM == 2048


# ---------------------------------------------------------------------------
# Honesty: синтетические эмбеддинги не порождают видовых метрик
# ---------------------------------------------------------------------------


def test_synthetic_embeddings_are_not_species_metrics():
    """Документирующий тест: числа из FakeEmbedder — SYNTHETIC, не метрика.

    Это страховка против того, что кто-то начнёт интерпретировать величины
    синтетических эмбеддингов как качество классификации видов.
    """
    emb = FakeEmbedder(dim=4)
    out = embed_batch([(np.ones(10, dtype=np.float32), 16000)], emb)
    # embed_batch really ran on the fake embedder: shape + finite values. The
    # numbers carry NO species meaning — they are SYNTHETIC by construction.
    assert out.shape == (1, 4)
    assert np.all(np.isfinite(out))


def test_embedding_cache_load_corrupt_raises_clear_error(tmp_path):
    """Битый .npz -> понятный ValueError с путём, а не сырой BadZipFile/KeyError."""
    import pytest

    from faun.embeddings import EmbeddingCache

    bad = tmp_path / "broken.npz"
    bad.write_bytes(b"not a real npz file")
    with pytest.raises(ValueError, match="corrupt embedding cache"):
        EmbeddingCache.load(bad)
