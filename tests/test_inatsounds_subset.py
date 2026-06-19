"""Тесты чистой селекции reserve-подмножества iNatSounds — без tar, без файлов, без TF.

Покрывают контракт ``select_targets`` (см.
``scripts/extract_inatsounds_subset.py``):
    (a) reserve-совпадения идут первыми и отсортированы по clip_count убыв.;
    (b) top-up добивает до n_species из не-reserve Aves;
    (c) пол min_clips отбрасывает слишком маленькие виды;
    (d) cap переносится, но виды всё равно возвращаются;
    (e) dedup по имени;
    (f) fail-loud путь (мало reserve-совпадений) ВСЁ РАВНО возвращает виды
        (через top-up) и НЕ бросает исключение.

Каталог — синтетический список словарей (NO tar, NO files). Проверяется
детерминизм. Импорт через ``scripts.extract_inatsounds_subset`` работает под
``python -m pytest`` из корня репозитория (scripts — namespace-пакет, без
``scripts/__init__.py``).
"""

from __future__ import annotations

import logging

from scripts.extract_inatsounds_subset import select_targets

# ---------------------------------------------------------------------------
# Хелперы построения синтетического каталога
# ---------------------------------------------------------------------------


def _entry(name: str, clip_count: int, audio_dir_name: str | None = None) -> dict:
    """Одна запись каталога ``{"name","audio_dir_name","clip_count"}`` (только Aves)."""
    return {
        "name": name,
        "audio_dir_name": audio_dir_name or f"dir_{name.replace(' ', '_')}",
        "clip_count": clip_count,
    }


# Несколько настоящих reserve-видов (для приоритета 1) + не-reserve Aves (для top-up).
RESERVE_SAMPLE = [
    "Turdus merula",
    "Parus major",
    "Erithacus rubecula",
    "Fringilla coelebs",
]


def _names(selected: list[dict]) -> list[str]:
    return [e["name"] for e in selected]


# ---------------------------------------------------------------------------
# (a) reserve-совпадения первыми и отсортированы по clip_count убыв.
# ---------------------------------------------------------------------------


def test_reserve_matches_chosen_first_and_sorted_desc() -> None:
    catalog = [
        _entry("Turdus merula", 100),
        _entry("Parus major", 300),
        _entry("Erithacus rubecula", 200),
        _entry("Corvus monedula", 999),  # не в RESERVE_SAMPLE -> только в top-up
    ]
    selected = select_targets(
        catalog, RESERVE_SAMPLE, n_species=3, cap=120, min_clips=10
    )

    # Три reserve-вида заполняют n_species=3; не-reserve (999) не вытесняет их.
    assert _names(selected) == ["Parus major", "Erithacus rubecula", "Turdus merula"]
    # Сортировка по clip_count убыв.
    assert [e["clip_count"] for e in selected] == [300, 200, 100]
    # Ни один не помечен как top-up (все из reserve).
    assert all(not e.get("_topped_up") for e in selected)


def test_reserve_hits_truncated_to_n_species_no_topup() -> None:
    # 4 reserve-совпадения, но n_species=2: берём 2 с наибольшим clip_count,
    # 2 меньших — отсутствуют, top-up НЕ запускается (reserve уже заполнил
    # n_species), значит ни один не помечен _topped_up (#CR-3).
    catalog = [
        _entry("Turdus merula", 100),
        _entry("Parus major", 400),
        _entry("Erithacus rubecula", 300),
        _entry("Fringilla coelebs", 50),
        _entry("Corvus monedula", 999),  # не-reserve: не должен влезть
    ]
    selected = select_targets(
        catalog, RESERVE_SAMPLE, n_species=2, cap=120, min_clips=10
    )
    assert _names(selected) == ["Parus major", "Erithacus rubecula"]
    assert "Turdus merula" not in _names(selected)
    assert "Fringilla coelebs" not in _names(selected)
    assert "Corvus monedula" not in _names(selected)  # top-up не сработал
    assert all(not e.get("_topped_up") for e in selected)


def test_reserve_tie_broken_by_name_asc() -> None:
    # Одинаковый clip_count -> детерминированный тай-брейк по имени возр.
    catalog = [
        _entry("Parus major", 50),
        _entry("Erithacus rubecula", 50),
        _entry("Turdus merula", 50),
    ]
    selected = select_targets(
        catalog, RESERVE_SAMPLE, n_species=3, cap=120, min_clips=10
    )
    assert _names(selected) == ["Erithacus rubecula", "Parus major", "Turdus merula"]


# ---------------------------------------------------------------------------
# (b) top-up добивает до n_species из не-reserve Aves
# ---------------------------------------------------------------------------


def test_topup_fills_to_n_species_from_non_reserve() -> None:
    catalog = [
        _entry("Turdus merula", 100),  # reserve
        _entry("Corvus monedula", 80),  # не-reserve Aves
        _entry("Sturnus roseus", 70),  # не-reserve Aves
        _entry("Passer domesticus", 60),  # не-reserve Aves
    ]
    selected = select_targets(
        catalog, RESERVE_SAMPLE, n_species=3, cap=120, min_clips=10
    )

    # 1 reserve + 2 top-up = 3.
    assert len(selected) == 3
    assert _names(selected)[0] == "Turdus merula"  # reserve первым
    # top-up по clip_count убыв.: Corvus(80) -> Sturnus(70).
    assert _names(selected)[1:] == ["Corvus monedula", "Sturnus roseus"]
    topped = [e for e in selected if e.get("_topped_up")]
    assert {e["name"] for e in topped} == {"Corvus monedula", "Sturnus roseus"}


def test_topup_does_not_reselect_reserve_entry() -> None:
    # Reserve-вид не должен попасть второй раз через top-up.
    catalog = [
        _entry("Turdus merula", 100),  # reserve
        _entry("Corvus monedula", 80),
    ]
    selected = select_targets(
        catalog, RESERVE_SAMPLE, n_species=5, cap=120, min_clips=10
    )
    assert _names(selected).count("Turdus merula") == 1
    assert len(selected) == 2


# ---------------------------------------------------------------------------
# (c) пол min_clips отбрасывает слишком маленькие виды
# ---------------------------------------------------------------------------


def test_min_clips_floor_excludes_too_small() -> None:
    catalog = [
        _entry("Turdus merula", 100),  # reserve, проходит
        _entry("Parus major", 5),  # reserve, НО < min_clips=40 -> выпадает
        _entry("Corvus monedula", 3),  # не-reserve, тоже выпадает
    ]
    selected = select_targets(
        catalog, RESERVE_SAMPLE, n_species=5, cap=120, min_clips=40
    )
    assert _names(selected) == ["Turdus merula"]
    # Ни один возвращённый вид не ниже пола.
    assert all(e["clip_count"] >= 40 for e in selected)


def test_min_clips_floor_blocks_topup_too() -> None:
    # top-up тоже уважает min_clips.
    catalog = [
        _entry("Turdus merula", 100),  # reserve
        _entry("Corvus monedula", 20),  # не-reserve, < 40 -> не годится в top-up
    ]
    selected = select_targets(
        catalog, RESERVE_SAMPLE, n_species=5, cap=120, min_clips=40
    )
    assert _names(selected) == ["Turdus merula"]


# ---------------------------------------------------------------------------
# (d) cap переносится, но виды всё равно возвращаются
# ---------------------------------------------------------------------------


def test_cap_is_carried_but_species_still_returned() -> None:
    catalog = [
        _entry("Turdus merula", 500),  # clip_count >> cap
        _entry("Parus major", 500),
    ]
    cap = 120
    selected = select_targets(
        catalog, RESERVE_SAMPLE, n_species=5, cap=cap, min_clips=40
    )
    # cap НЕ режет селекцию — виды возвращаются несмотря на clip_count >> cap.
    assert len(selected) == 2
    # cap перенесён в каждую запись.
    assert all(e["cap"] == cap for e in selected)
    # clip_count сохранён как есть (cap влияет только на извлечение).
    assert all(e["clip_count"] == 500 for e in selected)


# ---------------------------------------------------------------------------
# (e) dedup по имени
# ---------------------------------------------------------------------------


def test_dedup_by_name() -> None:
    # Дубликат имени (разные audio_dir_name / clip_count) -> один раз.
    catalog = [
        _entry("Turdus merula", 100, audio_dir_name="dirA"),
        _entry("Turdus merula", 90, audio_dir_name="dirB"),  # дубль по имени
        _entry("Parus major", 80),
    ]
    selected = select_targets(
        catalog, RESERVE_SAMPLE, n_species=5, cap=120, min_clips=10
    )
    assert _names(selected).count("Turdus merula") == 1
    # Выбран ПЕРВЫЙ встреченный (100, dirA) — детерминированно по порядку каталога.
    merula = next(e for e in selected if e["name"] == "Turdus merula")
    assert merula["audio_dir_name"] == "dirA"
    assert merula["clip_count"] == 100


def test_dedup_normalises_whitespace_and_case() -> None:
    # Нормализация: лишние пробелы / регистр сводятся к одному виду.
    catalog = [
        _entry("Turdus merula", 100),
        _entry("turdus   merula", 90),  # тот же вид после _norm
    ]
    selected = select_targets(
        catalog, RESERVE_SAMPLE, n_species=5, cap=120, min_clips=10
    )
    assert len(selected) == 1


# ---------------------------------------------------------------------------
# (f) fail-loud путь: мало reserve-совпадений -> ВСЁ РАВНО виды, БЕЗ исключения
# ---------------------------------------------------------------------------


def test_fail_loud_still_returns_species_via_topup(caplog) -> None:
    # Ни одного reserve-вида в каталоге -> срабатывает FAIL-LOUD guard (#F3),
    # но функция НЕ падает и добирает виды top-up'ом.
    catalog = [
        _entry("Corvus monedula", 200),
        _entry("Sturnus roseus", 150),
        _entry("Passer domesticus", 100),
    ]
    with caplog.at_level(logging.ERROR, logger="extract_inatsounds_subset"):
        selected = select_targets(
            catalog, RESERVE_SAMPLE, n_species=2, cap=120, min_clips=10
        )

    # НЕ пусто: top-up дал виды.
    assert len(selected) == 2
    assert _names(selected) == ["Corvus monedula", "Sturnus roseus"]
    assert all(e.get("_topped_up") for e in selected)
    # FAIL-LOUD залогирован на уровне ERROR.
    assert any(record.levelno == logging.ERROR for record in caplog.records)


def test_fail_loud_does_not_raise_on_empty_catalog() -> None:
    # Пустой каталог: guard срабатывает, но исключения нет — просто пусто.
    selected = select_targets([], RESERVE_SAMPLE, n_species=5, cap=120, min_clips=10)
    assert selected == []


# ---------------------------------------------------------------------------
# Детерминизм: повтор даёт идентичный результат
# ---------------------------------------------------------------------------


def test_selection_is_deterministic() -> None:
    catalog = [
        _entry("Turdus merula", 100),
        _entry("Parus major", 100),  # тай по clip_count -> тай-брейк по имени
        _entry("Corvus monedula", 90),
        _entry("Sturnus roseus", 90),
        _entry("Passer domesticus", 80),
    ]
    a = select_targets(catalog, RESERVE_SAMPLE, n_species=4, cap=120, min_clips=10)
    b = select_targets(catalog, RESERVE_SAMPLE, n_species=4, cap=120, min_clips=10)
    assert _names(a) == _names(b)
    assert [e["clip_count"] for e in a] == [e["clip_count"] for e in b]
