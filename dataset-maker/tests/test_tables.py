from __future__ import annotations

import pandas as pd

from emosi.tables import load_meta, merge_rows, save_meta


def test_merge_rows_keeps_other_films(tmp_path):
    path = tmp_path / "scenes.csv"
    merge_rows(path, [{"film_id": "AAA01", "index": 1}, {"film_id": "AAA01", "index": 2}])
    frame = merge_rows(path, [{"film_id": "BBB02", "index": 1}])
    assert list(frame["film_id"]) == ["AAA01", "AAA01", "BBB02"]
    assert len(pd.read_csv(path)) == 3


def test_rerun_replaces_rows_of_that_film_only(tmp_path):
    path = tmp_path / "scenes.csv"
    merge_rows(path, [{"film_id": "AAA01", "index": 1}, {"film_id": "AAA01", "index": 2}])
    merge_rows(path, [{"film_id": "BBB02", "index": 1}])
    frame = merge_rows(path, [{"film_id": "AAA01", "index": 1}])
    assert list(frame["film_id"]) == ["AAA01", "BBB02"]
    assert frame[frame["film_id"] == "AAA01"].shape[0] == 1


def test_merge_rows_creates_missing_directories(tmp_path):
    path = tmp_path / "scenes" / "_smoke" / "scenes.csv"
    merge_rows(path, [{"film_id": "AAA01", "index": 1}])
    assert path.exists()


def test_meta_roundtrip_is_sorted_by_film(tmp_path):
    path = tmp_path / "meta.json"
    save_meta(path, {"BBB02": {"n": 2}, "AAA01": {"n": 1}}, generated_by="test")
    meta = load_meta(path)
    assert list(meta) == ["AAA01", "BBB02"]
    assert meta["BBB02"]["n"] == 2


def test_broken_meta_is_treated_as_empty(tmp_path):
    path = tmp_path / "meta.json"
    path.write_text("{bukan json", encoding="utf-8")
    assert load_meta(path) == {}
