from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from emosi import agree

COLUMNS = [
    "annotator",
    "utterance_id",
    "film_id",
    "scene_id",
    "role",
    "label",
    "confidence",
    "labeled_at",
]


def make_labels(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLUMNS)


def vote_rows() -> pd.DataFrame:
    base = {"film_id": "AAA01", "scene_id": "AAA01-S001", "labeled_at": ""}
    return make_labels(
        [
            {**base, "annotator": "kesan", "utterance_id": "u1", "role": "milik",
             "label": "senang", "confidence": 4},
            {**base, "annotator": "budi", "utterance_id": "u1", "role": "silang",
             "label": "senang", "confidence": 3},
            {**base, "annotator": "kesan", "utterance_id": "u2", "role": "milik",
             "label": "senang", "confidence": 5},
            {**base, "annotator": "budi", "utterance_id": "u2", "role": "silang",
             "label": "sedih", "confidence": 2},
            {**base, "annotator": "ani", "utterance_id": "u2", "role": "silang",
             "label": "senang", "confidence": 4},
            {**base, "annotator": "kesan", "utterance_id": "u3", "role": "milik",
             "label": "sedih", "confidence": 5},
            {**base, "annotator": "budi", "utterance_id": "u3", "role": "silang",
             "label": "senang", "confidence": 5},
            {**base, "annotator": "kesan", "utterance_id": "u4", "role": "milik",
             "label": "netral", "confidence": 3},
        ]
    )


def test_cohen_kappa_known_value():
    first = "aaaaabbbbb"
    second = "aaaababbbb"
    value, observed = agree.cohen_kappa(first, second)
    assert observed == pytest.approx(0.8)
    assert value == pytest.approx(0.6)


def test_cohen_kappa_chance_is_zero():
    value, observed = agree.cohen_kappa("aabb", "abab")
    assert observed == pytest.approx(0.5)
    assert value == pytest.approx(0.0)


def test_cohen_kappa_single_category_is_undefined():
    value, observed = agree.cohen_kappa("aaaa", "aaaa")
    assert observed == pytest.approx(1.0)
    assert math.isnan(value)


def test_fleiss_kappa_known_matrix():
    counts = np.array([[3, 0], [2, 1], [1, 2], [0, 3]], dtype=float)
    value, observed = agree.fleiss_kappa(counts)
    assert observed == pytest.approx(2 / 3, abs=1e-6)
    assert value == pytest.approx(1 / 3, abs=1e-6)


def test_fleiss_kappa_unanimous_across_categories_is_one():
    counts = np.array([[3, 0], [3, 0], [0, 3], [0, 3]], dtype=float)
    value, observed = agree.fleiss_kappa(counts)
    assert observed == pytest.approx(1.0)
    assert value == pytest.approx(1.0)


def test_fleiss_kappa_with_one_category_is_undefined():
    counts = np.array([[3, 0], [3, 0]], dtype=float)
    value, observed = agree.fleiss_kappa(counts)
    assert observed == pytest.approx(1.0)
    assert math.isnan(value)


def test_fleiss_kappa_drops_single_rater_items():
    counts = np.array([[1, 0], [2, 1]], dtype=float)
    value, _ = agree.fleiss_kappa(counts)
    assert not math.isnan(value)


def test_basis_prefers_gold_then_cross():
    assert agree.basis_of(["milik", "gold"]) == "gold"
    assert agree.basis_of(["milik", "silang"]) == "silang"
    assert agree.basis_of(["milik"]) == "milik"


def test_add_basis_uses_all_raters_of_an_utterance():
    labels = agree.add_basis(vote_rows())
    basis = labels.groupby("utterance_id")["basis"].first().to_dict()
    assert basis["u1"] == "silang"
    assert basis["u4"] == "milik"


def test_item_votes_cover_every_status():
    votes = agree.item_votes(vote_rows()).set_index("utterance_id")
    assert votes.loc["u1", "status"] == agree.STATUS_AGREE
    assert votes.loc["u1", "label"] == "senang"
    assert votes.loc["u2", "status"] == agree.STATUS_MAJORITY
    assert votes.loc["u2", "label"] == "senang"
    assert votes.loc["u2", "agreement"] == pytest.approx(2 / 3, abs=1e-3)
    assert votes.loc["u3", "status"] == agree.STATUS_PENDING
    assert votes.loc["u3", "label"] == ""
    assert votes.loc["u4", "status"] == agree.STATUS_SINGLE
    assert votes.loc["u4", "label"] == "netral"


def test_item_votes_reports_vote_counts():
    votes = agree.item_votes(vote_rows()).set_index("utterance_id")
    assert votes.loc["u2", "suara"] == "senang×2, sedih×1"


def test_decisions_resolve_only_pending_items():
    votes = agree.item_votes(vote_rows())
    decided = agree.apply_decisions(
        votes,
        {"u3": {"decision": "sedih", "decided_by": "koordinator", "notes": "konteks"}},
        allowed={"senang", "sedih", "marah", "takut", "netral"},
    )
    row = decided.set_index("utterance_id").loc["u3"]
    assert row["status"] == agree.STATUS_DECIDED
    assert row["label"] == "sedih"
    assert row["decided_by"] == "koordinator"
    # Baris yang sudah sepakat tidak boleh berubah.
    assert decided.set_index("utterance_id").loc["u1", "label"] == "senang"


def test_invalid_decision_is_rejected():
    votes = agree.item_votes(vote_rows())
    with pytest.raises(agree.AgreementError, match="bukan label sah"):
        agree.apply_decisions(
            votes, {"u3": {"decision": "bahagia"}}, allowed={"senang", "sedih"}
        )


def test_final_rows_drop_pending_items():
    votes = agree.item_votes(vote_rows())
    final = agree.final_rows(votes)
    assert set(final["utterance_id"]) == {"u1", "u2", "u4"}
    assert list(final.columns) == agree.FINAL_COLUMNS


def test_adjudication_rows_keep_decided_items_with_their_decision():
    votes = agree.item_votes(vote_rows())
    decided = agree.apply_decisions(votes, {"u3": {"decision": "takut"}})
    rows = agree.adjudication_rows(decided).set_index("utterance_id")
    assert set(rows.index) == {"u3"}
    assert rows.loc["u3", "decision"] == "takut"
    assert rows.loc["u3", "status"] == agree.STATUS_DECIDED


def test_read_decisions_skips_empty_rows():
    frame = pd.DataFrame(
        [
            {"utterance_id": "u3", "decision": "takut", "decided_by": "a", "notes": ""},
            {"utterance_id": "u4", "decision": "", "decided_by": "", "notes": ""},
        ]
    )
    decisions = agree.read_decisions(frame)
    assert list(decisions) == ["u3"]


def test_read_decisions_ignores_nan_cells():
    frame = pd.DataFrame(
        [
            {"utterance_id": "u3", "decision": float("nan"),
             "decided_by": float("nan"), "notes": float("nan")},
            {"utterance_id": "u4", "decision": "takut", "decided_by": "ani",
             "notes": ""},
        ]
    )
    assert list(agree.read_decisions(frame)) == ["u4"]


def test_load_labels_keeps_the_latest_row_per_annotator():
    old = make_labels(
        [
            {"annotator": "kesan", "utterance_id": "u1", "film_id": "AAA01",
             "scene_id": "AAA01-S001", "role": "milik", "label": "sedih",
             "confidence": 2, "labeled_at": "2026-01-01T00:00:00+00:00"},
        ]
    )
    new = old.copy()
    new["label"] = "senang"
    new["labeled_at"] = "2026-01-02T00:00:00+00:00"
    labels = agree.load_labels([old, new])
    assert len(labels) == 1
    assert labels.loc[0, "label"] == "senang"


def test_validate_labels_lists_unknown_tokens():
    labels = make_labels(
        [
            {"annotator": "kesan", "utterance_id": "u1", "film_id": "AAA01",
             "scene_id": "AAA01-S001", "role": "milik", "label": "bahagia",
             "confidence": 3, "labeled_at": ""},
        ]
    )
    assert agree.validate_labels(labels, {"senang", "sedih"}) == ["bahagia"]


def test_overlap_matrix_counts_shared_utterances():
    table = agree.overlap_matrix(vote_rows())
    assert table.loc["kesan", "budi"] == 3
    assert table.loc["kesan", "ani"] == 1
    assert table.loc["kesan", "kesan"] == 4


def test_agreement_table_has_a_row_per_basis():
    base = {"film_id": "AAA01", "scene_id": "AAA01-S001", "labeled_at": ""}
    rows = []
    for person, label in zip(
        ("kesan", "budi", "ani", "dina"), ("senang", "senang", "senang", "senang"),
        strict=True,
    ):
        rows.append({**base, "annotator": person, "utterance_id": "g1",
                     "role": "gold", "label": label, "confidence": 4})
    for person, label in zip(("kesan", "budi"), ("senang", "sedih"), strict=True):
        rows.append({**base, "annotator": person, "utterance_id": "g2",
                     "role": "gold", "label": label, "confidence": 3})
    rows.append({**base, "annotator": "kesan", "utterance_id": "m1",
                 "role": "milik", "label": "netral", "confidence": 3})
    rows.append({**base, "annotator": "budi", "utterance_id": "m1",
                 "role": "milik", "label": "netral", "confidence": 5})
    rows.append({**base, "annotator": "kesan", "utterance_id": "s1",
                 "role": "milik", "label": "takut", "confidence": 4})
    rows.append({**base, "annotator": "budi", "utterance_id": "s1",
                 "role": "silang", "label": "takut", "confidence": 4})

    _, table = agree.agreement_table(make_labels(rows), min_items=1)
    assert {"gold", "silang", "milik", "semua"} <= set(table["basis"])
    assert (table["metode"] == "fleiss").any()
