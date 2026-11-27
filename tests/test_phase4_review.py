"""Human decisions for review-band pairs: strict file format and what a decision may change."""

import pytest

from idefect.data.phase4_leakage import (groups_from_pairs, load_review_decisions, review_decided,
                                         review_needed, same_pairs)


def _result():
    return {"exact": [{"a": "C/x1", "b": "C/x2"}],
            "pairs_same_or_review": [
                {"a": "C/a", "b": "C/b", "fine": 0.31, "kind": "review"},
                {"a": "C/c", "b": "C/d", "fine": 0.42, "kind": "review"},
                {"a": "C/e", "b": "C/f", "fine": 0.71, "kind": "same"}]}


def _write(tmp_path, text):
    f = tmp_path / "review.yaml"
    f.write_text(text)
    return f


def test_distinct_removes_only_that_review_pair(tmp_path):
    dec = load_review_decisions(_write(tmp_path, "dagm:\n  distinct:\n    - [C/b, C/a]\n"), "dagm")
    pairs = same_pairs(_result(), dec)
    assert ("C/a", "C/b") not in pairs
    assert pairs == [("C/x1", "C/x2"), ("C/c", "C/d"), ("C/e", "C/f")]
    assert [(p["a"], p["b"]) for p in review_needed(_result(), dec)] == [("C/c", "C/d")]
    assert [(p["a"], p["decision"]) for p in review_decided(_result(), dec)] == [("C/a", "distinct")]


def test_without_decisions_review_pairs_stay_grouped():
    assert len(same_pairs(_result())) == 4
    assert len(review_needed(_result())) == 2


def test_distinct_cannot_overrule_same_or_exact(tmp_path):
    for pair in ("[C/e, C/f]", "[C/x1, C/x2]"):
        dec = load_review_decisions(_write(tmp_path, f"dagm:\n  distinct:\n    - {pair}\n"), "dagm")
        with pytest.raises(ValueError):
            same_pairs(_result(), dec)


def test_same_decision_adds_pair_inside_the_image_set(tmp_path):
    dec = load_review_decisions(_write(tmp_path, "dagm:\n  same:\n    - [C/g, C/h]\n    - [C/g, D/z]\n"), "dagm")
    pairs = same_pairs(_result(), dec, ids=["C/g", "C/h", "C/a"])
    assert ("C/g", "C/h") in pairs and ("C/g", "D/z") not in pairs
    assert ["C/g", "C/h"] in groups_from_pairs(["C/g", "C/h"], [("C/g", "C/h")])


@pytest.mark.parametrize("text", [
    "dagm:\n  distinct:\n    - [C/a]\n",                          # not a pair
    "dagm:\n  distinct:\n    - [C/a, C/a]\n",                     # same id twice
    "dagm:\n  distinct:\n    - [C/a, C/b]\n    - [C/b, C/a]\n",   # listed twice
    "dagm:\n  distinct:\n    - [C/a, C/b]\n  same:\n    - [C/a, C/b]\n",   # both
    "dagm:\n  maybe:\n    - [C/a, C/b]\n",                        # unknown key
    "neu:\n  distinct: []\n",                                     # unknown dataset
])
def test_malformed_files_raise(tmp_path, text):
    with pytest.raises(ValueError):
        load_review_decisions(_write(tmp_path, text), "dagm")


def test_missing_file_means_no_decisions(tmp_path):
    assert load_review_decisions(tmp_path / "none.yaml", "dagm") == {"same": set(), "distinct": set()}


def test_shipped_review_file_is_valid():
    dec = load_review_decisions("configs/phase4_review.yaml", "dagm")
    assert len(dec["distinct"]) == 9 and not dec["same"]
    assert load_review_decisions("configs/phase4_review.yaml", "kolektorsdd") == {"same": set(), "distinct": set()}
