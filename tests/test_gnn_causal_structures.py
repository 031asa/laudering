"""Small semantic and brute-force checks of exact causal, label-free motifs."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from gnn_causal_structures import CausalStructures, amount_match


def arrays(rows):
    # src,dst,time,paid,received,payment_currency,receiving_currency
    names = ("src", "dst", "timestamp", "paid_scaled", "received_scaled",
             "payment_currency_cat", "receiving_currency_cat")
    a = {k: np.asarray([row[i] for row in rows], dtype=np.int64) for i, k in enumerate(names)}
    size = max(int(a["src"].max()), int(a["dst"].max())) + 1
    for direction, column in (("in", "dst"), ("out", "src")):
        a[direction + "_order"] = np.lexsort((np.arange(len(rows)), a["timestamp"], a[column]))
        a[direction + "_ptr"] = np.r_[0, np.cumsum(np.bincount(a[column], minlength=size))]
    return a


def e(src, dst, time, amount=100, pc=1, rc=None, received=None):
    return (src, dst, time, amount, amount if received is None else received, pc, pc if rc is None else rc)


def run(rows, target=-1, window=100, **kw):
    target = len(rows) - 1 if target == -1 else target
    return CausalStructures(arrays(rows), window_us=window, **kw).extract(target)


def test_boundaries_same_timestamp_self_history_and_direction():
    rows = [e(9, 8, 0), e(1, 2, 99), e(1, 2, 100), e(2, 2, 180),
            e(1, 2, 199), e(1, 2, 200), e(1, 2, 201), e(2, 4, 199), e(2, 3, 200)]
    f = run(rows)["features"]
    assert f["prior_in_count_72h"] == 2
    assert f["prior_in_amount_match_count_72h"] == 2
    assert f["full_window_72h"] is True
    assert f["prior_in_match_min_gap_minutes_72h"] == 1 / 60_000000


def test_chain_has_four_distinct_accounts_strict_time_and_currency():
    rows = [e(0, 1, 130), e(1, 2, 160), e(2, 1, 110), e(3, 1, 120),
            e(4, 1, 160), e(5, 1, 100, pc=2), e(2, 3, 200)]
    out = run(rows)
    assert out["features"]["chain3_path_count_72h"] == 1
    assert out["features"]["chain3_amount_match_path_count_72h"] == 1
    assert out["witnesses"]["chain3"] == [0, 1, 6]


def test_receipt_currency_for_one_hop_but_both_sides_for_paths():
    rows = [e(0, 1, 120, pc=2, rc=1), e(1, 2, 150, pc=2, rc=1), e(2, 3, 200)]
    f = run(rows)["features"]
    assert f["prior_in_same_currency_count_72h"] == 1
    assert f["prior_in_amount_match_count_72h"] == 1
    assert f["chain3_path_count_72h"] == 0


def test_cycle_two_and_three_and_closing_join_amount():
    rows = [e(3, 2, 120), e(3, 1, 130), e(1, 2, 160),
            e(3, 4, 125, amount=1, received=100), e(4, 2, 170), e(2, 3, 200)]
    out = run(rows)
    f, w = out["features"], out["witnesses"]
    assert f["cycle2_count_72h"] == f["cycle2_amount_match_count_72h"] == 1
    assert f["cycle3_path_count_72h"] == 2
    assert f["cycle3_amount_match_path_count_72h"] == 1
    assert w["cycle2"] == [0, 5]
    assert w["cycle3_amount_match"] == [1, 2, 5]
    assert f["chain3_path_count_72h"] == 0


def test_parallel_paths_count_transactions_and_distinct_middle_accounts():
    rows = [e(2, 1, 110), e(2, 1, 120), e(1, 3, 130), e(1, 3, 150),
            e(2, 4, 150, amount=20), e(4, 3, 180, amount=100), e(2, 3, 200)]
    out = run(rows)
    f, w = out["features"], out["witnesses"]
    assert f["parallel2_path_count_72h"] == 5
    assert f["parallel2_middle_accounts_72h"] == 2
    assert f["parallel2_amount_match_path_count_72h"] == 4
    assert f["parallel2_amount_match_middle_accounts_72h"] == 1
    assert w["parallel2"] == [0, 2]
    assert w["parallel2_amount_match"] == [1, 2]


def test_cross_currency_target_has_missing_rings_and_parallel_paths():
    out = run([e(1, 2, 120), e(2, 3, 200, pc=1, rc=2)])
    f = out["features"]
    assert f["prior_in_amount_match_count_72h"] == 1
    assert f["cycle2_count_72h"] is None
    assert f["parallel2_path_count_72h"] is None
    assert f["same_currency_target"] is False


def test_zero_only_matches_zero_and_empty_history_is_zero():
    assert amount_match(0, 0)
    assert not amount_match(1, 0)
    assert not amount_match(0, 1)
    out = run([e(0, 1, 120, amount=0), e(1, 2, 160, amount=0), e(2, 3, 200, amount=0)])
    assert out["features"]["chain3_amount_match_path_count_72h"] == 1
    empty = run([e(2, 3, 200)])["features"]
    assert empty["prior_in_count_72h"] == 0
    assert empty["chain3_path_count_72h"] == 0
    assert empty["prior_in_match_min_gap_minutes_72h"] is None
    assert empty["full_window_72h"] is False


def test_large_integer_boundaries_do_not_round_or_overflow():
    value = 8_000_000_000_000_000_000
    lo, hi = 7_200_000_000_000_000_000, 8_800_000_000_000_000_000
    assert amount_match(lo, value) and amount_match(hi, value)
    assert not amount_match(lo - 1, value) and not amount_match(hi + 1, value)
    rows = [e(0, 1, 110, received=lo - 1), e(0, 1, 120, received=lo),
            e(0, 1, 130, received=hi), e(0, 1, 140, received=hi + 1),
            e(1, 2, 160, amount=value), e(2, 3, 200, amount=value)]
    out = run(rows)
    assert out["features"]["chain3_path_count_72h"] == 4
    assert out["features"]["chain3_amount_match_path_count_72h"] == 2
    maximum = np.iinfo(np.int64).max
    out = run([e(0, 1, 130, amount=maximum), e(1, 2, 160, amount=maximum), e(2, 3, 200, amount=maximum)])
    assert out["features"]["chain3_amount_match_path_count_72h"] == 1
    assert out["features"]["prior_in_amount_match_count_72h"] == 1


def test_current_self_transfer_is_not_applicable_not_zero():
    out = run([e(1, 2, 130), e(2, 2, 200)])
    assert out["features"]["not_applicable"] is True
    assert out["features"]["eligible"] is False
    assert out["features"]["prior_in_count_72h"] is None
    assert out["work"]["adjacency_rows_examined"] == 0


def test_future_and_labels_do_not_change_features_or_witnesses():
    rows = [e(0, 1, 120), e(1, 2, 150), e(2, 3, 200)]
    old = run(rows)
    with_future = rows + [e(3, 2, 220), e(2, 1, 250), e(1, 3, 210)]
    a = arrays(with_future)
    class LabelTrap(dict):
        def __getitem__(self, key):
            assert key not in ("label", "is_laundering")
            return super().__getitem__(key)
    a["label"] = np.ones(len(with_future))
    engine = CausalStructures(LabelTrap(a), window_us=100)
    new = engine.extract(2)
    assert new["features"] == old["features"]
    assert new["witnesses"] == old["witnesses"]
    a["label"][:] = 0
    assert engine.extract(2)["features"] == old["features"]


def test_exact_counts_without_path_caps_and_bounded_cache():
    rows = [e(0, 1, 110 + i % 5) for i in range(90)] + [e(1, 2, 150 + i % 5) for i in range(80)] + [e(2, 3, 200)]
    a = arrays(rows)
    engine = CausalStructures(a, window_us=100, max_cache_bytes=1024)
    out = engine.extract(len(rows) - 1)
    assert out["features"]["chain3_path_count_72h"] == 90 * 80
    assert out["features"]["chain3_amount_match_path_count_72h"] == 90 * 80
    assert engine.cache_bytes <= 1024
    assert out["work"]["pair_first_rows"] < 90 * 80
    assert out["work"]["pair_second_rows"] < 90 * 80
    with pytest.raises(ValueError):
        CausalStructures(a, max_cache_bytes=129 * 1024 * 1024)


@pytest.mark.parametrize("seed", range(12))
def test_pair_sweep_agrees_with_brute_force(seed):
    rng = np.random.default_rng(seed)
    rows = [e(0, 1, int(rng.integers(100, 190)), amount=int(rng.integers(0, 150))) for _ in range(15)]
    rows += [e(1, 2, int(rng.integers(100, 190)), amount=int(rng.integers(0, 150))) for _ in range(13)]
    rows += [e(2, 3, 200, amount=100)]
    count = match = 0
    for first in rows[:15]:
        for second in rows[15:28]:
            if first[2] < second[2]:
                count += 1
                match += int(amount_match(first[4], second[3]) and amount_match(second[4], 100))
    out = run(rows)
    assert out["features"]["chain3_path_count_72h"] == count
    assert out["features"]["chain3_amount_match_path_count_72h"] == match
