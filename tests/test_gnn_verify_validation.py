import math

import numpy as np
import pyarrow as pa
import pytest

from gnn_features.contract import DAY
from gnn_features.verify_validation import assert_finite_nullable, require_validation_date


def scalar_finite(values):
    return all(value is None or math.isfinite(value) for value in values)


@pytest.mark.parametrize('null_kind', ['sparse', 'dense', 'all', 'none'])
@pytest.mark.parametrize('chunked', [False, True])
def test_large_nullable_values_match_scalar_definition(null_kind, chunked):
    length = 100_000
    raw = np.linspace(-1.e25, 1.e25, length, dtype=np.float64)
    nulls = np.zeros(length, dtype=bool)
    if null_kind == 'sparse':
        nulls[[1, 16_384, 77_777, 99_999]] = True
    elif null_kind == 'dense':
        nulls[:] = True
        nulls[[0, 16_383, 65_000, 99_998]] = False
    elif null_kind == 'all':
        nulls[:] = True
    # Underlying NULL slots may contain any bit pattern, including NaN/Inf.
    raw[nulls] = np.inf
    array = pa.array(raw, mask=nulls, type=pa.float64())
    column = pa.chunked_array([array.slice(0, 16_384), array.slice(16_384, 49_152), array.slice(65_536)]) if chunked else array
    assert scalar_finite(column.to_pylist())
    assert_finite_nullable(column)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('position', [0, 16_384, 77_777, 99_999])
def test_reject_nonnull_nonfinite_among_dense_nulls(bad, position):
    raw = np.zeros(100_000, dtype=np.float64)
    nulls = np.ones(100_000, dtype=bool)
    nulls[position] = False
    raw[position] = bad
    array = pa.array(raw, mask=nulls, type=pa.float64(), from_pandas=False)
    column = pa.chunked_array([array.slice(0, 16_384), array.slice(16_384)])
    assert not scalar_finite(column.to_pylist())
    with pytest.raises(AssertionError, match='Nonfinite non-NULL'):
        assert_finite_nullable(column)


def test_empty_and_all_null_float_arrays():
    for column in [pa.array([], type=pa.float64()), pa.array([None] * 1000, type=pa.float32()),
                   pa.chunked_array([], type=pa.float64())]:
        assert scalar_finite(column.to_pylist())
        assert_finite_nullable(column)


def test_randomized_scalar_equivalence_with_nullable_nonfinites():
    rng = np.random.default_rng(20260928)
    options = [None, -100., 0., 0.125, 1.e300, float('nan'), float('inf'), -float('inf')]
    for _ in range(100):
        values = [options[int(index)] for index in rng.integers(0, len(options), size=257)]
        column = pa.chunked_array([pa.array(values[:128], type=pa.float64(), from_pandas=False),
                                  pa.array(values[128:], type=pa.float64(), from_pandas=False)])
        expected = scalar_finite(values)
        if expected:
            assert_finite_nullable(column)
        else:
            with pytest.raises(AssertionError, match='Nonfinite non-NULL'):
                assert_finite_nullable(column)


@pytest.mark.parametrize('date', ['2022-09-11', '2022-09-12', '2022-09-13'])
def test_accepts_only_complete_matching_validation_days(date):
    start = int(np.datetime64(date, 'us').astype(np.int64))
    prep = dict(date=date, start_us=start, end_us=start + DAY)
    require_validation_date(dict(date=date), prep)
    with pytest.raises(ValueError, match='complete day'):
        require_validation_date(dict(date=date), {**prep, 'end_us': start + DAY - 1})
    with pytest.raises(ValueError, match='restricted to Validation'):
        require_validation_date(dict(date='2022-09-14'), prep)


@pytest.mark.parametrize('date', ['2022-09-10', '2022-09-14', '2022-09-16', '2022-09-28', None])
def test_rejects_training_test_tail_and_missing_dates(date):
    with pytest.raises(ValueError, match='restricted to Validation'):
        require_validation_date(dict(date=date), dict(date=date))
