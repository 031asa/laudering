"""P80 ties, actual balance gates, paired accounts, and immutable model loading."""
import copy
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from gnn_features import v2_compare as compare
from gnn_features import v2_evaluate as evaluate
from gnn_features.cache_train import sha
from gnn_features.model import BidirectionalGINE


def test_p80_whole_ties_cannot_take_positive_out_of_mixed_top_group():
    assert evaluate.precision80_point([1, 0, 1], [.9, .9, .1]) is None
    p = evaluate.precision80_point([1, 1, 1, 1, 0, 0], [.9, .9, .9, .9, .9, .1])
    assert (p['tp'], p['fp'], p['threshold'], p['precision'], p['recall']) == (4, 1, .9, .8, 1)


def test_same_recall_prefers_higher_precision_and_threshold():
    p = evaluate.precision80_point([1, 1, 1, 1, 0], [.9, .8, .7, .6, .5])
    assert p['threshold'] == .6 and p['precision'] == 1


def test_max_recall_is_chosen_even_when_top_precision_is_higher():
    p = evaluate.precision80_point([1, 1, 1, 0, 1, 0], [.9, .8, .7, .6, .5, .1])
    assert (p['tp'], p['fp'], p['threshold']) == (4, 1, .5)


def test_p80_randomized_brute_force_over_distinct_score_thresholds():
    rng = np.random.default_rng(18)
    for _ in range(80):
        y = rng.integers(0, 2, size=80)
        s = rng.integers(0, 8, size=80)/8
        eligible = [evaluate.confusion(y, s >= t, t) for t in np.unique(s)]
        eligible = [p for p in eligible if p['tp'] > 0 and p['precision'] >= .8]
        expected = max(eligible, key=lambda p: (p['tp'], p['precision'], p['threshold'])) if eligible else None
        assert evaluate.precision80_point(y, s) == expected
        order = rng.permutation(len(y))
        assert evaluate.precision80_point(y[order], s[order]) == expected


@pytest.mark.parametrize('y,s', [([], []), ([1, 2], [.9, .8]), ([1], [np.nan]),
                               ([1], [1.1]), ([1, 0], [.8]), ([[1]], [[.8]])])
def test_invalid_threshold_inputs(y, s):
    with pytest.raises(ValueError):
        evaluate.precision80_point(y, s)


def test_no_positive_class_or_no_qualified_threshold_returns_none():
    assert evaluate.precision80_point([0, 0], [.8, .1]) is None
    assert evaluate.precision80_point([1, 0], [.1, .9]) is None


def selection_point(tp=3469, fp=867, non_ach_tp=143):
    point = dict(tp=tp, fp=fp, positives=6420, alerts=tp+fp, recall=tp/6420, precision=tp/(tp+fp))
    non = dict(tp=non_ach_tp, positives=715, recall=non_ach_tp/715)
    return point, non


@pytest.mark.parametrize('tp,fp,ntp,passed', [(3469, 867, 143, True), (3468, 0, 715, False),
                                           (3469, 868, 143, False), (4000, 0, 142, False)])
def test_integer_balance_gates(tp, fp, ntp, passed):
    point, non = selection_point(tp, fp, ntp)
    assert compare.qualification(point, non)['passed'] is passed


def test_gate_rejects_smaller_validation_population():
    point, non = selection_point()
    point['positives'] = 6000
    with pytest.raises(ValueError, match='full Validation'):
        compare.qualification(point, non)


def candidates():
    results = []
    for name in compare.NAMES:
        p, n = selection_point()
        results.append(dict(name=name, precision80=p, non_ach_precision80=n,
                            metrics=dict(average_precision=.6)))
    return results


def test_selection_does_not_use_ap_before_recall_or_nonach():
    results = candidates()
    results[1]['precision80'], results[1]['non_ach_precision80'] = selection_point(3500, 100, 144)
    results[2]['precision80'], results[2]['non_ach_precision80'] = selection_point(3500, 100, 145)
    results[1]['metrics']['average_precision'] = .99
    best, _ = compare.choose_best(results)
    assert best['name'] == 'features_pruned'
    results[1]['precision80'], results[1]['non_ach_precision80'] = selection_point(3501, 100, 143)
    assert compare.choose_best(results)[0]['name'] == 'features_only'


def test_selection_ap_tie_break_stable_name_and_reference_fallback():
    results = candidates()
    results[2]['metrics']['average_precision'] = .7
    assert compare.choose_best(results)[0]['name'] == 'features_pruned'
    results[3]['metrics']['average_precision'] = .7
    assert compare.choose_best(results)[0]['name'] == 'features_balanced'
    for r in results[1:]:
        r['non_ach_precision80']['tp'] = 142
    assert compare.choose_best(results)[0]['name'] == 'lr_only'
    results[1]['precision80'] = None
    assert compare.choose_best(results)[0]['name'] == 'lr_only'


def test_selection_requires_exactly_declared_trials():
    with pytest.raises(ValueError):
        compare.choose_best(candidates()[:-1])
    with pytest.raises(ValueError):
        compare.choose_best(candidates()+[candidates()[1]])


def test_support_count_categories_keep_self_missing_distinct_from_zero():
    assert evaluate.support_bucket([-1, 0, 1, 4, 5, 100]).tolist() == ['not_applicable', '0', '1-4', '1-4', '>=5', '>=5']
    with pytest.raises(ValueError):
        evaluate.support_bucket([-2])
    with pytest.raises(ValueError):
        evaluate.support_bucket([1.5])


def test_groups_use_one_global_threshold_and_track_account_coverage():
    y, s = np.array([1, 1, 0, 0]), np.array([.9, .3, .8, .2])
    src, dst = np.array([10, 10, 20, 20]), np.array([30, 40, 30, 40])
    p = dict(threshold=.85)
    groups = evaluate.group_report(y, s, p, {'payment': ['ACH', 'Cash', 'ACH', 'Cash']}, src, dst)
    assert groups['payment']['Cash']['operating_point']['tp'] == 0
    assert groups['payment']['Cash']['operating_point']['threshold'] == .85
    assert groups['payment']['ACH']['detected_positive_accounts']['payer_accounts'] == 1
    assert groups['payment']['ACH']['operating_point']['fp'] == 0
    json.dumps(groups, allow_nan=False)


def test_paired_changes_distinguish_repeated_transactions_from_accounts():
    y = np.array([1, 1, 1, 0, 0])
    a, b = np.array([.1, .1, .9, .1, .9]), np.array([.9, .9, .1, .9, .1])
    src, dst = np.array([7, 7, 8, 9, 9]), np.array([10, 11, 12, 13, 13])
    result = evaluate.paired_changes(y, a, {'threshold': .5}, b, {'threshold': .5}, src, dst)
    assert result['gained_tp'] == dict(transactions=2, payer_accounts=1, receiver_accounts=2, either_accounts=3)
    assert result['lost_tp']['transactions'] == 1
    assert result['new_fp']['transactions'] == 1
    assert result['removed_fp']['transactions'] == 1
    assert evaluate.paired_changes(y, a, None, b, {'threshold': .5}, src, dst)['available'] is False


def test_safe_block_path_rejects_escape(tmp_path):
    with pytest.raises(ValueError):
        evaluate.checked_path(tmp_path, '../frozen')
    assert evaluate.checked_path(tmp_path, 'day/hour/block') == tmp_path/'day/hour/block'


def training_fixture(tmp_path):
    folder = tmp_path/'result/gnn_tuning_v2/trials/features_only'
    folder.mkdir(parents=True)
    architecture = dict(bank_count=4, edge_category_sizes=[4, 4, 4], node_dim=63, r_dim=169, width=64, dropout=.1)
    config = dict(model=architecture)
    (folder/'config.json').write_text(json.dumps(config), encoding='utf-8')
    config_hash = sha(folder/'config.json')
    model = BidirectionalGINE(**architecture)
    torch.save(dict(model=model.state_dict(), config=architecture, config_sha256=config_hash,
                    epoch=1, targets=20318992, positives=17794), folder/'final_model.pt')
    result = dict(status='completed', epoch=1, targets=20318992, positives=17794, blocks=797,
                  config_sha256=config_hash, final_model_sha256=sha(folder/'final_model.pt'))
    (folder/'training_result.json').write_text(json.dumps(result), encoding='utf-8')
    return folder, result


def test_final_model_loads_and_config_hash_or_count_corruption_is_rejected(tmp_path):
    folder, result = training_fixture(tmp_path)
    checked = evaluate.verified_training(tmp_path, 'features_only')
    assert isinstance(checked['model'], BidirectionalGINE)
    result['targets'] -= 1
    (folder/'training_result.json').write_text(json.dumps(result), encoding='utf-8')
    with pytest.raises(ValueError, match='training artifact'):
        evaluate.verified_training(tmp_path, 'features_only')


def test_final_model_nonfinite_weights_are_rejected_even_with_updated_file_hash(tmp_path):
    folder, result = training_fixture(tmp_path)
    saved = torch.load(folder/'final_model.pt', weights_only=True)
    next(iter(saved['model'].values())).view(-1)[0] = float('nan')
    torch.save(saved, folder/'final_model.pt')
    result['final_model_sha256'] = sha(folder/'final_model.pt')
    (folder/'training_result.json').write_text(json.dumps(result), encoding='utf-8')
    with pytest.raises(ValueError, match='Nonfinite'):
        evaluate.verified_training(tmp_path, 'features_only')


@pytest.fixture
def prediction_fixture(tmp_path, monkeypatch):
    # Preserve the production 197-block inventory shape, using one row/block.
    monkeypatch.setattr(compare, 'TOTAL', 197)
    monkeypatch.setattr(compare, 'POSITIVES', 99)
    monkeypatch.setattr(compare, 'TRAIN_TOTAL', 10)
    root = tmp_path/'result/gnn_tuning_v2/evaluations/features_only'
    cache = tmp_path/'data/gnn_validation_cache_parquet'
    cache.mkdir(parents=True)
    for name in ('config.json', 'manifest.json'):
        (cache/name).write_text('{}', encoding='utf-8')
    root.mkdir(parents=True)
    config = dict(dates=compare.DATES, expected_targets=197, expected_positives=99,
                  model_sha256='model', code_sha256={}, cache_config_sha256=sha(cache/'config.json'),
                  cache_manifest_sha256=sha(cache/'manifest.json'))
    (root/'config.json').write_text(json.dumps(config), encoding='utf-8')
    digest = sha(root/'config.json')
    y, s, times = (np.arange(197)%2 == 0).astype(np.int8), np.linspace(0, 1, 197).astype(np.float32), np.arange(197)+100
    blocks = []
    for i in range(197):
        block = f'transaction_date=2022-09-11/hour=00/block={i:04d}'
        folder = root/'predictions'/block
        folder.mkdir(parents=True)
        ids = np.array([i+10], dtype=np.int64)
        pq.write_table(evaluate.prediction_table(ids, times[i:i+1], y[i:i+1], s[i:i+1]), folder/'predictions.parquet')
        manifest = dict(config_sha256=digest, targets=1, target_ids_sha256=evaluate.array_sha(ids),
                        file_sha256=sha(folder/'predictions.parquet'))
        (folder/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        blocks.append(dict(block=block, targets=1, manifest_sha256=sha(folder/'manifest.json')))
    metrics = dict(status='completed', config_sha256=digest, model_sha256='model', blocks=blocks,
                   metrics=compare.binary_metrics(y, s))
    (root/'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
    return tmp_path, root, times, y, s, metrics


def test_prediction_replay_verifies_whole_population_and_does_not_rewrite_files(prediction_fixture):
    project, root, times, y, s, _ = prediction_fixture
    before = sha(root/'metrics.json')
    loaded_y, loaded_s, report = compare.load_predictions(project, 'features_only', times)
    assert np.array_equal(loaded_y, y) and np.array_equal(loaded_s, s)
    assert len(report['blocks']) == 197
    assert sha(root/'metrics.json') == before


def test_prediction_replay_rejects_cutoff_or_reference_label_identity(prediction_fixture):
    project, _, times, y, _, _ = prediction_fixture
    wrong_time = times.copy(); wrong_time[0] += 1
    with pytest.raises(ValueError, match='cutoff'):
        compare.load_predictions(project, 'features_only', wrong_time)
    wrong_label = y.copy(); wrong_label[0] = 1-wrong_label[0]
    with pytest.raises(ValueError, match='labels differ'):
        compare.load_predictions(project, 'features_only', times, wrong_label)


def test_prediction_replay_rejects_committed_file_mutation(prediction_fixture):
    project, root, times, _, _, metrics = prediction_fixture
    path = root/'predictions'/metrics['blocks'][0]['block']/'predictions.parquet'
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='hash/config'):
        compare.load_predictions(project, 'features_only', times)


def test_prediction_replay_rejects_incomplete_or_duplicated_inventory(prediction_fixture):
    project, root, times, _, _, metrics = prediction_fixture
    metrics['blocks'][-1] = copy.deepcopy(metrics['blocks'][0])
    (root/'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
    with pytest.raises(ValueError, match='inventory'):
        compare.load_predictions(project, 'features_only', times)


def test_encoded_count_buckets_match_raw_at_zero_four_five_and_large_values():
    counts = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 10000, 25007998], dtype=np.int64)
    valid = counts >= 0
    encoded = np.log1p(np.where(valid, counts, 0)).astype(np.float32)
    actual = compare.encoded_support_codes(encoded, valid)
    assert actual.dtype == np.int8
    assert actual.tolist() == [-1, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    assert np.array_equal(actual, compare.support_codes(counts))
    # Positive integer classes do not move for adjacent float32 representations.
    for direction in (-np.inf, np.inf):
        rounded = encoded.copy()
        rounded[2:] = np.nextafter(rounded[2:], np.float32(direction))
        assert np.array_equal(compare.encoded_support_codes(rounded, valid), actual)


@pytest.mark.parametrize('encoded,valid', [([np.nan], [1]), ([-.1], [1]), ([1.], [0]), ([1.], [.5])])
def test_encoded_count_buckets_reject_invalid_values_or_masks(encoded, valid):
    with pytest.raises(ValueError, match='encoded support'):
        compare.encoded_support_codes(encoded, valid)


def test_compact_group_codes_have_explicit_human_labels():
    y = np.array([0, 1, 1, 0])
    codes = np.array([-1, 0, 1, 2], dtype=np.int8)
    groups = evaluate.group_report(y, np.array([.1, .8, .9, .2]), {'threshold': .5},
        {'payer_support_72h': codes}, np.arange(4), np.arange(4)+10,
        {'payer_support_72h': compare.SUPPORT_LABELS})
    assert set(groups['payer_support_72h']) == {'not_applicable', '0', '1-4', '>=5'}
    assert groups['payer_support_72h']['1-4']['operating_point']['tp'] == 1


@pytest.fixture
def support24_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, 'TOTAL', 6)
    monkeypatch.setattr(compare, 'TRAIN_TOTAL', 10)
    monkeypatch.setattr(compare, 'DATES', ['2022-09-11'])
    folder = tmp_path/'result/gnn_validation_v1/transaction_date=2022-09-11'
    folder.mkdir(parents=True)
    path = folder/'features.parquet'
    times = np.arange(6, dtype=np.int64)+int(np.datetime64('2022-09-11', 'us').astype(np.int64))
    self_transfer = np.array([False, False, False, False, False, True])
    table = pa.table(dict(edge_id=np.arange(10, 16, dtype=np.int64),
        timestamp=pa.array(times).cast(pa.timestamp('us')),
        r_src_ext_out_count_24h=pa.array([0., 1., 4., 5., 10000., None]),
        r_dst_ext_in_count_24h=pa.array([5., 4., 1., 0., 10000., None])))
    pq.write_table(table, path)
    manifest_path = folder/'merge_phase.json'
    manifest_path.write_text(json.dumps(dict(parts=[dict(path=str(path))])), encoding='utf-8')
    sidecar = dict(input_hashes={str(path.relative_to(tmp_path)): sha(path)})
    cache = dict(hashes={str(manifest_path.relative_to(tmp_path)): sha(manifest_path)})
    return tmp_path, path, times, self_transfer, sidecar, cache, table


def test_support24_complete_identity_join_and_both_roles(support24_fixture):
    project, _, times, self_transfer, sidecar, cache, _ = support24_fixture
    groups = compare._support24(project, times, self_transfer, sidecar, cache)
    assert groups['src_count_24h'].tolist() == [0, 1, 1, 2, 2, -1]
    assert groups['dst_count_24h'].tolist() == [2, 1, 1, 0, 2, -1]


def test_support24_rejects_changed_source_even_when_shape_stays_valid(support24_fixture):
    project, path, times, self_transfer, sidecar, cache, table = support24_fixture
    pq.write_table(table.slice(0, 5), path)
    with pytest.raises(ValueError, match='frozen identity or changed'):
        compare._support24(project, times, self_transfer, sidecar, cache)


@pytest.mark.parametrize('mutation,message', [('missing', 'Incomplete'), ('duplicate', 'duplicated'),
    ('wrong_time', 'timestamp/date'), ('wrong_null', 'applicability'), ('fraction', 'raw historical')])
def test_support24_independent_coverage_semantics_even_with_matching_hash(support24_fixture, mutation, message):
    project, path, times, self_transfer, sidecar, cache, table = support24_fixture
    if mutation == 'missing':
        table = table.slice(0, 5)
    elif mutation == 'duplicate':
        table = table.set_column(0, 'edge_id', pa.array([10, 10, 12, 13, 14, 15], type=pa.int64()))
    elif mutation == 'wrong_time':
        changed = times.copy(); changed[0] += 1
        table = table.set_column(1, 'timestamp', pa.array(changed).cast(pa.timestamp('us')))
    elif mutation == 'wrong_null':
        table = table.set_column(2, 'r_src_ext_out_count_24h', pa.array([0., 1., 4., 5., 10000., 0.]))
    else:
        table = table.set_column(2, 'r_src_ext_out_count_24h', pa.array([0., 1.5, 4., 5., 10000., None]))
    pq.write_table(table, path)
    sidecar['input_hashes'][str(path.relative_to(project))] = sha(path)
    with pytest.raises(ValueError, match=message):
        compare._support24(project, times, self_transfer, sidecar, cache)


@pytest.mark.parametrize('decreasing', [False, True])
def test_three_window_support_assembly_and_monotonic_buckets(support24_fixture, monkeypatch, decreasing):
    from gnn_features import v2_features as features
    project, _, times, self_transfer, sidecar, cache_config, _ = support24_fixture
    cache = project/'data/gnn_validation_cache_parquet'
    cache.mkdir(parents=True)
    (cache/'config.json').write_text(json.dumps(cache_config), encoding='utf-8')
    sidecar['base_config_sha256'] = sha(cache/'config.json')
    block = 'transaction_date=2022-09-11/hour=00/block=0000'
    folder = cache/block
    folder.mkdir(parents=True)
    pq.write_table(pa.table(dict(targets=[[10, 11, 12, 13, 14, 15]])), folder/'plans.parquet')
    count72 = {'src': np.array([0, 2, 4, 6, 10000, -1]), 'dst': np.array([6, 4, 3, 1, 10000, -1])}
    if decreasing:
        count72['dst'][0] = 1
    counts7 = dict(src_count_7d=np.array([0, 3, 5, 8, 10000, -1]),
                   dst_count_7d=np.array([7, 5, 3, 2, 10000, -1]))
    matrix = np.zeros((6, 59), np.float32)
    for role in ('src', 'dst'):
        column = features.EXTRA_NAMES.index(f'v2_{role}_count_72h')
        valid = count72[role] >= 0
        matrix[:, column] = np.log1p(np.where(valid, count72[role], 0))
        matrix[:, column+len(features.EXTRA_NAMES)] = valid
    class FakeStore:
        def __init__(self, *args, **kwargs):
            self.config = sidecar
            self.manifest_sha256 = self.config_sha256 = self.contract_sha256 = 'identity'
        def support(self, *args): return counts7
        def block(self, *args): return matrix
    monkeypatch.setattr(features, 'SidecarStore', FakeStore)
    if decreasing:
        with pytest.raises(ValueError, match='decrease across longer'):
            compare._support(project, [dict(block=block, targets=6)], times, self_transfer)
    else:
        groups, _ = compare._support(project, [dict(block=block, targets=6)], times, self_transfer)
        assert set(groups) == {f'{role}_count_{window}' for role in ('src', 'dst') for window in ('24h', '72h', '7d')}
        assert all(value.dtype == np.int8 for value in groups.values())
        assert groups['dst_count_24h'].tolist() == [2, 1, 1, 0, 2, -1]
        assert groups['dst_count_72h'].tolist() == [2, 1, 1, 1, 2, -1]
        assert groups['dst_count_7d'].tolist() == [2, 2, 1, 1, 2, -1]
