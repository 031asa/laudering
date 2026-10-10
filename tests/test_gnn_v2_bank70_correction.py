"""Prevent raw-bank/encoded-key confusion and accidental frozen-report mutation."""
import copy
import numpy as np
import pytest
from correct_gnn_v2_bank70_report import resolve_bank70, assert_only_bank_groups_changed, write_new


def test_raw_bank70_is_resolved_through_permuted_identity_not_literal_key():
    identity = [str(1000+i) for i in range(75)]
    identity[3] = '70'
    code, groups = resolve_bank70(identity, np.array([3, 70, 4]), np.array([0, 1, 2]), np.array([1, 0, 0]))
    assert code == 3
    assert groups['payer_bank70'].tolist() == ['bank70', 'other', 'other']
    assert groups['receiver_bank70'].tolist() == ['other', 'bank70', 'bank70']


@pytest.mark.parametrize('identity,keys,src,dst', [
    (['0'], [0], [0], [0]),
    (['70', '70'], [0], [0], [0]),
    (['70', 70], [0], [0], [0]),
    (['70'], [-1], [0], [0]),
    (['70'], [1], [0], [0]),
    (['70'], [0.], [0], [0]),
    (['70'], [0], [1], [0]),
    (['70'], [0], [0], [0,0]),
])
def test_invalid_identity_is_rejected(identity, keys, src, dst):
    with pytest.raises(ValueError):
        resolve_bank70(identity, np.array(keys), np.array(src), np.array(dst))


def reports():
    original = {'best':{'name':'lr_only'}, 'results':[{'name':'lr_only','metrics':{'ap':.6},
        'groups':{'payer_bank70':{'old':1}, 'receiver_bank70':{'old':2}, 'payment':{'ACH':9}}}]}
    corrected = copy.deepcopy(original)
    corrected['results'][0]['groups']['payer_bank70'] = {'new':3}
    corrected['results'][0]['groups']['receiver_bank70'] = {'new':4}
    corrected['bank70_correction'] = {'sha256':'test'}
    return original, corrected


def test_only_two_groups_and_metadata_may_change():
    original, corrected = reports()
    before = copy.deepcopy(original)
    assert_only_bank_groups_changed(original, corrected)
    assert original == before


@pytest.mark.parametrize('change', ['best','metric','unrelated_group','model_name','removed_group'])
def test_unrelated_changes_are_rejected(change):
    original, corrected = reports()
    row = corrected['results'][0]
    if change == 'best': corrected['best']['name'] = 'other'
    elif change == 'metric': row['metrics']['ap'] = .7
    elif change == 'unrelated_group': row['groups']['payment']['ACH'] = 10
    elif change == 'model_name': row['name'] = 'other'
    else: del row['groups']['payer_bank70']
    with pytest.raises(ValueError): assert_only_bank_groups_changed(original, corrected)


def test_exclusive_publication_preserves_existing_bytes(tmp_path):
    p = tmp_path/'result.json'
    write_new(p, b'original')
    with pytest.raises(FileExistsError): write_new(p, b'changed')
    assert p.read_bytes() == b'original'
