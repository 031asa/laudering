"""Reject resume states that would omit or replay committed targets."""
import pytest

from gnn_features.train_model import validate_cursor


def checkpoint():
    return dict(config_sha256='frozen', next_block=1,
                metrics=dict(targets=64, positives=2, weighted_loss_sum=10.,
                             blocks=[dict(block='day/block0')]))


BLOCKS = [dict(block='day/block0', targets=64, positives=2),
          dict(block='day/block1', targets=37, positives=0)]


def test_resume_prefix():
    assert validate_cursor(checkpoint(), BLOCKS, 'frozen') == 1


@pytest.mark.parametrize('field,value', [('targets', 63), ('positives', 1),
                                        ('weighted_loss_sum', float('nan'))])
def test_reject_incomplete_prefix(field, value):
    payload = checkpoint()
    payload['metrics'][field] = value
    with pytest.raises(ValueError):
        validate_cursor(payload, BLOCKS, 'frozen')


def test_reject_stale_or_skipped_cursor():
    payload = checkpoint()
    with pytest.raises(ValueError):
        validate_cursor(payload, BLOCKS, 'changed')
    payload['next_block'] = 2
    with pytest.raises(ValueError):
        validate_cursor(payload, BLOCKS, 'frozen')
    payload = checkpoint()
    payload['metrics']['blocks'][0]['block'] = 'other'
    with pytest.raises(ValueError):
        validate_cursor(payload, BLOCKS, 'frozen')
