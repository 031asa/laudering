"""Independent v3 distribution inputs over the frozen, unpruned v2 A model."""
from __future__ import annotations

import copy
import numpy as np
import torch

from .model import BidirectionalGINE
from .training_state import capture_rng, restore_rng
from .v2_adapter import (NODE_NAMES, R_NAMES, adapt_arrays as adapt_v2,
                         make_initial_model as make_v2_initial)

TRIALS = ('amount_distribution', 'amount_time_distribution')
AMOUNT_DIM, EXTRA_DIM, BASE_R_DIM = 52, 108, 169


def check_trial(trial):
    if trial not in TRIALS:
        raise ValueError('Only the two predeclared v3 GNN trials are allowed')
    return trial


def architecture(base, trial):
    check_trial(trial)
    result = copy.deepcopy(base)
    result.update(node_dim=63, r_dim=BASE_R_DIM + (AMOUNT_DIM if trial == TRIALS[0] else EXTRA_DIM))
    return result


def projection_contract(trial):
    from .v2_features import EXTRA_COLUMNS as V2_COLUMNS
    from .v3_features import AMOUNT_COLUMNS, TIME_COLUMNS, contract
    check_trial(trial)
    added = list(AMOUNT_COLUMNS) + (list(TIME_COLUMNS) if trial == TRIALS[1] else [])
    return dict(version=3, trial=trial, node_columns=list(NODE_NAMES),
                r_columns=list(R_NAMES) + list(V2_COLUMNS) + added,
                original_r_dim=BASE_R_DIM, added_dim=len(added),
                distribution_contract=contract(), removed_features=[],
                scope='Extra distribution statistics on target R only; every graph node and history edge unchanged')


def adapt_arrays(arrays, v2_extra, distribution, trial):
    check_trial(trial)
    distribution = np.asarray(distribution)
    count = len(arrays['target_edge_id'])
    if (distribution.shape != (count, EXTRA_DIM) or distribution.dtype != np.float32
            or not np.isfinite(distribution).all()):
        raise ValueError('V3 sidecar shape/dtype/finite mismatch')
    result = adapt_v2(arrays, v2_extra, 'features_only')
    selected = distribution[:, :AMOUNT_DIM] if trial == TRIALS[0] else distribution
    result['target_attr'] = np.concatenate((result['target_attr'], selected), axis=1)
    return result


def expand_initial_state(a_state, r_dim):
    if r_dim not in (221, 277) or a_state['r_encoder.0.weight'].shape[1] != BASE_R_DIM:
        raise ValueError('Only original A 169-column initialization can be expanded')
    state = {name: value.detach().clone() for name, value in a_state.items()}
    original = a_state['r_encoder.0.weight']
    expanded = original.new_zeros((original.shape[0], r_dim))
    expanded[:, :BASE_R_DIM] = original
    state['r_encoder.0.weight'] = expanded
    return state


def make_initial_model(config, device='cpu'):
    """Copy fresh A, zero new columns, preserve A's post-construction RNG exactly."""
    trial = check_trial(config['trial'])
    a = make_v2_initial(dict(config, trial='features_only'), device)
    rng = capture_rng()
    model = BidirectionalGINE(**architecture(config['model'], trial)).to(device)
    model.load_state_dict(expand_initial_state(a.state_dict(), architecture(config['model'], trial)['r_dim']), strict=True)
    restore_rng(rng)
    return model
