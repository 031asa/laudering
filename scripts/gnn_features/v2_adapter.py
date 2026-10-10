"""Versioned v2 projections over immutable v1 minibatches."""
from __future__ import annotations

import copy
import random
import numpy as np
import torch

from .contract import N_FIELDS, R_FIELDS
from .model import BidirectionalGINE

TRIALS = ('features_only', 'features_pruned', 'features_balanced')
EXTRA_DIM = 59
SEED = 20260920
REMOVED_N = tuple(f'n_{direction}_ach_{kind}_24h'
                  for direction in ('in', 'out') for kind in ('count', 'share'))
REMOVED_R = ('r_src_own_ext_ach_amount_share_24h', 'r_dst_own_ext_ach_amount_share_24h')


def encoded_names(fields, coverage):
    return ([f['name'] + '__encoded' for f in fields]
            + [f['name'] + '__valid' for f in fields if f['nullable']] + list(coverage))


NODE_NAMES = encoded_names(N_FIELDS, ('full_1h', 'full_24h', 'full_72h'))
R_NAMES = encoded_names(R_FIELDS, ('full_6h', 'full_day'))


def _keep(names, removed):
    expected = {name + suffix for name in removed for suffix in ('__encoded', '__valid')}
    return [i for i, name in enumerate(names) if name not in expected]


NODE_KEEP = _keep(NODE_NAMES, REMOVED_N)
R_KEEP = _keep(R_NAMES, REMOVED_R)
assert len(NODE_NAMES) == 63 and len(R_NAMES) == 110
assert len(NODE_KEEP) == 57 and len(R_KEEP) == 106


def check_trial(trial):
    if trial not in TRIALS:
        raise ValueError('Only the three frozen v2 trials are allowed')
    return trial


def projection_contract(trial):
    check_trial(trial)
    pruning = trial != 'features_only'
    return dict(version=1, trial=trial, extra_dim=EXTRA_DIM,
                node_columns=[NODE_NAMES[i] for i in (NODE_KEEP if pruning else range(63))],
                original_r_columns=[R_NAMES[i] for i in (R_KEEP if pruning else range(110))],
                node_indices=NODE_KEEP if pruning else list(range(63)),
                r_indices=R_KEEP if pruning else list(range(110)),
                removed_node_features=list(REMOVED_N) if pruning else [],
                removed_r_features=list(REMOVED_R) if pruning else [],
                removal_scope='Every sampled node; target R; nullable masks removed with values',
                payment_category_preserved=True)


def adapt_arrays(arrays, extra, trial):
    """Never mutate cached arrays. Remove ACH columns from every graph node."""
    check_trial(trial)
    extra = np.asarray(extra)
    if (arrays['x_num'].ndim != 2 or arrays['x_num'].shape[1] != 63
            or arrays['target_attr'].shape != (len(arrays['target_edge_id']), 110)
            or extra.shape != (len(arrays['target_edge_id']), EXTRA_DIM)
            or extra.dtype != np.float32 or not np.isfinite(extra).all()):
        raise ValueError('V1 input or v2 sidecar dimension/dtype/finite mismatch')
    result = dict(arrays)
    prune = trial != 'features_only'
    result['x_num'] = arrays['x_num'][:, NODE_KEEP].copy() if prune else arrays['x_num']
    original = arrays['target_attr'][:, R_KEEP] if prune else arrays['target_attr']
    result['target_attr'] = np.concatenate([original, extra], axis=1)
    return result


def architecture(base, trial):
    check_trial(trial)
    config = copy.deepcopy(base)
    config.update(node_dim=63 if trial == 'features_only' else 57,
                  r_dim=169 if trial == 'features_only' else 165)
    return config


def project_initial_state(full_state):
    """Preserve bank embedding offset and all shared parameter values exactly."""
    state = {name: value.detach().clone() for name, value in full_state.items()}
    state['node_encoder.0.weight'] = full_state['node_encoder.0.weight'][:,
        NODE_KEEP + list(range(63, 79))].detach().clone()
    state['r_encoder.0.weight'] = full_state['r_encoder.0.weight'][:,
        R_KEEP + list(range(110, 169))].detach().clone()
    return state


def make_initial_model(config, device='cpu'):
    """A is fresh seeded; B/C are column projections of that same initialization.

    Restore CPU RNG after constructing the smaller module, so dropout starts
    from the same state in all trials. CUDA RNG is also seeded identically.
    """
    trial = check_trial(config['trial'])
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.device(device).type == 'cuda':
        torch.cuda.manual_seed_all(SEED)
    full = BidirectionalGINE(**architecture(config['model'], 'features_only'))
    after_full = torch.get_rng_state().clone()
    if trial == 'features_only':
        model = full
    else:
        model = BidirectionalGINE(**architecture(config['model'], trial))
        model.load_state_dict(project_initial_state(full.state_dict()), strict=True)
        del full
    torch.set_rng_state(after_full)
    return model.to(device)
