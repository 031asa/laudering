"""Only three causal structures added to the unpruned, freshly initialized A."""
import copy
import numpy as np
from .model import BidirectionalGINE
from .training_state import capture_rng, restore_rng
from .v2_adapter import NODE_NAMES, R_NAMES, adapt_arrays as adapt_a, make_initial_model as make_a

TRIALS = ('structure_three',)
BASE_R_DIM, EXTRA_DIM = 169, 6


def check_trial(trial):
    if trial not in TRIALS: raise ValueError('Only structure_three is authorized')
    return trial


def architecture(base, trial):
    check_trial(trial)
    result=copy.deepcopy(base); result.update(node_dim=63,r_dim=175)
    return result


def projection_contract(trial):
    from .v2_features import EXTRA_COLUMNS as BASE_COLUMNS
    from .v4_features import EXTRA_COLUMNS, contract
    check_trial(trial)
    return dict(version=4,trial=trial,node_columns=list(NODE_NAMES),
        r_columns=list(R_NAMES)+list(BASE_COLUMNS)+list(EXTRA_COLUMNS),
        original_r_dim=169,added_dim=6,structure_contract=contract(),removed_features=[],
        scope='Only target R appended; all graph nodes and sampled history unchanged')


def adapt_arrays(arrays, v2_extra, structure, trial):
    check_trial(trial)
    values=np.asarray(structure)
    if (values.shape != (len(arrays['target_edge_id']),6) or values.dtype != np.float32
            or not np.isfinite(values).all() or np.any(values[:,:3]<0)
            or not np.isin(values[:,3:],[0,1]).all() or np.any(values[:,:3][values[:,3:]==0]!=0)):
        raise ValueError('Structure input shape/dtype/mask mismatch')
    result=adapt_a(arrays,v2_extra,'features_only')
    result['target_attr']=np.concatenate((result['target_attr'],values),axis=1)
    return result


def expand_initial_state(a_state,r_dim=175):
    if r_dim!=175 or a_state['r_encoder.0.weight'].shape[1]!=169:
        raise ValueError('Only untrained A169 -> R175 is allowed')
    state={name:value.detach().clone() for name,value in a_state.items()}
    old=a_state['r_encoder.0.weight']; new=old.new_zeros((old.shape[0],175));new[:,:169]=old
    state['r_encoder.0.weight']=new
    return state


def make_initial_model(config,device='cpu'):
    trial=check_trial(config['trial'])
    a=make_a(dict(config,trial='features_only'),device)
    rng=capture_rng()
    model=BidirectionalGINE(**architecture(config['model'],trial)).to(device)
    model.load_state_dict(expand_initial_state(a.state_dict()),strict=True)
    restore_rng(rng)
    return model
