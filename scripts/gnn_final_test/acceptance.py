"""Real CUDA inference checks before publishing Test predictions."""
import torch
from gnn_features.model import BidirectionalGINE, to_torch

def verify_inference(model,batch,expected,model_path,config_hash):
    saved=torch.load(model_path,map_location='cpu',weights_only=True)
    restored=BidirectionalGINE(**(saved.get('config') or saved['model_config']))
    restored.load_state_dict(saved['model'],strict=True)
    restored=restored.to('cuda').eval()
    with torch.inference_mode():
        again=model(to_torch(batch,'cuda'))
        loaded=restored(to_torch(batch,'cuda'))
    torch.testing.assert_close(expected,again,rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(expected,loaded,rtol=1e-6,atol=1e-6)
    if not torch.isfinite(loaded).all(): raise ValueError('Nonfinite inference acceptance')
    return dict(status='passed',config_sha256=config_hash,device='cuda',
        targets=len(batch['target_edge_id']),strict_reload=True,repeat_forward_matches=True,
        max_abs_difference=float((expected-loaded).abs().max()),
        scope='Inference only, no optimizer or training; frozen trained model')
