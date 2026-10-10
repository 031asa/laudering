"""Saved real Train minibatches -> forward/backward/update/checkpoint acceptance."""
import argparse
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from .model import BidirectionalGINE,to_torch


def run(batch_dir,archive,out,rounds=20,device='cuda'):
    device=torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use the project GPU environment. No automatic CPU fallback.')
    if out.exists():raise ValueError('Choose a new output directory')
    out.mkdir(parents=True);torch.manual_seed(20260920);torch.set_num_threads(4)
    vocab=json.loads((archive/'metadata/vocabulary.json').read_text(encoding='utf-8'))['values']
    config=dict(bank_count=len(vocab['bank'])+2,edge_category_sizes=[len(vocab['payment_format'])+2,len(vocab['currency'])+2,len(vocab['currency'])+2,7],width=64,dropout=.1)
    data=[];load_start=time.perf_counter()
    for clock in ['0600','1200','1900']:
        with np.load(batch_dir/('batch_'+clock+'.npz'),allow_pickle=False) as f:batch=to_torch({k:f[k] for k in f.files},device=device)
        labels=torch.from_numpy(np.load(batch_dir/('labels_'+clock+'.npy'),allow_pickle=False)).to(device)
        data.append((batch,labels))
    def synchronize():
        if device.type=='cuda':torch.cuda.synchronize(device)
    synchronize();load_seconds=time.perf_counter()-load_start
    count=sum(len(y) for _,y in data);positive=sum(int(y.sum()) for _,y in data)
    if not 0<positive<count:raise ValueError('Acceptance sample needs both classes')
    model=BidirectionalGINE(**config).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((count-positive)/positive,device=device))
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    def evaluate():
        model.eval()
        with torch.no_grad():return float(torch.stack([loss_fn(model(b),y) for b,y in data]).mean())
    initial=evaluate();first_params={k:p.detach().clone() for k,p in model.named_parameters()};step_times=[];losses=[]
    for epoch in range(rounds):
        model.train()
        for batch,labels in data:
            synchronize();start=time.perf_counter();optimizer.zero_grad(set_to_none=True);logits=model(batch);loss=loss_fn(logits,labels)
            if not torch.isfinite(loss):raise AssertionError('Nonfinite loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise AssertionError('Nonfinite gradients')
            torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            synchronize();step_times.append(time.perf_counter()-start);losses.append(float(loss.detach()))
    final=evaluate()
    if final>=initial:raise AssertionError('Fixed-sample loss did not decrease')
    changed=[k for k,p in model.named_parameters() if not torch.equal(p.detach(),first_params[k])]
    prefixes=['bank','node_encoder','history_encoder','target_encoder','r_encoder','in_layers.0','in_layers.1','out_layers.0','out_layers.1','head']
    if not all(any(k.startswith(prefix+'.') for k in changed) for prefix in prefixes):raise AssertionError('A model branch did not update')
    # Numeric paths must all participate in prediction, including historical edges.
    original=data[1][0];probe={k:v.detach().clone() for k,v in original.items()}
    fields=['x_num','edge_attr_num','target_base_num','target_attr']
    for key in fields:probe[key].requires_grad_(True)
    model.zero_grad(set_to_none=True);loss_fn(model(probe),data[1][1]).backward()
    gradients={key:float(probe[key].grad.abs().sum()) for key in fields}
    if not all(np.isfinite(v) and v>0 for v in gradients.values()):raise AssertionError('Input path inactive')
    model.eval()
    with torch.no_grad():before=model(original).clone()
    checkpoint=out/'acceptance_only.pt'
    torch.save(dict(config=config,state_dict=model.state_dict(),seed=20260920,scope='fixed training samples; not a production or validated AML model'),checkpoint)
    restored=torch.load(checkpoint,map_location=device,weights_only=True);clone=BidirectionalGINE(**restored['config']).to(device);clone.load_state_dict(restored['state_dict']);clone.eval()
    with torch.no_grad():
        restored_output=clone(original)
        # CUDA index_add may change accumulation order; compare within float32 tolerance.
        torch.testing.assert_close(restored_output,before,rtol=1e-5,atol=1e-5)
    checkpoint_exact=torch.equal(restored_output,before)
    # Input batch is reused: neural-network timing excludes graph construction.
    inference=[]
    with torch.no_grad():
        for _ in range(20):
            synchronize();start=time.perf_counter();model(original);synchronize();inference.append(time.perf_counter()-start)
    report=dict(status='passed',python=platform.python_version(),torch=torch.__version__,device=str(device),cuda_available=torch.cuda.is_available(),
        cuda_runtime=torch.version.cuda,gpu_name=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        input_load_and_transfer_seconds=load_seconds,
        target_transactions=count,positive_targets=positive,unique_batches=len(data),training_steps=len(step_times),rounds=rounds,
        parameters=sum(p.numel() for p in model.parameters()),config=config,initial_weighted_loss=initial,final_weighted_loss=final,
        fixed_sample_positive_weight=(count-positive)/positive,input_gradient_l1=gradients,updated_parameter_tensors=len(changed),
        mean_training_step_seconds=float(np.mean(step_times)),median_training_step_seconds=float(np.median(step_times)),
        median_forward_seconds=float(np.median(inference)),checkpoint_roundtrip_exact=checkpoint_exact,checkpoint_roundtrip_within_tolerance=True,
        gpu_peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,
        scope='Engineering smoke test on 96 fixed Train targets, repeated optimization; no generalization claim or validation/test access. Timings synchronize CUDA; data already on device; finite-gradient checks included.')
    (out/'acceptance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch-dir',type=Path,default=Path('result/gnn_loader_benchmark_compact_v1'))
    p.add_argument('--archive',type=Path,default=Path('data/gnn_v1_train'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=p.parse_args();run(args.batch_dir,args.archive,args.output,device=args.device)
