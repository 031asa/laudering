"""Bounded real-cache input, CUDA update, and checkpoint-resume acceptance."""
import gc
from pathlib import Path
import time

import numpy as np
import torch

from .loader import TemporalLoader, EdgeStore
from .model import to_torch
from .model_cache import ParquetBlockCache
from .training_state import (atomic_json, capture_rng, restore_rng,
                             save_checkpoint, load_checkpoint)


def accept(output, config, config_hash, verified, progress):
    from .train_model import (ROOT, ARCHIVE, FEATURES, CACHE, EDGE_CACHE, require_cuda,
                             seed_all, make_model, train_step, finite_parameters)
    device = require_cuda()
    began = time.perf_counter()
    candidates = [b for b in verified['blocks']
                  if b['block'].startswith('transaction_date=2022-09-04/') and b['positives'] > 0]
    selected_block = next((b for b in candidates if '/hour=12/' in b['block']), candidates[0])
    loader = TemporalLoader(ARCHIVE, FEATURES, '2022-09-04', edge_cache=EDGE_CACHE)
    cache = ParquetBlockCache(CACHE / selected_block['block'], loader, verified['config_sha256'])
    selected = {0, len(cache) // 2, len(cache) - 1}
    for i, plan in enumerate(cache.plans):
        p = loader.day.positions(plan['targets'])
        if loader.day.labels[p].sum() > 0:
            selected.add(i)
            break
    data, comparison, raw_edges = [], [], EdgeStore(ARCHIVE)
    for i in sorted(selected):
        loader.builder.clear_cache()
        arrays, y, _ = cache.batch(i)
        loader.builder.clear_cache()
        direct, z, _ = loader.batch(cache.plans[i]['targets'])
        if set(arrays) != set(direct):
            raise ValueError('Cached input field mismatch')
        for name in arrays:
            if name == 'x_num':
                np.testing.assert_allclose(arrays[name], direct[name], rtol=1e-6, atol=1e-6)
            else:
                np.testing.assert_array_equal(arrays[name], direct[name])
        np.testing.assert_array_equal(y, z)
        # Existing compact E inputs must also match persisted Parquet E values.
        for ids, num_name, cat_name in [
                (arrays['edge_id'], 'edge_attr_num', 'edge_attr_cat'),
                (arrays['target_edge_id'], 'target_base_num', 'target_base_cat')]:
            expected_num, expected_cat = raw_edges.get(ids)
            np.testing.assert_array_equal(arrays[num_name][:, :8], expected_num)
            np.testing.assert_array_equal(arrays[cat_name], expected_cat)
        data.append((arrays, y))
        comparison.append(dict(batch=i, targets=len(y), positives=int(y.sum()),
                               nodes=len(arrays['node_id']), history_edges=len(arrays['edge_id'])))
        progress(stage='acceptance_inputs', compared_batches=len(data))
    if len(data) < 2 or sum(int(y.sum()) for _, y in data) == 0:
        raise ValueError('Real acceptance requires multiple batches and a positive target')
    del raw_edges, direct, loader, cache
    gc.collect()
    seed_all()
    model, optimizer, loss_fn = make_model(config, device)
    initial = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    loss0, norm0 = train_step(model, optimizer, loss_fn, *data[0], device)
    finite_parameters(model)
    scratch = Path(output) / 'acceptance_checkpoints'
    scratch.mkdir(exist_ok=True)
    if any(scratch.iterdir()):
        raise ValueError('Unreviewed acceptance checkpoint scratch exists')
    payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                   rng=capture_rng(), config_sha256=config_hash, next_block=1,
                   targets=len(data[0][1]))
    save_checkpoint(scratch, payload)
    loss_continued, norm1 = train_step(model, optimizer, loss_fn, *data[1], device)
    expected = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    restored = load_checkpoint(scratch, map_location='cpu')
    if restored['config_sha256'] != config_hash or restored['next_block'] != 1:
        raise ValueError('Acceptance resume cursor/config mismatch')
    resumed, resumed_opt, resumed_loss = make_model(config, device)
    resumed.load_state_dict(restored['model'])
    resumed_opt.load_state_dict(restored['optimizer'])
    restore_rng(restored['rng'])
    rng = capture_rng()
    if not torch.equal(rng['torch_cpu'], restored['rng']['torch_cpu']):
        raise ValueError('CPU RNG was not restored')
    # Input and optimizer restoration are verified against a continuous next update.
    loss_resumed, norm2 = train_step(resumed, resumed_opt, resumed_loss, *data[1], device)
    np.testing.assert_allclose(loss_resumed, loss_continued, rtol=1e-4, atol=1e-5)
    for name, value in resumed.state_dict().items():
        torch.testing.assert_close(value.detach().cpu(), expected[name], rtol=1e-4, atol=1e-5)
    finite_parameters(resumed)
    changed = [name for name, p in resumed.named_parameters()
               if not torch.equal(p.detach().cpu(), initial[name])]
    required = ['bank', 'node_encoder', 'history_encoder', 'target_encoder', 'r_encoder',
                'in_layers.0', 'in_layers.1', 'out_layers.0', 'out_layers.1', 'head']
    if not all(any(n.startswith(prefix + '.') for n in changed) for prefix in required):
        raise ValueError('At least one model branch did not update')
    probe_arrays, probe_y = data[-1]
    probe = to_torch(probe_arrays, device=device)
    fields = ['x_num', 'edge_attr_num', 'target_base_num', 'target_attr']
    for field in fields:
        probe[field].requires_grad_(True)
    resumed.zero_grad(set_to_none=True)
    resumed_loss(resumed(probe), torch.as_tensor(probe_y.copy(), device=device)).backward()
    gradients = {field: float(probe[field].grad.abs().sum()) for field in fields}
    if not all(np.isfinite(x) and x > 0 for x in gradients.values()):
        raise ValueError('An N/E/R input gradient path is inactive')
    # A second saved cursor verifies that the two-generation rotation remains readable.
    save_checkpoint(scratch, dict(model=resumed.state_dict(), optimizer=resumed_opt.state_dict(),
                                 rng=capture_rng(), config_sha256=config_hash, next_block=2,
                                 targets=sum(len(y) for _, y in data[:2])))
    if load_checkpoint(scratch)['next_block'] != 2:
        raise ValueError('Checkpoint rotation failed')
    report = dict(status='passed', config_sha256=config_hash,
                  block=selected_block['block'], input_comparisons=comparison,
                  gpu=torch.cuda.get_device_name(), positive_weight=config['positive_weight'],
                  first_loss=loss0, continuous_next_loss=loss_continued, resumed_next_loss=loss_resumed,
                  gradient_norms=[norm0, norm1, norm2], updated_parameter_tensors=len(changed),
                  input_gradient_l1=gradients, cached_inputs_match_direct=True,
                  compact_E_matches_Parquet=True, resume_update_within_float32_tolerance=True,
                  acceptance_seconds=time.perf_counter() - began,
                  scope='Engineering acceptance on real Train batches, not the production epoch.')
    # Delete only the two checkpoint files created for this bounded acceptance.
    if scratch.resolve().parent != Path(output).resolve() or not scratch.resolve().is_relative_to(ROOT):
        raise ValueError('Unsafe acceptance scratch cleanup')
    for name in ['checkpoint_latest.pt', 'checkpoint_previous.pt']:
        path = scratch / name
        if path.exists():
            path.unlink()
    scratch.rmdir()
    atomic_json(Path(output) / 'acceptance.json', report)
    return report
