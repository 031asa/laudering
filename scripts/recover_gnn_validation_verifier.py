"""Audited, one-time pipeline revision for the recorded Arrow verifier crash.

No existing feature, archive, model or verification output is changed here.
The new Validation verifier must still pass every existing check on resume.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import psutil

from gnn_features.cache_train import sha
from gnn_features.training_state import ProcessLock, atomic_json

ROOT = Path(__file__).resolve().parents[1]
OLD_PIPELINE = '83080628f8df2e898d7b1d3e2b6cececc45d7b88250492da56b60ea6a724ff07'
OLD_CONFIG = '5795159729ceb0d349d9645d0afb430bbfe6488ff7a79fa44838c0a9ba43f448'
OLD_VERIFIER = 'a428751ff9c1afaf9f6c540f7fb83a043568577a56d2cd7bd35d46689a903d12'
PRESERVED = {
    'data/gnn_v1_validation/build.json': '31a8cdafd2a8ec14cf90e5f8f9002f84a638f4fb19ff0ad7abb9babb352286d0',
    'data/gnn_v1_validation/config.json': 'a78eed216d98021cac18a2b00e7a4aae78a1d9aa5204b74ad107357171b70a1f',
    'data/gnn_v1_validation/stages.json': '92d2aef22b3724b530e7b5f22394eeb1eed6c1343444e703f5b3b38381aa5044',
    'result/gnn_validation_v1/transaction_date=2022-09-11/daily_run.json': '9b3c613d3889325bf830ca8c335aaac928c935ddf075d403717e58ef5a8ec54c',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def run(pipeline_sha, verifier_sha, apply=False):
    out = ROOT / 'result/gnn_tuning_v1'
    audit = out / 'recovery/20260928_validation_verifier'
    code = ROOT / 'scripts/gnn_features'
    day = ROOT / 'result/gnn_validation_v1/transaction_date=2022-09-11'
    with ProcessLock(out):
        state = read(out / 'status.json')
        if state['status'] != 'failed' or state['stage'] != 'verify_validation_features' or state['last_exit_code'] != 3221225477:
            raise ValueError('This revision is only for the recorded stopped verification crash')
        for process_id in [state['pid'], state.get('child_pid'), 126100, 93536]:
            if process_id and psutil.pid_exists(process_id):
                raise ValueError('Recorded pipeline or diagnostic process remains alive')
        for name, digest in [('pipeline_config.before.json', OLD_CONFIG),
                             ('tune_pipeline.before.py', OLD_PIPELINE), ('verify_daily.before.py', OLD_VERIFIER)]:
            if sha(audit / name) != digest:
                raise ValueError('Original snapshot changed: ' + name)
        for rel, digest in PRESERVED.items():
            if sha(ROOT / rel) != digest:
                raise ValueError('Preserved stage changed; review again: ' + rel)
        if read(day / 'daily_run.json')['status'] != 'computed_pending_verification' or (day / 'verification.json').exists():
            raise ValueError('Verification has already changed stage')
        for relative in ['result/gnn_validation_v1/transaction_date=2022-09-12',
                         'result/gnn_validation_v1/transaction_date=2022-09-13',
                         'data/gnn_validation_cache_parquet', 'result/gnn_tuning_v1/evaluations']:
            folder = ROOT / relative
            if folder.exists() and any(folder.iterdir()):
                raise ValueError('Later stages exist; review again')
        original = read(audit / 'pipeline_config.before.json')
        for name, digest in original['code_sha256'].items():
            if name != 'tune_pipeline.py' and sha(code / name) != digest:
                raise ValueError('Unreviewed code change: ' + name)
        if sha(code / 'verify_daily.py') != OLD_VERIFIER:
            raise ValueError('Original Train verifier must remain unchanged')
        if sha(code / 'tune_pipeline.py') != pipeline_sha or sha(code / 'verify_validation.py') != verifier_sha:
            raise ValueError('New source differs from reviewed bytes')
        if pipeline_sha == OLD_PIPELINE:
            raise ValueError('New pipeline entry is required')
        trials = {}
        for name in ['lr_only', 'sqrt_weight']:
            folder = out / 'trials' / name
            if any((folder / n).exists() for n in ['checkpoint_latest.pt', 'checkpoint_previous.pt', 'final_model.pt']):
                raise ValueError('Production trial already exists')
            accepted = read(folder / 'acceptance.json')
            if accepted['status'] != 'passed' or accepted['config_sha256'] != sha(folder / 'config.json'):
                raise ValueError('Train-only candidate acceptance changed')
            trials[name] = sha(folder / 'config.json')
        baseline = ROOT / 'result/gnn_train_gpu_v1'
        for name, field in [('config.json', 'baseline_config_sha256'), ('final_model.pt', 'baseline_model_sha256')]:
            if sha(baseline / name) != original[field]:
                raise ValueError('Baseline identity changed')
        revised = copy.deepcopy(original)
        revised['code_sha256']['tune_pipeline.py'] = pipeline_sha
        revised['code_sha256']['verify_validation.py'] = verifier_sha
        if read(out / 'config.json') not in [original, revised]:
            raise ValueError('Unrelated live configuration change')
        report = dict(status='reviewed', timestamp=datetime.now(timezone.utc).isoformat(),
                      issue='Arrow fill_null access violation confirmed by faulthandler at original verify_daily.py:64',
                      equivalence='NULL remains permitted; every non-null NaN or infinity is rejected using a NumPy validity mask; all other integrity/reference/audit/label checks retained',
                      old_pipeline_config_sha256=OLD_CONFIG, old_pipeline_source_sha256=OLD_PIPELINE,
                      original_verifier_sha256=OLD_VERIFIER, new_pipeline_sha256=pipeline_sha,
                      new_verifier_sha256=verifier_sha, preserved_stage_hashes=PRESERVED,
                      preserved_candidate_config_sha256=trials,
                      completed_results_impact='None: complete archive and computed day remain unchanged; no Validation verification was published',
                      resume_action='Original pipeline entry reruns complete September 11 verification before any later stage')
        if apply:
            revision = audit / 'pipeline_config.revision3.json'
            if revision.exists() and read(revision) != revised:
                raise ValueError('Revision record differs')
            if not revision.exists():
                atomic_json(revision, revised)
            for source, name, digest in [(code / 'tune_pipeline.py', 'tune_pipeline.revision3.py', pipeline_sha),
                                          (code / 'verify_validation.py', 'verify_validation.revision3.py', verifier_sha)]:
                target = audit / name
                if target.exists() and sha(target) != digest:
                    raise ValueError('New source snapshot differs')
                if not target.exists():
                    target.write_bytes(source.read_bytes())
            atomic_json(audit / 'impact_review.json', report)
            pending = out / 'config.migration.pending'
            pending.write_bytes(revision.read_bytes())
            os.replace(pending, out / 'config.json')
            report.update(status='applied', new_pipeline_config_sha256=sha(out / 'config.json'))
            atomic_json(audit / 'impact_review.json', report)
        print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline-sha256', required=True)
    parser.add_argument('--verifier-sha256', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    run(args.pipeline_sha256, args.verifier_sha256, args.apply)
