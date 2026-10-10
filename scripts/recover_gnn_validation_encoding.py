"""One reviewed configuration revision after the uncommitted bank-encoding failure.

This is deliberately restricted to the recorded v1 failure and one source file.
It preserves original configurations/source, checks every committed output, and
publishes a new revision without weakening normal configuration/hash checks.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import psutil

from gnn_features.cache_train import sha
from gnn_features.training_state import ProcessLock, atomic_json

ROOT = Path(__file__).resolve().parents[1]
OLD_CODE = '61852df756df70c3044ff6d8f44e3bfd5244f328254f187d3e06511c57976c72'
OLD_ARCHIVE_CONFIG = '445b44456f3a2820ada007fdca665cf3c402a5cb8912e5f812d2af4abefecd18'
OLD_PIPELINE_CONFIG = '85dfcb1158bdc49c10f31a54baf4bb37beb7e0d7b82ed96e78db3db99b65dbcc'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def run(new_code_sha256, apply=False):
    pipeline = ROOT / 'result/gnn_tuning_v1'
    archive = ROOT / 'data/gnn_v1_validation'
    audit = pipeline / 'recovery/20260928_bank_encoding'
    source = ROOT / 'scripts/gnn_features/validation_archive.py'
    if sha(source) != new_code_sha256 or new_code_sha256 == OLD_CODE:
        raise ValueError('Expected explicitly reviewed new source bytes')
    with ProcessLock(pipeline), ProcessLock(archive):
        for state_path in (pipeline / 'status.json', archive / 'status.json'):
            state = read(state_path)
            if state['status'] != 'failed':
                raise ValueError('Only the stopped failed preparation can be migrated')
            if psutil.pid_exists(state['pid']) or (state.get('child_pid') and psutil.pid_exists(state['child_pid'])):
                raise ValueError('Recorded preparation process is still alive')
        if read(archive / 'status.json').get('error') != "ValueError('Train bank identity prefix changed')":
            raise ValueError('Failure differs from reviewed encoding issue')
        expected_backups = {'validation_archive.before.py': OLD_CODE,
                            'archive_config.before.json': OLD_ARCHIVE_CONFIG,
                            'pipeline_config.before.json': OLD_PIPELINE_CONFIG}
        for name, digest in expected_backups.items():
            if sha(audit / name) != digest:
                raise ValueError('Original immutable snapshot differs: ' + name)
        stages = read(archive / 'stages.json')
        if stages != read(audit / 'stages.before.json') or set(stages['phases']) != {'prefix_parquet'} or stages['appended_files']:
            raise ValueError('Additional committed work exists; impact must be reviewed anew')
        if (archive / 'build.json').exists():
            raise ValueError('Complete archive cannot use this migration')
        for path in (ROOT / 'result/gnn_validation_v1', ROOT / 'data/gnn_validation_cache_parquet', pipeline / 'evaluations'):
            if path.exists() and any(path.iterdir()):
                raise ValueError('Downstream Validation outputs exist; migration is not safe')
        old_archive = read(audit / 'archive_config.before.json')
        old_pipeline = read(audit / 'pipeline_config.before.json')
        if old_archive['code_hashes']['validation_archive.py'] != OLD_CODE or old_pipeline['code_sha256']['validation_archive.py'] != OLD_CODE:
            raise ValueError('Original configuration/source provenance differs')
        for rel, digest in old_archive['input_hashes'].items():
            path = (ROOT / rel).resolve()
            if not path.is_relative_to(ROOT) or sha(path) != digest:
                raise ValueError('Frozen archive source changed: ' + rel)
        for name, digest in old_pipeline['code_sha256'].items():
            if name != 'validation_archive.py' and sha(ROOT / 'scripts/gnn_features' / name) != digest:
                raise ValueError('Unreviewed additional code change: ' + name)
        committed = stages['phases']['prefix_parquet']['outputs']
        for rel, digest in committed.items():
            path = (archive / rel).resolve()
            if not path.is_relative_to(archive) or sha(path) != digest:
                raise ValueError('Committed Train-prefix output changed: ' + rel)
        trials = {}
        for name in ('lr_only', 'sqrt_weight'):
            folder = pipeline / 'trials' / name
            if any((folder / p).exists() for p in ('checkpoint_latest.pt', 'checkpoint_previous.pt', 'final_model.pt')):
                raise ValueError('Production candidate has already trained')
            accepted = read(folder / 'acceptance.json')
            if accepted['status'] != 'passed' or accepted['config_sha256'] != sha(folder / 'config.json'):
                raise ValueError('Existing Train-only GPU acceptance changed')
            trials[name] = dict(config_sha256=sha(folder / 'config.json'), acceptance_sha256=sha(folder / 'acceptance.json'))
        baseline = ROOT / 'result/gnn_train_gpu_v1'
        result = read(baseline / 'training_result.json')
        if result['config_sha256'] != sha(baseline / 'config.json') or result['final_model_sha256'] != sha(baseline / 'final_model.pt'):
            raise ValueError('Baseline identity changed')
        new_archive, new_pipeline = copy.deepcopy(old_archive), copy.deepcopy(old_pipeline)
        new_archive['code_hashes']['validation_archive.py'] = new_code_sha256
        new_pipeline['code_sha256']['validation_archive.py'] = new_code_sha256
        for live, before, after in ((archive / 'config.json', old_archive, new_archive),
                                    (pipeline / 'config.json', old_pipeline, new_pipeline)):
            if read(live) not in (before, after):
                raise ValueError('Live configuration has an unrelated change')
        review = dict(status='reviewed', timestamp=datetime.now(timezone.utc).isoformat(),
                      issue='Arrow 21 fill_null misplaced sparse-null replacement; original Train data is consistent',
                      change='Preserve original Train bank_cat; map new categories using explicit NumPy validity masks',
                      old_code_sha256=OLD_CODE, new_code_sha256=new_code_sha256,
                      old_archive_config_sha256=OLD_ARCHIVE_CONFIG, old_pipeline_config_sha256=OLD_PIPELINE_CONFIG,
                      preserved_phase='prefix_parquet', committed_files_checked=len(committed),
                      affected_completed_results='None: identities, append, R, node caches, evaluation and production trials were not committed',
                      preserved_trials=trials, baseline_final_model_sha256=result['final_model_sha256'],
                      recovery='Replay only the uncommitted identities stage; all original configuration checks remain enabled')
        if not apply:
            print(json.dumps(review))
            return review
        # Store immutable before/after revision records before changing the live pointer.
        for name, value in [('archive_config.revision2.json', new_archive), ('pipeline_config.revision2.json', new_pipeline)]:
            path = audit / name
            if path.exists() and read(path) != value:
                raise ValueError('Recovery revision record already differs')
            if not path.exists():
                atomic_json(path, value)
        fixed_copy = audit / 'validation_archive.revision2.py'
        if fixed_copy.exists() and sha(fixed_copy) != new_code_sha256:
            raise ValueError('Recovery source record differs')
        if not fixed_copy.exists():
            fixed_copy.write_bytes(source.read_bytes())
        atomic_json(audit / 'impact_review.json', review)
        for live, revision in ((archive / 'config.json', audit / 'archive_config.revision2.json'),
                               (pipeline / 'config.json', audit / 'pipeline_config.revision2.json')):
            pending = live.with_name('config.migration.pending')
            pending.write_bytes(revision.read_bytes())
            os.replace(pending, live)
        review.update(status='applied', new_archive_config_sha256=sha(archive / 'config.json'),
                      new_pipeline_config_sha256=sha(pipeline / 'config.json'))
        atomic_json(audit / 'impact_review.json', review)
        print(json.dumps(review))
        return review


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--new-code-sha256', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    run(args.new_code_sha256, args.apply)
