"""Read-only source provenance and Train-only pattern-list audit.

Raw transaction bytes are hashed, never parsed here. Only Sep 1-10 pattern
transaction lines enter analysis. Labels/pattern membership are audit evidence,
not training features. Existing published results and frozen sources stay intact.
"""
from pathlib import Path
from collections import Counter
import base64
import csv
import hashlib
import json
import re
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'result/amlworld_provenance_20261004'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    if (OUT / 'audit.json').exists():
        raise RuntimeError('Preserve completed provenance audit')
    manifest = json.loads((ROOT / 'data/raw/download_manifest.json').read_text(encoding='utf-8'))
    remote = json.loads((ROOT / 'data/raw/source_integrity.json').read_text(encoding='utf-8'))
    remote = {r['name']: r for r in remote}
    verified = []
    for item in manifest['files']:
        path = ROOT / 'data/raw' / item['name']
        md5, digest = hashlib.md5(), hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda: f.read(8 << 20), b''):
                md5.update(b)
                digest.update(b)
        archived_md5 = next(x[4:] for x in remote[item['name']]['google_storage_hashes'] if x.startswith('md5='))
        assert path.stat().st_size == item['bytes'] == int(remote[item['name']]['stored_content_length'])
        assert digest.hexdigest() == item['sha256']
        assert md5.hexdigest() == item['md5']
        assert base64.b64encode(md5.digest()).decode() == archived_md5
        verified.append(dict(name=item['name'], bytes=path.stat().st_size,
                             sha256=digest.hexdigest(), md5=md5.hexdigest(),
                             matches_archived_official_metadata=True))

    seen = set()
    counts = Counter({'transactions': 0, 'non_ach': 0, 'payer_bank70': 0, 'payer_bank70_non_ach': 0})
    payments, patterns = Counter(), Counter()
    duplicates = 0
    independent = Counter()
    pattern = None
    pattern_path = ROOT / 'data/raw/HI-Medium_Patterns.txt'
    with pattern_path.open(encoding='utf-8-sig') as f:
        for line in f:
            if line.startswith('BEGIN LAUNDERING ATTEMPT - '):
                pattern = line.strip().split(' - ', 1)[1].split(':', 1)[0]
            # Date selection precedes parsing labels/fields of any transaction.
            if not re.match(r'^2022/09/(0[1-9]|10) ', line):
                continue
            independent[line.rstrip().rsplit(',', 2)[-2]] += 1
            row = next(csv.reader([line]))
            assert len(row) == 11 and row[-1] == '1' and pattern is not None
            key = tuple(row)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            counts['transactions'] += 1
            counts['non_ach'] += row[9] != 'ACH'
            counts['payer_bank70'] += int(row[1]) == 70
            counts['payer_bank70_non_ach'] += int(row[1]) == 70 and row[9] != 'ACH'
            payments[row[9]] += 1
            patterns[pattern] += 1
    assert duplicates == 0 and payments == independent

    entities = []
    with (ROOT / 'data/raw/HI-Medium_accounts.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if int(row['Bank ID']) == 70:
                entities.append(row)
    account_summary = dict(accounts=len(entities), distinct_entities=len({r['Entity ID'] for r in entities}),
                           entity_types=dict(Counter(r['Entity Name'].split(' #')[0] for r in entities)))
    prior = ROOT / 'result/gnn_non_ach_matched_differences_20261004/repeated_payment_audit.json'
    repeated = json.loads(prior.read_text(encoding='utf-8'))['results'][1]['periods']
    audit = dict(created_at=datetime.now(timezone.utc).isoformat(), dataset=manifest['dataset'], subset=manifest['subset'],
                 generator='AMLworld', verified_raw_files=verified,
                 official_metadata_verified_at=manifest['verified_at_utc'],
                 remote_metadata_note='Current bytes compared with archived official metadata from initial download; remote files were not downloaded again.',
                 train_pattern_list=dict(counts=counts, payments=payments, patterns=patterns,
                                         duplicate_entries=duplicates, independent_payment_count_check=True),
                 bank70_accounts=account_summary,
                 repeated_payment_evidence=dict(source=str(prior.relative_to(ROOT)), sha256=sha(prior),
                                                all_train_positive=repeated['all_train']['1'],
                                                later_train_positive=repeated['later_sep09_10']['1']),
                 scope='Pattern transaction analysis is Train Sep 1-10 only. All raw files hashed as bytes. No Validation/Test transaction fields evaluated.',
                 interpretation_limits=['Pattern-list absence does not establish a specific integration or propagation cause.',
                                        'Repeated payments can have different full histories.',
                                        'Published benchmark performance does not guarantee bank70 non-ACH separability.',
                                        'Paper discloses data, not generator code; exact per-transaction illicit-fund attribution is not established.'],
                 sources=[dict(url='https://github.com/IBM/AML-Data', supports='Official dataset entry point and Kaggle link'),
                          dict(url='https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml', supports='Official dataset identity'),
                          dict(url='https://arxiv.org/html/2306.16424v3', supports='Sections 3.3-3.4 labels; Table 2 benchmarks; Section 4 graph snapshots; Section 7 generator code not released')],
                 script_sha256=sha(__file__))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
