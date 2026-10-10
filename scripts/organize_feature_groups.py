"""Publish a design-only feature family registry for GNN feature research."""
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
doc=(ROOT/'反洗钱候选特征矩阵.md').read_text(encoding='utf-8')
found=re.findall(r'^\| ((?:F|B)\d{2}) \| ([^|]+) \|',doc,re.M)
assert len(found)==len(dict(found))==63
assert {i for i,_ in found}=={f'F{i:02}' for i in range(1,56)}|{f'B{i:02}' for i in range(1,9)}
graph={f'F{i:02}' for i in [13,14,15,39,40,41,42,43,44,47,48]}
conditional={f'F{i:02}' for i in [16,38,45,46,47,48]}
core={f'F{i:02}' for i in [1,2,3,4,5,6,7,8,11,12,17,22,25,27,35,37,49,51,52,53]}|{f'B{i:02}' for i in range(1,9)}
extra={'F15','F39','F40','F44'}
assert core.isdisjoint(graph) and extra<=graph
assert (core|extra).isdisjoint(conditional)
families=[]
for family,name in sorted(found):
    families.append(dict(id=family,name=name.strip(),group='G' if family in graph else 'T',
                         planned_in_A=family in core,planned_in_B=family in core|extra,
                         requires_historical_entity_mapping=family in conditional,
                         status='planned' if family in core|extra else 'candidate_deferred'))
registry=dict(version='1.2',updated='2026-09-15',status='design_only_not_computed_not_trained',
              objective='Candidate transaction and graph relationship features for GNN research',
              groups={'T':'Shared transaction, account/pair behavior and bank preference; includes simple one-hop statistics',
                      'G':'Additional network patterns and multi-account structure'},
              graph_initial_scope=dict(history_hours=24,cycle_total_edges=3,path_max_total_edges_including_current=3,
                                       common_neighbors='historical shared payers and payees',
                                       timestamp_rule='history timestamp < target timestamp; path timestamps strictly increasing',
                                       high_degree_policy='profile cost on train; if bounded search is needed freeze label-independent bounds before evaluation and report truncation'),
              excludes=['labels_as_features','full_train_bank_label_rates','Patterns annotations','future_edges','entity_mapping_without_asof_assumption'],
              families=families)
dest=ROOT/'config/feature_groups.json'
dest.write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
read=json.loads(dest.read_text(encoding='utf-8'))
assert set(read['experiments']['B']['feature_families'])-set(read['experiments']['A']['feature_families'])==extra
print(json.dumps({'families':len(families),'T':sum(f['group']=='T' for f in families),'G':sum(f['group']=='G' for f in families),'planned_A':len(core),'planned_B':len(core|extra)},ensure_ascii=True))
