"""Package verified local analysis outputs for the user's manual handoff."""
from pathlib import Path
import hashlib, json, zipfile

P=Path(__file__).resolve().parents[1]
D=P/'result/gnn_chain_audit_20261003'
read=lambda n:json.loads((D/n).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert read('run_manifest.json')['status']=='completed'
verification=read('verification.json')
assert verification['status'] in ['passed','verified','completed']
assert (D/'链条审计结果.txt').is_file()
names=['链条审计结果.txt','transaction_audit.csv','transaction_audit.parquet',
       'positive_edges.parquet','ach_deep_walk_witnesses.json',
       'cohort_summary.json','matched_statistics.json','endpoint_cluster_sensitivity.json',
       'bank70_receiver_followup.json','ach_detected_reference.json','verification.json',
       'run_manifest.json','audit_inputs.json','labels_manifest.json','candidate_manifest.json',
       'sampling_format_recovery.json']
for c in ['ACH_FN','G_FN']:
    names += [c+'_pairs.csv',c+'_unmatched.csv',c+'_matching.json']
for c in ['positive_history_features','full_history_neighbors','sampling_audit']:
    names += [c+'.audit.json']
for c in ['positive_history_features','full_history_neighbors']:
    names += [c+'.reuse_audit.json']
notes='''历史链条回溯审计，2026-10-03

先读“链条审计结果.txt”。transaction_audit每行一笔，共5138笔：772 ACH漏检、1508同条件ACH检出、662付款银行70非ACH漏检，以及484/1712笔不重复正常对照。
ACH_FN/G_FN_pairs保存配对和control_weight：每个病例权重1，其1至3个对照平分权重。对照可以复用，不能当成独立样本。两个unmatched表列出未匹配病例。
positive_edges是Train+Validation全部24,214笔最终正例的精确历史边身份；不是完整无标签交易图。最终标签没有确认时间，仅用于事后审计，不能直接投入模型。
所有historical字段限[t-7d,t)或[t-72h,t)，排除全部同时间戳；future_字段单独是事后观察，通常右删失，不能当完整72h结果或输入特征。Test未读取。
src/dst为冻结账户索引，edge_id为冻结交易索引；cutoff_us为原始交易时间的微秒整数。raw银行与账户字符串在逐笔CSV中保留，读CSV时不要把账号转换为浮点数。
prior_positive_any_7d：任一端点之前7天接触最终正例。mixed_2hop：经任意无标签首边可接到正例边的拓扑二跳，排除直接正例边；不要求路径内时间有序。
src_temporal_walk_ge2_7d：至少两条严格时间递增、有向历史正例边，终点是当前付款账户。最长walk可重访账户，不保证同币种、金额守恒或同一资金。36笔ACH案例的完整见证在ach_deep_walk_witnesses中。
sampled字段来自原实际批次缓存，未重采样；标签只在本次审计中叠加，原GNN没有收到历史标签。
matched_statistics为预先列明的六项探索检验；endpoint_cluster_sensitivity合并付款、收款共享账户作保守复核。bank70_receiver_followup是事后描述，没有新增p检验。
manifest列出SHA及复核记录。完整100000负例候选池仍留档于项目审计目录的candidates.parquet，本包不重复携带；逐笔审计和全部匹配结果已包含。
'''
(D/'数据说明.txt').write_text(notes,encoding='utf-8')
names.append('数据说明.txt')
inventory={n:dict(bytes=(D/n).stat().st_size,sha256=sha(D/n)) for n in names}
manifest=dict(status='verified_delivery',source='result/gnn_chain_audit_20261003',files=inventory)
(D/'delivery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
archive=P/'outputs/GNN历史链条审计_20261003.zip'
assert not archive.exists(), 'Preserve an existing delivery; do not overwrite it'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for n in names+['delivery_manifest.json']:z.write(D/n,arcname=n)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for n,v in inventory.items():assert hashlib.sha256(z.read(n)).hexdigest()==v['sha256']
receipt=dict(path=archive.relative_to(P).as_posix(),bytes=archive.stat().st_size,sha256=sha(archive),files=len(names)+1)
(D/'delivery_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=True))
