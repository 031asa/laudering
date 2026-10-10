"""Export immutable Validation evidence for independent error analysis."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import zipfile

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gnn_features.contract import R_FIELDS, manifest as v1_contract
from gnn_features.v2_features import contract as v2_contract
from gnn_features.v2_adapter import projection_contract, TRIALS, REMOVED_R
from gnn_features.v2_compare import load_predictions, NAMES
from gnn_features.v2_evaluate import precision80_point
from export_gnn_missed_v2_features import export_v2_validation_features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/gnn_missed_data_20261003'
PREFIX, TOTAL = 20318992, 4689006
DATES = ['2022-09-11', '2022-09-12', '2022-09-13']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def log(stage, **values):
    print(json.dumps(dict(stage=stage, **values), ensure_ascii=True), flush=True)


def save_json(name, obj):
    with (OUT/name).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    assert not OUT.exists(), 'Preserve previously exported evidence'
    assert shutil.disk_usage(ROOT).free > 20*1024**3
    archive = ROOT/'data/gnn_v1_validation'
    cache = ROOT/'data/gnn_validation_cache_parquet'
    comparison_path = ROOT/'result/gnn_tuning_v2/comparison.corrected.json'
    comparison = read(comparison_path)
    rows = {x['name']: x for x in comparison['results']}
    frozen = {str(Path(k)): v for k,v in read(cache/'config.json')['hashes'].items()}
    side_config = read(ROOT/'data/gnn_v2_target_features/validation/config.json')
    r_hashes = {str(Path(k)): v for k,v in side_config['input_hashes'].items()}
    input_hashes = {}

    def check(path, inventory=frozen):
        p = Path(path).resolve()
        assert p.is_relative_to(ROOT)
        key = str(p.relative_to(ROOT))
        digest = sha(p)
        assert digest == inventory[key], key
        input_hashes[key] = digest
        return p

    check(archive/'build.json')
    meta = read(archive/'build.json')
    assert meta['rows'] == PREFIX+TOTAL and meta['end_exclusive'] == '2022-09-14T00:00:00'
    arr = {}
    for name in ['timestamp','src','dst','bank_key','payment_format_cat','payment_currency_cat']:
        arr[name] = np.load(check(archive/'history_index'/f'{name}.npy'), mmap_mode='r')
    banks = read(check(archive/'metadata/bank_identity.json'))
    vocab = read(check(archive/'metadata/vocabulary.json'))['values']
    bank70 = banks.index('70')
    assert bank70 == 122017
    times = arr['timestamp'][PREFIX:]
    scores, provenance = {}, {}
    y = None
    for name in NAMES:
        yy, score, audit = load_predictions(ROOT, name, times, y)
        if y is None: y = yy
        assert precision80_point(y, score) == rows[name]['precision80']
        assert audit['model_sha256'] == rows[name]['model_sha256']
        scores[name], provenance[name] = score, audit
        log('predictions_verified', model=name, rows=len(score))
    positive = y == 1
    src, dst = arr['src'][PREFIX:], arr['dst'][PREFIX:]
    non_ach = arr['payment_format_cat'][PREFIX:] != vocab['payment_format'].index('ACH')+2
    group_g = non_ach & (arr['bank_key'][src] == bank70)
    assert (int(positive.sum()),int((positive&non_ach).sum()),int((positive&group_g).sum()),int(group_g.sum())) == (6420,715,662,477760)
    hits = {name: scores[name] >= rows[name]['precision80']['threshold'] for name in NAMES}
    fn = positive & ~hits['lr_only']
    assert int(fn.sum()) == 2887

    # Reproduce the existing label-independent uniform negative sample.
    sample_path = ROOT/'result/gnn_error_analysis_v1/inspection_sample.parquet'
    sampled = pd.read_parquet(sample_path, columns=['edge_id','uniform_negative_control'])
    uniform = np.sort(sampled.loc[sampled.uniform_negative_control, 'edge_id'].to_numpy(np.int64)-PREFIX)
    expected = np.sort(np.random.default_rng(20261002).choice(np.flatnonzero(~positive),100000,replace=False))
    assert np.array_equal(uniform, expected)
    input_hashes[str(sample_path.relative_to(ROOT))] = sha(sample_path)
    rng = np.random.default_rng(20261003)
    chosen, strata = [], {}
    masks = {'bank70_non_ACH':group_g, 'other_non_ACH':non_ach&~group_g, 'ACH':~non_ach}
    for name,mask in masks.items():
        candidates = uniform[mask[uniform]]
        take = len(candidates) if name == 'bank70_non_ACH' else min(3000,len(candidates))
        keep = np.sort(rng.choice(candidates,take,replace=False)) if take<len(candidates) else candidates
        chosen.append(keep)
        strata[name] = dict(population_negatives=int((mask&~positive).sum()), source_uniform_sample=len(candidates), exported_controls=len(keep))
    control = np.zeros(TOTAL,bool)
    control[np.concatenate(chosen)] = True
    assert strata['bank70_non_ACH']['exported_controls'] == 10177
    fp_union = ~positive & np.logical_or.reduce(list(hits.values()))
    select = positive | control | fp_union
    ids = np.flatnonzero(select)+PREFIX
    positions = ids-PREFIX
    raw_columns = ['edge_id','transaction_id','source_record_number','timestamp','src','dst',
                   'from_bank','from_account','to_bank','to_account','amount_paid','amount_received',
                   'payment_currency','receiving_currency','payment_format',
                   'is_self_transfer','is_cross_bank','is_cross_currency']
    frames = []
    for date in DATES:
        for p in sorted((archive/f'graph/edges/transaction_date={date}').glob('*.parquet')):
            check(p)
            for batch in pq.ParquetFile(p).iter_batches(batch_size=65536, columns=raw_columns, use_threads=False):
                tab = pa.Table.from_batches([batch])
                local = tab['edge_id'].to_numpy().astype(np.int64)-PREFIX
                assert ((local>=0)&(local<TOTAL)).all()
                keep = select[local]
                if keep.any():
                    f = tab.filter(pa.array(keep)).to_pandas()
                    for col in ['amount_paid','amount_received']:
                        f[col] = f[col].map(lambda x: format(x,'.6f'))
                    frames.append(f)
        log('raw_transactions_verified',date=date)
    data = pd.concat(frames,ignore_index=True).sort_values('edge_id').reset_index(drop=True)
    assert np.array_equal(data.edge_id, ids) and data.edge_id.is_unique
    assert np.array_equal(data.timestamp.to_numpy().astype('datetime64[us]').astype(np.int64),times[positions])
    assert np.array_equal(data.src,src[positions]) and np.array_equal(data.dst,dst[positions])
    assert np.array_equal(data.from_bank.astype(str)=='70', arr['bank_key'][src[positions]]==bank70)
    assert np.array_equal(data.payment_format!='ACH',non_ach[positions])
    data.insert(4,'date',data.timestamp.dt.strftime('%Y-%m-%d'))
    data.insert(5,'label',y[positions])
    data['is_non_ACH'] = non_ach[positions]
    data['is_bank70_non_ACH'] = group_g[positions]
    data['is_uniform_negative_control'] = control[positions]
    data['control_stratum'] = np.where(control[positions],np.where(group_g[positions],'bank70_non_ACH',np.where(non_ach[positions],'other_non_ACH','ACH')),'')
    data['is_false_positive_any_model'] = fp_union[positions]
    for name in NAMES:
        data[f'score_{name}'] = scores[name][positions].astype(np.float64)
        data[f'alert_{name}'] = hits[name][positions]
        data[f'outcome_{name}'] = np.where(positive[positions],np.where(hits[name][positions],'TP','FN'),np.where(hits[name][positions],'FP','TN'))
    data['missed_by_all_four'] = positive[positions] & ~np.logical_or.reduce(list(hits.values()))[positions]

    # Re-read original raw R values from their frozen, checksum-bound outputs.
    r_cols = [f['name'] for f in R_FIELDS]
    r_frames = []
    for date in DATES:
        merged = check(ROOT/f'result/gnn_validation_v1/transaction_date={date}/merge_phase.json')
        for part in read(merged)['parts']:
            p = check(Path(part['path']),r_hashes)
            for batch in pq.ParquetFile(p).iter_batches(batch_size=65536,columns=['edge_id','timestamp']+r_cols,use_threads=False):
                tab = pa.Table.from_batches([batch])
                local = tab['edge_id'].to_numpy().astype(np.int64)-PREFIX
                assert ((local>=0)&(local<TOTAL)).all()
                keep = select[local]
                if keep.any(): r_frames.append(tab.filter(pa.array(keep)).to_pandas())
        log('original_R_verified',date=date)
    raw_r = pd.concat(r_frames,ignore_index=True).sort_values('edge_id').reset_index(drop=True)
    assert np.array_equal(raw_r.edge_id,ids)
    assert np.array_equal(raw_r.timestamp.to_numpy(),data.timestamp.to_numpy())
    for c in r_cols:
        assert not np.isinf(raw_r[c].to_numpy(dtype=float)).any()
    data = pd.concat([data,raw_r[r_cols]],axis=1)
    extra, extra_audit = export_v2_validation_features(ROOT, ids, times[positions], progress=lambda detail: log('v2_feature_export',detail=detail))
    assert np.array_equal(extra.edge_id,ids)
    for role,col in [('src','r_src_ext_out_count_24h'),('dst','r_dst_ext_in_count_24h')]:
        assert np.array_equal(data[col].fillna(-1).to_numpy(),extra[f'v2_{role}_count_24h_reused'].to_numpy())
    extra = extra.drop(columns=[c for c in ['edge_id','cutoff_us'] if c in extra.columns])
    data = pd.concat([data,extra],axis=1)
    assert data.columns.is_unique
    for role,c24 in [('src','r_src_ext_out_count_24h'),('dst','r_dst_ext_in_count_24h')]:
        for window,col in [('24h',c24),('72h',f'v2_{role}_count_72h'),('7d',f'v2_{role}_count_7d')]:
            values=data[col]
            data[f'history_support_{role}_{window}'] = np.where(values.isna(),'not_applicable',np.where(values==0,'0',np.where(values<5,'1-4','>=5')))
    # No implicit timezone conversion: timestamps retain the source's unspecified timezone.
    data['timestamp'] = data.timestamp.dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    OUT.mkdir(parents=True)
    tables = {}

    def save_csv(name, frame):
        path=OUT/name
        frame.to_csv(path,index=False,encoding='utf-8-sig',mode='x',float_format='%.17g',na_rep='')
        tables[name] = dict(rows=len(frame),columns=len(frame.columns),bytes=path.stat().st_size,sha256=sha(path))

    save_csv('01_当前模型漏检_2887.csv',data[data.outcome_lr_only=='FN'])
    save_csv('02_非ACH漏检_715.csv',data[(data.label==1)&data.is_non_ACH])
    save_csv('03_四模型正例全集_6420.csv',data[data.label==1])
    save_csv('04_正常对照样本.csv',data[data.is_uniform_negative_control])
    save_csv('05_四模型误报并集.csv',data[data.is_false_positive_any_model])

    # Full G population, without sampling, allows independent exact group ranking.
    gp=np.flatnonzero(group_g)
    gdata=pd.DataFrame({'edge_id':gp+PREFIX,'timestamp':pd.to_datetime(times[gp],unit='us').strftime('%Y-%m-%d %H:%M:%S.%f'),
                       'src':src[gp],'dst':dst[gp],'from_bank':'70','label':y[gp],
                       'payment_format':[vocab['payment_format'][int(c)-2] for c in arr['payment_format_cat'][PREFIX:][gp]],
                       'payment_currency':[vocab['currency'][int(c)-2] for c in arr['payment_currency_cat'][PREFIX:][gp]]})
    for name in NAMES: gdata['score_'+name]=scores[name][gp].astype(np.float64)
    save_csv('06_银行70非ACH全量分数_477760.csv',gdata)
    dictionary=[]
    basic_notes={
        'edge_id':'冻结归档的交易行标识，跨表一对一连接键；不能当作时间顺序。',
        'transaction_id':'原始交易标识，按字符串读取。','source_record_number':'源数据记录号。',
        'timestamp':'原始交易时间，微秒精度，源时区未指定；禁止自行转为UTC或当地时间。',
        'date':'从原始timestamp提取的日期YYYY-MM-DD。','label':'数据集真实标签：1正例，0负例；仅用于诊断，不是模型输入。',
        'src':'付款账户内部node_id，与原始银行编号不是同一编码。','dst':'收款账户内部node_id。',
        'from_bank':'原始付款银行字符串；银行70写作70，内部bank_key为122017。',
        'to_bank':'原始收款银行字符串，禁止用bank_key直接当银行编号。',
        'from_account':'原始付款账号字符串，应与from_bank联合识别账户。','to_account':'原始收款账号字符串，应与to_bank联合识别账户。',
        'amount_paid':'精确六位小数的付款金额文本；按Decimal读取，币种见payment_currency。',
        'amount_received':'精确六位小数的收款金额文本；按Decimal读取，币种见receiving_currency。',
        'payment_currency':'付款币种。','receiving_currency':'收款币种。','payment_format':'当前交易支付方式。',
        'is_self_transfer':'付款与收款node_id相同。','is_cross_bank':'双方原始银行不同。','is_cross_currency':'付款与收款币种不同。',
        'is_non_ACH':'支付方式不为ACH。','is_bank70_non_ACH':'原始付款银行70且非ACH。',
        'is_uniform_negative_control':'属于本包的分层随机负例对照；不是全量负例。',
        'control_stratum':'随机对照层：bank70_non_ACH/other_non_ACH/ACH，非对照行留空。',
        'is_false_positive_any_model':'四模型中至少一个在自身全局P80阈值产生FP；不是均匀样本。',
        'missed_by_all_four':'真实正例且四模型在各自全局P80阈值下都漏检。',
    }
    fields={f['name']:f for f in R_FIELDS}
    v2fields={f['name']:f for f in v2_contract()['fields']}
    for col in data.columns:
        used='诊断元数据，不作模型输入'
        note=basic_notes.get(col,'')
        if col in fields:
            f=fields[col];note=f"原始R特征；公式={f['formula']}；窗口={f['window']}；币种={f['currency']}；空值表示不适用或无有效参照，不能填成观测零；入模转换={f['transform']}。"
            used='lr_only/A；B/C删除此项及掩码' if col in REMOVED_R else 'lr_only/A/B/C'
        elif col in v2fields:
            f=v2fields[col];note=f"v2原始float64历史统计；账户侧={f['role']}；统计={f['kind']}；窗口={f['window']}；付款侧同付款币种外部付款、收款侧同收款币种外部收款；先log1p再float32入模。"
            used='A/B/C；lr_only未使用'
        elif col.endswith('__encoded'):
            note='原v2缓存的float32编码值，以float64精确展开存入CSV；log1p编码；无效时编码为0，必须结合__valid。';used='A/B/C'
        elif col.endswith('__valid'):
            note='对应v2特征有效性标记：1有效，0缺失/不适用；历史0笔的笔数为有效零。';used='A/B/C'
        elif col.startswith('v2_full_'):
            note='历史窗口是否从归档开始时间得到完整覆盖；不是账户首次出现时间。';used='A/B/C'
        elif col.startswith('score_'):
            note='冻结最终模型输出分数；原float32值以17位有效数字保存。加权损失分数不是已经校准的洗钱概率。'
        elif col.startswith('alert_'):
            note='分数>=该模型自身全局P80阈值；阈值见08文件，禁止对子群单独换阈值比较正式效果。'
        elif col.startswith('outcome_'):
            note='各模型自身全局P80阈值下的TP/FN/FP/TN；旧F1标签已弃用。'
        elif col.startswith('history_support_'):
            note='双方历史支持数分层：0、1-4、>=5、not_applicable。'
        elif col in ['src_count_7d','dst_count_7d'] or col.endswith('_count_24h_reused'):
            note='已核验精确整数历史笔数；-1是不适用，不是负的交易笔数。24h值与原R列交叉核验一致。'
        assert note, col
        dictionary.append(dict(field=col,dtype=str(data[col].dtype),used_by=used,description=note))
    save_csv('07_字段字典.csv',pd.DataFrame(dictionary))
    model_summary={name:dict(model_path=rows[name]['model_path'],model_sha256=rows[name]['model_sha256'],metrics=rows[name]['metrics'],
                            global_P80=rows[name]['precision80'],non_ACH_global_P80=rows[name]['non_ach_precision80']) for name in NAMES}
    save_json('08_模型阈值与全量指标.json',dict(selected_model='lr_only',validation_rows=TOTAL,validation_positives=6420,models=model_summary))
    breakdown=[]
    for keys in [['date'],['payment_format'],['from_bank'],['is_bank70_non_ACH']]:
        sub=data[data.outcome_lr_only=='FN'].groupby(keys,dropna=False).size()
        breakdown.append(dict(group_by=keys,counts={str(k):int(v) for k,v in sub.items()}))
    save_json('09_分组计数与抽样说明.json',dict(reference_FN=2887,non_ACH_FN=715,bank70_non_ACH_FN=662,
             old_best_F1_FN=2905,old_best_F1_threshold=0.9529672861099243,global_P80_threshold=rows['lr_only']['precision80']['threshold'],
             negative_control_strata=strata,negative_control_seed=20261003,original_uniform_sample_seed=20261002,
             sampling_note='Controls are stratified subsamples of the independently reproduced 100000 uniform Validation negatives. Full G scores are not sampled. FP enrichment is separate. Do not compute population precision or AP from enriched feature tables.',
             missed_breakdown=breakdown,all_four_missed=int(data.missed_by_all_four.sum()),false_positive_union=int(fp_union.sum())))
    save_json('10_特征契约.json',dict(v1=v1_contract(),v2=v2_contract(),projections={n:projection_contract(n) for n in TRIALS},
             exported='Raw current-transaction metadata, all 54 target R values, all 28 v2 raw values and 59 v2 encoded/valid/coverage columns. Not a complete graph minibatch: node N and historical message-edge E tensors are not included.'))

    # Round-trip the most important user-facing table, maintaining exact IDs/scores.
    recovered=pd.read_csv(OUT/'01_当前模型漏检_2887.csv',dtype={'from_bank':str,'to_bank':str,'from_account':str,'to_account':str,'amount_paid':str,'amount_received':str},float_precision='round_trip')
    want=data[data.outcome_lr_only=='FN']
    assert np.array_equal(recovered.edge_id,want.edge_id)
    for name in NAMES:
        assert np.array_equal(recovered['score_'+name].to_numpy(),want['score_'+name].to_numpy())
    assert recovered.amount_paid.tolist()==want.amount_paid.tolist()
    assert len(recovered)==2887 and (recovered.label==1).all() and (recovered.outcome_lr_only=='FN').all()
    assert recovered.from_bank.eq('70').sum()==662
    missing={c:int(data[c].isna().sum()) for c in r_cols+list(v2fields)}
    save_json('11_审计清单.json',dict(status='passed',created_at=datetime.now(timezone.utc).isoformat(),
             source_comparison_sha256=sha(comparison_path),export_script_sha256=sha(Path(__file__)),
             v2_export_helper_sha256=sha(Path(__file__).with_name('export_gnn_missed_v2_features.py')),
             selected_rows=len(data),tables=tables,input_hashes=input_hashes,
             prediction_provenance=provenance,v2_feature_audit=extra_audit,
             selected_feature_null_counts=missing,roundtrip_exact_score_ids_amounts=True,
             test_accessed=False,training_or_inference=False))
    readme=f'''GNN漏检交易数据包（2026-10-03）

用途：供GPTPro复盘交易级漏检、正负例特征差异与四模型分数变化。
范围：仅2022-09-11至13完整Validation。Test及尾部没有读取。本包使用真实保存的结果，无重训或新推理。
基准：目前保留的lr_only，单一全局P80阈值{rows['lr_only']['precision80']['threshold']!r}，判定score>=threshold为告警。
全量Validation 4,689,006笔、6,420正例；此阈值TP3533、FP880、FN2887。
旧报告的2905个漏检用的是bestF1阈值0.9529672861099243，差出的18笔在当前P80已被检出。不要混用两种口径。

建议先读取：
1. 01_当前模型漏检_2887.csv：当前保留模型全部漏检，含原交易信息、原54项R特征、新28项历史金额统计、掩码、覆盖信息以及四模型分数与判定。
2. 02_非ACH漏检_715.csv：重点非ACH正例。是01的子集，其中原始付款银行70有662笔。
3. 03_四模型正例全集_6420.csv：全部正例，可比较各模型新增检出、损失检出和共同漏检，不限于旧模型漏检。
4. 04_正常对照样本.csv：{int(control.sum()):,}个分层随机正常对照，G（付款银行70且非ACH）10,177个，其余非ACH3,000个、ACH3,000个。没有逐笔匹配，不能说是严格配对对照。
5. 05_四模型误报并集.csv：{int(fp_union.sum()):,}个至少被一个模型误报的负例。它与正常对照有重叠，跨文件必须按edge_id去重。
6. 06_银行70非ACH全量分数_477760.csv：G完整477,760笔，662正例、477,098负例，没有抽样。用于计算组内真实AP、ROC和误报预算下召回；这里只包含基础身份与分数，丰富特征在前面表中。
7. 07字段字典、08完整指标与阈值、09分组与抽样说明、10特征契约、11审计清单，定义与证据均保留。

读取注意：
- 文件UTF-8带BOM；所有交易/账户/银行标识按字符串读取，edge_id作为跨表连接键。金额保留六位小数精确文本，请用Decimal，不要让Excel自动改写账号或金额。
- timestamp来自原始数据，时区未指定；不要转为UTC/JST。不要用edge_id推断时间顺序。
- score是加权训练模型输出，不等于已校准的洗钱概率。四模型各用08中的一个全局阈值，不能用某一模型阈值套另一个。
- 空白特征是缺失/不适用，不是0。v2编码缺失时填0并同时valid=0；历史0笔时count=0且valid=1。带__encoded的列是实际float32缓存值；不带后缀的v2列是编码前float64统计。
- 付款侧统计同付款币种外部付款，收款侧统计同收款币种外部收款；只使用[t-W,t)历史，全部同时间戳交易排除，未使用历史标签。24h样本数复用原R列。
- 银行70指原始字符串70，内部bank_key为122017，不能把编码70当银行70。本包按原始身份校验。
- 数据表之间存在有意重叠。04是分层样本，05是定向误报集合，03全正例，不能拼接后直接声称总体精确率/AP。全量真实指标在08，G精确排序用06。
- 本包没有完整导出节点N、历史消息边E或完整时间图，因此不能仅凭本包断言GNN所有输入都有/没有信息。原特征定义在10；真实loader及完整图输入审计仍是后续工作。
- 数据集标签用于描述正负例，不代表现实世界司法认定。Validation已被用于诊断、特征设计与选型，任何探索发现都属于开发结论，未经过Test独立检验。

可交给GPTPro的分析请求：
请先核对漏检口径与四模型阈值，重点研究662笔“付款银行70且非ACH”正例。与04中同群体正常样本比较，并控制付款账户、支付方式、币种、日期及历史支持数；不把边际差异直接归因为模型漏检原因。用06区分分数水平偏移和组内排序失败，检查少量高分正常交易、正负例特征重叠、可分模式和时间变化。比较A/B/C对同一交易的变化，明确哪些结论有数据支持、哪些仍需真实输入或Train固定模型推理验证。请勿因抽样比例失真计算总体精确率，勿把标签/事后分数重新当作训练特征，勿自行开启Test或宣称新方案已验证有效。
'''
    (OUT/'00_阅读说明.txt').write_text(readme,encoding='utf-8')
    zip_path=OUT.parent/'GNN漏检交易数据包_20261003.zip'
    assert not zip_path.exists()
    with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file(): z.write(p,p.name)
    with zipfile.ZipFile(zip_path) as z: assert z.testzip() is None
    log('complete',zip=str(zip_path),zip_bytes=zip_path.stat().st_size,selected_rows=len(data),controls=int(control.sum()),fp_union=int(fp_union.sum()))


if __name__=='__main__':
    main()
