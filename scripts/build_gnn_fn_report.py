"""Build the Chinese report from saved, auditable Validation diagnostics."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'result/gnn_error_analysis_v1'
def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def pct(x):return f'{x:.2%}'
def num(x):return f'{int(x):,}'
def val(x):return '缺失' if x is None else f'{x:,.6g}'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
    b=read('basic_analysis.json');hist=read('history_summary.json');ev=read('case_evidence.json')
    p=pd.read_parquet(OUT/'inspection_sample.parquet')
    ach=p[(p.payment_format=='ACH')&(p.src!=p.dst)&(p.from_bank!='70')].copy()
    fn=ach[ach.group=='FN'];tp=ach[ach.group=='TP'];neg=ach[ach.uniform_negative_control]
    keys=['date','payment_format','currency','amount_band']
    fc=fn.groupby(keys).size().rename('fn_count');nc=neg.groupby(keys).size().rename('negative_count')
    strata=pd.concat([fc,nc],axis=1).dropna().reset_index()
    matched=neg.merge(strata,on=keys,validate='many_to_one');matched['weight']=matched.fn_count/matched.negative_count
    masks={
      'last_same_currency_receipt_over_72h':lambda x:x.r_src_last_same_currency_ext_in_gap_min>4320,
      'no_same_currency_receipt_in_available_history':lambda x:x.r_src_last_same_currency_ext_in_gap_min.isna(),
      'source_24h_mean_reference_missing':lambda x:x.r_src_current_to_own_ext_mean_24h.isna(),
      'destination_no_same_currency_out_24h':lambda x:x.r_dst_ext_out_count_24h.eq(0),
      'pair_no_forward_72h':lambda x:x.r_pair_fwd_count_72h.eq(0)}
    match_summary={'scope':'ACH, non-self, source bank != 70; matched by day/payment/currency/within-currency amount band',
        'fn_rows':len(fn),'tp_rows':len(tp),'negative_uniform_rows':len(neg),
        'matched_fn_support':int(strata.fn_count.sum()),'strata':len(strata),'matched_negative_rows':len(matched),
        'effective_negative_sample_size':float(matched.weight.sum()**2/(matched.weight**2).sum()),'features':{}}
    for k,f in masks.items():
        match_summary['features'][k]={'fn_yes':int(f(fn).sum()),'fn_rate':float(f(fn).mean()),'tp_yes':int(f(tp).sum()),'tp_rate':float(f(tp).mean()),
            'matched_negative_rate':float(np.average(f(matched),weights=matched.weight))}
    (OUT/'matched_analysis.json').write_text(json.dumps(clean(match_summary),ensure_ascii=False,indent=2),encoding='utf-8')
    cby={c['edge_id']:c for c in ev['cases']}; fcases=[c for c in ev['cases'] if c['role']=='FN']
    explanations={
      24588727:'这是高频付款源中的小额、重复金额交易：模型已经看到了双方此前往来和收款方历史，当前金额只占付款方24h平均额约0.14%。不能仅凭这笔重复或银行70就断言风险；同一来源大量交易是标签0。应检查收款对象的差异、账户角色和源账户内部的金额/节奏分布，而不是整体拉黑源账户。',
      22658671:'现有收款端金额均值比约50,315倍，异常量级已经明确进入R，仍被低评分。这个例子直接反驳“只缺一个金额异常字段”的解释。它支持进一步检验模型是否过度依赖支付方式、正常账户习惯或某种特征组合；单个案例不能确定因果。',
      21710846:'上一笔同币种入账在约11.22天前，超出72h采图窗口；其16160分钟间隔已经入模，但老入账的对手和金额关系没有通过该条图边传入。当前收款方24h有64笔入账，显示可能还需看更长历史及收款侧群体结构。较老金额与当前金额并不一致，不能直接认定它们是同一笔资金。',
      24293068:'19分钟前收到3583.79欧元，当前转出3597.07欧元，差0.3692%；期间还有其它入账，所以当前额/24h总入账仅约0.311。单笔金额匹配与总额比例表达的信息不同。但匹配边已经在原采样图中，不能说模型没看到这条边；这支持明确表达“金额接近×等待时间”的候选交互，而非保证加特征必然改善。',
      22992244:'13:12 当前收款方→中间账户，13:52 中间账户→当前付款方，19:27 当前交易闭环。两条历史边均在实际缓存中。三笔金额不守恒，故只能称有序拓扑环。sqrt_weight对此笔分数约0.986而lr_only约0.033，同一特征和采样也能产生截然不同排序，说明训练/参数影响不能排除。',
      24329059:'当前付款约为付款方24h均值的21.4倍，且此前当前支付方式份额为0，这些偏离已经入模；但收款方自身金额基数大，当前额仅为收款方历史均值的约3.58%。这是双方尺度相冲突的低分案例，适合研究角色交互、账户对金额参照和分位数，而不是再加一份相同的均值比。',
      21448983:'分数0.952129紧邻当前阈值0.952967；这是典型阈值边界样本，不能从它单独反推出缺少某种图特征。此前六笔入账已进入实际采样图。调阈值的收益必须和全量误报一起衡量。'}
    # Separate detailed case notebook, preserving raw account identity and exact amounts.
    caseparts=['# 典型漏检流水与对照（2026-10-02）','这7笔由已知漏检中按不同条件选取，是解释性案例，不是随机总体样本。所有历史行严格早于目标交易；表中每端每方向最多展示最近5笔，再加入金额匹配/最近入账/环见证。24h计数是完整窗口计数，未因展示截断。账户是“银行编号＋账号”，edge_id是归档内部交易标识。标签只用作事后分组，历史流水表不把历史标签当作当时已知风险。',
      '原采样图来自真实预测使用的缓存；“进入图”表示该历史边在目标所属批次的输入图中，不能保证模型充分利用了该信息。目标时刻的其它交易也排除；数值金额按原币种记录，不跨币种相加。']
    ledger=[]
    for i,c in enumerate(fcases,1):
        x=c['target'];e=c['edge_id']
        caseparts.extend([f'## {i}. {c["title"]}（edge_id={e}）',
          f'目标：**{x["timestamp"]}**，`{x["from_bank"]}/{x["from_account"]}` → `{x["to_bank"]}/{x["to_account"]}`，**{x["amount_paid"]} {x["payment_currency"]} / {x["payment_format"]}**。标签1，lr_only分数 **{x["score_lr_only"]:.9g}**，当前阈值0.952967286，属于漏检。',
          f'原始 transaction_id：`{x["transaction_id"]}`。',explanations.get(e,''),
          table(['已有R特征','原始值'],[[k,val(v)] for k,v in c['existing_R'].items()]),
          '完整24h外部流水笔数：'+ '；'.join(f'{k}={v:,}' for k,v in c['history_24h_counts'].items())+'。'])
        rows=[]
        for h in c['history_events']:
            rows.append([h['edge_id'],h['timestamp'].replace('T',' ')[:19],h['from']['bank']+'/'+h['from']['account'],h['to']['bank']+'/'+h['to']['account'],h['paid']+' '+h['payment_currency'],h['received']+' '+h['receiving_currency'],h['payment_format'],'是' if h['in_original_plan'] else '否'])
            ledger.append({'case_edge_id':e,'row_type':'history','edge_id':h['edge_id'],'timestamp':h['timestamp'],
                'from_bank':h['from']['bank'],'from_account':h['from']['account'],'to_bank':h['to']['bank'],'to_account':h['to']['account'],
                'amount_paid':h['paid'],'payment_currency':h['payment_currency'],'amount_received':h['received'],'receiving_currency':h['receiving_currency'],
                'payment_format':h['payment_format'],'in_original_plan':h['in_original_plan']})
        target_row=p.loc[p.edge_id.eq(e)].iloc[0]
        ledger.append({'case_edge_id':e,'row_type':'target','edge_id':e,'timestamp':x['timestamp'],
            'from_bank':x['from_bank'],'from_account':x['from_account'],'to_bank':x['to_bank'],'to_account':x['to_account'],
            'amount_paid':x['amount_paid'],'payment_currency':x['payment_currency'],'amount_received':x['amount_received'],
            'receiving_currency':target_row.receiving_currency,'payment_format':x['payment_format'],'in_original_plan':None})
        caseparts.append(table(['edge_id','发生时间','付款方','收款方','付款金额','收款金额','方式','进入原图'],rows))
        controls=[]
        for m in ev['controls']:
            if m['fn_edge_id']!=e:continue
            if m['edge_id'] is None:controls.append([m['control'],'无符合条件的对照','—','—','—']);continue
            t=cby[m['edge_id']]['target']
            controls.append([m['control']+'（'+t['group']+'）',m['edge_id'],t['amount_paid']+' '+t['payment_currency'],f'{t["score_lr_only"]:.9g}','是' if m['same_src'] else '否'])
        caseparts.extend(['对照按同日期、支付方式、币种，尽量靠近金额和时刻选择；负例有同付款账户时优先同账户。未控制所有历史属性，所以只是阅读对照，不能计算增量模型收益。',table(['类型','edge_id','金额','分数','同付款账户'],controls)])
    (OUT/'典型漏检流水.md').write_text('\n\n'.join(caseparts)+'\n',encoding='utf-8')
    pd.DataFrame(ledger).to_csv(OUT/'case_history_transactions.csv',index=False,encoding='utf-8-sig')
    # Concise primary report.
    m=match_summary['features']
    parts=['# GNN漏检分析：从实际交易反推特征缺口','分析日期：2026-10-02。数据：IBM AML HI-Medium合成交易；只分析2022-09-11至13的完整Validation。',
      '## 先给结论',
      '**主要问题不是简单的“只做了一跳”，也不能只靠补几个环特征解释。** 当前模型确实是两层、两跳历史GNN；本次发现的主要盲区是非ACH正例被整体压低、部分ACH缺少近期可用的金额参照，以及较老历史关系超出72h采图范围。同时，有明显异常特征/历史边已进入模型却仍低分的案例，因此下一步需要同时检查特征表达与模型对现有信息的使用。',
      '本次完成：核验三组197块预测的哈希、交易身份、时间、标签和完整覆盖；逐笔分析全部6,420个正例及2,905个漏检；以100,000个均匀负例作已有特征对照，再以其中10,000个负例作严格历史结构对照；回放7个典型漏检及对照交易，并核对真实预测使用的采样图。没有重新训练，也没有评估Test。',
      '## 1. 漏检口径和阈值代价',
      '采用胜出模型lr_only和此前Validation最佳F1阈值 **0.9529672861**。完整Validation为4,689,006笔、6,420正例；TP=3,515，FN=2,905，FP=810，TN=4,681,776；精确率81.27%，召回54.75%。以下“漏检”均按此阈值定义。',
      '2,905个漏检中，2,041个分数低于0.5；其中852个低于0.01。降低到0.5只能找回864个，同时新增13,308次误报，新增告警的正例比例约6.10%。不能把全部问题当成阈值设高了。',
      table(['目标/当前选择','阈值','精确率','召回率','TP','FP'],
        [['当前最佳F1','0.952967','81.27%','54.75%','3,515','810']]+
        [[('精确率≥' if x['constraint']=='precision' else '召回≥')+pct(x['target']),f'{x["threshold"]:.6f}',pct(x['precision']),pct(x['recall']),num(x['tp']),num(x['fp'])] for x in b['threshold_tradeoffs']]),
      '这些阈值仍是对同一Validation的事后选择，不是新的独立效果验证；模型分数也不等于已经校准的真实风险概率。',
      '## 2. 最清楚的盲区：715个非ACH正例全漏',
      table(['方式','全量交易','正例','识别TP','漏检FN','召回率'],[[x['payment_format'],num(x['total']),num(x['positives']),num(x['tp']),num(x['fn']),'—' if x['recall'] is None else pct(x['recall'])] for x in b['groups']['payment_format']]),
      '非ACH正例分数最高仅 **0.034779**，连0.5都很远。支付方式已经入模；这提示应检验模型是否过度依赖方式或账户角色，而不是再添加“是否ACH”字段。此前Train也有2,727个非ACH正例，不能用“训练时完全没有这类正例”解释。是否存在捷径依赖，仍需要后续去除类别信息等对照实验确定。',
      '**其中662个漏检来自银行编号70的15个付款账户，占全部漏检22.79%，与非ACH盲区高度重叠。** 前两个账户分别贡献245和140笔。但银行70在Validation共有477,760笔，只有662个标签1，比例约0.1386%；均匀负例样本中同样15户还贡献10,177笔标签0。整户或整行拉黑会产生大量误报。更值得补的是同一高频账户内部的收款对象差异、金额相对分布、异常往来和行为变化。编号70只是合成数据标识。',
      'ACH分日召回为62.01%、60.90%、61.92%，较稳定；总体后两天召回较低，与非ACH正例从121增加到299、295的构成变化一致，不能据此直接判定模型逐日衰退。',
      '## 3. ACH内部：哪些是证据，哪些只是表象',
      f'以下排除自转和付款银行70，比较{len(fn):,}个ACH漏检、{len(tp):,}个ACH已识别正例。负例按日期×支付方式×币种×金额带匹配，可支持{match_summary["matched_fn_support"]:,}/{len(fn):,}个漏检，共{len(strata)}层、{len(matched):,}个负例；按各层漏检数加权。它调整了部分构成差异，仍不是因果检验。',
      table(['观察条件','FN中占比','TP中占比','匹配负例占比','解读'],[
        ['上次同币种入账在72h以前',pct(m['last_same_currency_receipt_over_72h']['fn_rate']),pct(m['last_same_currency_receipt_over_72h']['tp_rate']),pct(m['last_same_currency_receipt_over_72h']['matched_negative_rate']),'值得查长时滞历史关系'],
        ['24h付款金额均值参照缺失',pct(m['source_24h_mean_reference_missing']['fn_rate']),pct(m['source_24h_mean_reference_missing']['tp_rate']),pct(m['source_24h_mean_reference_missing']['matched_negative_rate']),'只加同24h中位数不能填补支持不足'],
        ['收款方24h同币种无转出',pct(m['destination_no_same_currency_out_24h']['fn_rate']),pct(m['destination_no_same_currency_out_24h']['tp_rate']),pct(m['destination_no_same_currency_out_24h']['matched_negative_rate']),'正常交易也常见，不能直接报险'],
        ['双方72h没有同向往来',pct(m['pair_no_forward_72h']['fn_rate']),pct(m['pair_no_forward_72h']['tp_rate']),pct(m['pair_no_forward_72h']['matched_negative_rate']),'已入模，FN和TP相近，不能解释两者差异']]),
      '“较老入账”组中，467笔漏检、120笔识别，召回仅20.44%。**最近同币种入账间隔本来已经入模**，缺口候选是较老边的金额、对手及结构关系，而不是缺少这个时间间隔字段。不能反过来说“没有任何历史导致漏检”：没有同币种入账历史在TP中反而更常见。2,905笔FN中，有2,846笔（97.97%）的付款账户曾在Train出现，也不支持把主要原因归为新账户冷启动。',
      '另一个局部问题是部分币种的ACH识别弱：排除自转和银行70后，BRL/MXN/SAR/CHF合计160个正例仅3个TP；加CAD/Shekel后共354个正例、43个TP。币种本身已经入模；应检查训练支持、相对金额尺度和角色交互，不能直接给这些币种贴风险标签。',
      '## 4. 显式复杂特征：测过覆盖后再排优先级',
      '严格取目标前24h，同时间戳全部排除。金额匹配按“历史入账收款币种=当前付款币种”，比较单笔历史收款额与当前付款额；环要求当前u→v闭合历史v→x、x→u，且t₁<t₂<t、三个账户不同。下面负例为均匀10,000笔样本，正例和FP仍为全量。',
      table(['候选形态','FN / 2,905','TP / 3,515','负例样本 / 10,000','FP / 810'],
        [[label]+[str(hist['groups'][g]['flags'][flag]['yes']) for g in ['FN','TP','negative_uniform','FP']]
         for label,flag in [('24h单笔入账与当前付款差≤1%','amount_match_1pct_24h'),('24h单笔入账与当前付款差≤5%','amount_match_5pct_24h'),('≤1h且金额差≤5%','amount_match_5pct_1h'),('24h有序三边闭环','cycle_found')]]),
      '**三边环只覆盖2笔FN；金额差≤5%覆盖47笔FN，负例样本也有142笔符合。** 它们可以作为局部候选交互，但目前不支持“加上这几个复杂特征就能大幅解决漏检”。负例样本未见环不代表全量没有环型误报；本次也没有排查所有长度、全部时间窗口、扇入扇出序列或社区结构。',
      '## 5. 真实漏检流水告诉我们的区别',
      table(['案例edge_id','观察','得分','意味着什么'],[
        [24588727,'242.29美元信用卡；付款方前24h外部转出72,252笔','0.000207','高频角色下的细粒度异常，不能整户判险'],
        [22658671,'7.0269亿欧元支票；收款端均值比约50,315倍','0.034779','明显已有异常信号仍低分，应查模型如何使用特征'],
        [21710846,'5,573.72巴西雷亚尔ACH；上次入账约11.22天前','0.000191','较老关系超72h采图窗口；间隔字段已入模'],
        [24293068,'19分钟前入账3,583.79欧元；当前转出3,597.07欧元','0.023314','需要区分单笔金额匹配与24h总额比；历史边已入图'],
        [22992244,'同日13:12→13:52→19:27闭合有序三边环','0.032694','两条历史边都已入图，不能归因没采到'],
        [24329059,'ACH付款为自身24h均值21.4倍，但仅收款方均值3.58%','0.0000126','双端尺度/账户对角色交互值得检查'],
        [21448983,'欧元ACH，已有6笔前序入账','0.952129','阈值边界案例，不宜硬解释为缺少图特征']]),
      '详见[7例完整流水与对照](result/gnn_error_analysis_v1/典型漏检流水.md)，含原始交易ID、银行/账号、精确金额、严格时间顺序、已入模R值和原采样图核验。案例按机制选取，不能当成各机制总体发生率。',
      '## 6. 下一轮该补什么',
      table(['优先级','建议','本次支持与限制'],[
        ['优先','把非ACH、超高频付款账户单独评估；检验类别信息依赖、角色交互','715个非ACH正例全漏，包含已有巨大金额异常的样本。先确认模型是否忽略已有信息。'],
        ['优先','更长窗口的可用历史数、金额分位数、跨窗口回退，以及较老关系摘要','ACH较老入账组召回低，24h金额参照大量缺失；已有间隔不应重复新增。'],
        ['随后','账户对金额变化、重复金额、付款/收款角色条件下的相对异常','高频源的账户整体均值可能淹没小额异常；要和同源正常流水比。'],
        ['局部候选','有序短链、单笔金额匹配×等待时间、环/扇入扇出组合','本次已测环和金额匹配覆盖有限；需逐项做同告警预算的增量验证。'],
        ['有条件','交易对手历史确证风险','必须知道标签何时可用，并在时间上合法构造；不能直接把Validation标签回填。']]),
      '这轮分析已经使用Validation错误寻找方案，下一轮在同一Validation变好只能算开发结果。应在Train内部按时间做特征取舍，并持续报告非ACH/高频账户等分组表现以及固定精确率、固定误报预算下的召回。Test仍保留给最终一次独立检验；本次没有启动新候选、增加epoch或改动冻结模型。']
    if (OUT/'history_long_window_summary.json').exists():
        lw=pd.read_parquet(OUT/'history_long_window.parquet')
        la=lw[(lw.payment_format=='ACH')&(~lw.self_transfer)]
        lf=la[la.group=='FN'];lt=la[la.group=='TP'];ln=la[la.history_negative_control]
        missing=lf[lf.src_original_24h_mean_missing]
        supports={n:int((missing.src_same_currency_ext_count_7d>=n).sum()) for n in [1,3,5]}
        assert len(missing)==1698 and supports=={1:593,3:233,5:178}
        idx=parts.index('## 4. 显式复杂特征：测过覆盖后再排优先级')
        parts[idx:idx]=['### 已补查7天历史：有用的候选，但不能无限回退',
          f'上述1,698笔缺少24h可用付款均值的ACH漏检，向前查7天后，{supports[1]}笔（34.92%）出现同币种外部付款参照；至少3笔支持仅{supports[3]}个，至少5笔仅{supports[5]}个。多数仍然参照不足，应把历史笔数、缺失标记和窗口长度一起表达，不把一两笔历史当成稳定基线。',
          table(['当前收款额超过收款方7天P90，且历史≥5笔','命中数','该组内占比'],
            [[label,f'{int(df.dst_amount_above_p90_with_ge5.sum()):,} / {len(df):,}',pct(df.dst_amount_above_p90_with_ge5.mean())] for label,df in [('ACH漏检',lf),('ACH已识别',lt),('ACH均匀负例样本',ln)]]),
          '这个金额分位数信号覆盖503笔ACH漏检，值得做增量检验；但TP中的出现率更高，上表未做额外匹配，也不是“新规则精确率”。现有收款端均值比已经入模，分位数能否提供额外信息尚未证明。原始结果见[长窗口分析](result/gnn_error_analysis_v1/history_long_window_summary.json)。']
    if (OUT/'train_label_link.json').exists():
        label=read('train_label_link.json')['groups_and_slices']
        parts.extend(['### 你提出的“交易对手本身风险”，这次也核对了',
          '只固定09-01至10的Train交易标签，不使用较早Validation标签回流。检查当前收款账户是否在Train曾经作为付款或收款方参与标签1交易，结果如下：',
          table(['组别','收款方有Train正例关联','占该组比例'],
            [[title,f'{label[scope][group]["payee_train_any_positive"]["yes"]:,} / {label[scope][group]["n"]:,}',pct(label[scope][group]['payee_train_any_positive']['rate'])]
             for title,scope,group in [('全部漏检','all','FN'),('全部已识别正例','all','TP'),('均匀负例样本','all','uniform_negative_control'),('银行70付款的漏检','payer_bank70','FN'),('银行70付款的负例样本','payer_bank70','uniform_negative_control')]]),
          '该关联对部分交易有研究价值，但仅覆盖292个漏检（10.05%），不能期待它单独补回大部分FN；在银行70付款这组里，漏检与正常对照的发生率几乎相同。15个高频付款账户本身虽然都曾付款正例，其正常对照也全部触发这一标志，所以“以前出现过正例”不能直接等于当前每笔交易高风险。',
          '**这只是事后标签关联，不代表当时已知风险。** 数据没有标签确认时间；若下一轮实验，必须明确何时获知标签、做时间递进编码与小样本收缩，不能把当前或未来标签泄漏给特征。详情及身份核验见[交易对手历史标签关联](result/gnn_error_analysis_v1/train_label_link.json)。'])
    parts.extend(['## 7. 数据与复核入口',
      '- [2,905笔漏检明细](result/gnn_error_analysis_v1/false_negatives.csv)：三模型分数、原始交易身份和54个已有R特征。\n- [全部6,420笔正例](result/gnn_error_analysis_v1/all_validation_positives.csv)。\n- [分层与阈值统计](result/gnn_error_analysis_v1/basic_analysis.json)。\n- [匹配对照口径和结果](result/gnn_error_analysis_v1/matched_analysis.json)。\n- [严格历史结构统计](result/gnn_error_analysis_v1/history_summary.json)及[独立核验记录](result/gnn_error_analysis_v1/history_verification.json)。\n- [案例与原图证据JSON](result/gnn_error_analysis_v1/case_evidence.json)、[案例历史交易CSV](result/gnn_error_analysis_v1/case_history_transactions.csv)。\n- [先前的项目完整交接文档](GPTPro交接_GNN交易反洗钱项目_2026-10-02.md)。',
      '复现代码：scripts/analyze_gnn_false_negatives.py、scripts/analyze_gnn_fn_history.py、scripts/analyze_gnn_fn_long_window.py、scripts/analyze_gnn_train_label_link.py、scripts/extract_gnn_fn_cases.py、scripts/build_gnn_fn_report.py。输出与冻结训练/预测目录分开；输入始终只读。金额按各自币种比较，未做汇率折算；“历史”从数据集2022-09-01开始，不代表账户真实开户以来的全部历史。',
      '最终[核验记录](result/gnn_error_analysis_v1/analysis_verification.json)确认漏检/正例明细唯一且计数一致、18个案例及对照的历史严格早于目标，冻结流水线、验证器和胜出模型哈希与原记录一致。'])
    (ROOT/'GNN漏检分析_2026-10-02.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    print(json.dumps({'report':'GNN\u6f0f\u68c0\u5206\u6790_2026-10-02.md','case_count':len(fcases),'matched_fn':match_summary['matched_fn_support']},ensure_ascii=True))

if __name__=='__main__':main()
