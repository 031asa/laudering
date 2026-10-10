"""Render the verified aggregate results into a Chinese analysis note."""
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/train_bank_analysis'
p=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
m=pd.DataFrame(p['methods']['train']).set_index('payment_format')
b={r:pd.read_csv(OUT/f'banks_{r}.csv',dtype={'bank':str}).set_index('bank') for r in ['from','to']}
g=pd.read_csv(OUT/'bank_payment_daily_from.csv',dtype={'bank':str})
s=g[g.bank=='70'].groupby('payment_format')[['total','positive']].sum()
ex=pd.read_csv(OUT/'banks_to_excluding_from70.csv',dtype={'bank':str}).set_index('bank')
a=pd.read_csv(OUT/'bank70_account_currency_payment.csv',dtype={'from_account':str})
assert int(m.total.sum())==20318992 and int(m.positive.sum())==17794
assert int(s.total.sum())==1801617 and int(s.positive.sum())==2551
assert a.from_account.nunique()==15
positive_export=pd.read_csv(ROOT/'result/train_laundering/训练集_全部洗钱标签交易.csv',dtype={'from_bank':str,'to_bank':str})
checks={}
for role in ['from','to']:
    grouped=positive_export.groupby(role+'_bank').size().reindex(b[role].index,fill_value=0)
    assert (b[role].positive==grouped).all()
    assert b[role].total.sum()==20318992 and b[role].positive.sum()==17794
    checks[role+'_bank_positive_counts_exact']=True
assert (m.positive==positive_export.groupby('payment_format').size().reindex(m.index,fill_value=0)).all()
checks['method_positive_counts_exact']=True
checks['comparison_source']='independently verified full training positive export'
(OUT/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
lines=['# 训练集银行与支付方式分析','',
'范围：2022-09-01至09-10完整Train（发现期＋内部验证期），共20,318,992笔交易，17,794笔标签1，整体正例率约0.0876%。本次按用户要求观察完整训练期，并分前7天/后3天描述稳定性；后3天因此已被观察，不能再称未经探索。没有读取主验证集、测试集或尾部交易。', '',
'数据是IBM AML HI-Medium合成数据，标签表示洗钱，不含单独的诈骗类别。以下银行名称和编号仅是此数据中的标识，不构成对现实机构的评价。', '',
'## 结论', '',
'支票正例数量排第二，但相对其庞大的交易基数，正例率低于全局。更关键的是，支票正例94.53%来自编号70的付款银行，其付款方只有15个极高频账户。另一方面，部分收款银行在调整日期和支付方式、以及排除70号银行付款后，仍然有更高的正例比例。这是局部关联信号，不是用户主动偏好或银行导致风险的证据。', '',
'## 1. 支票不少，是否意味着偏好', '',
'Lift＝该类别正例率÷整体正例率。总交易占比与正例占比必须同时看。', '',
'| 支付方式 | 全部交易数 | 正例数 | 占全部交易 | 占全部正例 | 类内正例率 | Lift |',
'| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for name in ['ACH','Cheque','Credit Card','Cash','Bitcoin','Wire','Reinvestment']:
    r=m.loc[name]
    lines.append(f'| {name} | {int(r.total):,} | {int(r.positive):,} | {r.share_all:.2%} | {r.share_positive:.2%} | {r.positive_rate:.4%} | {r.lift:.2f} |')
non=m.drop('ACH')
cheque_non_lift=(m.loc['Cheque','positive']/m.loc['Cheque','total'])/(non.positive.sum()/non.total.sum())
lines += ['',f'只在非ACH交易中比较，支票占非ACH交易{m.loc["Cheque","total"]/non.total.sum():.2%}、占非ACH正例{m.loc["Cheque","positive"]/non.positive.sum():.2%}，相对这一基线的Lift为{cheque_non_lift:.2f}。因此“相对所有交易偏低”与“相对非ACH平均略高”可以同时成立，但下面的付款来源集中性说明，不能仅把它归因于支票方式。', '',
'前7天支票正例率0.01829%，后3天0.01804%，基本相近；ACH则为0.55864%和0.79353%。在训练期没有正例的类别也不能直接作为未来绝对安全规则。', '',
'## 2. 银行70：支票现象主要集中在这里', '',
'| 方式 | 全训练期正例 | 其中银行70付款 | 银行70占该方式正例 |',
'| --- | ---: | ---: | ---: |']
for name in ['Cheque','Credit Card','Cash','Bitcoin']:
    lines.append(f'| {name} | {int(m.loc[name,"positive"]):,} | {int(s.loc[name,"positive"]):,} | {s.loc[name,"positive"]/m.loc[name,"positive"]:.2%} |')
lines += ['',
'银行70有1,801,617笔付款、2,551笔正例，付款侧活跃账户仅15个，15个均参与过正例付款；这不意味着其全部交易都是洗钱。它在本训练期的付款方式为Cheque、Credit Card、Cash、Bitcoin，没有ACH。前7天/后3天分别有1,802/749笔正例，现象跨两个时间段存在。', '',
'剔除银行70付款后，支票仍有6,658,963笔，但正例只剩75笔，正例率约0.00113%；现金剩6笔正例，信用卡剩0笔正例。这是敏感性分析，未从训练数据中删除任何记录。', '',
'此前8个人工案例未覆盖这组极高频账户，因此不能靠那8例概括支付方式分布。15个账户的明细统计单独保存，建议后续观察它们的对手、金额、币种和时间形态；当前尚未确认其在合成器中的生成机制。', '',
'## 3. 银行差异不只是交易多或ACH多', '',
'银行必须区分付款侧与收款侧。为避免小样本极端比例，下面展示至少10,000笔交易、至少20笔正例的银行；这是阅读筛选门槛，不是风险规则或显著性判据。', '',
'“调整后倍数”＝实际正例数÷预期正例数。其中预期数＝Σ(该银行在某日某支付方式的交易数×全训练数据同日同方式正例率)。它控制了日期与支付方式构成，没有控制金额、币种、客户构成或图结构，也不是因果效应。全期标签参与了这项描述性统计，不能直接回填为训练特征。', '',
'| 方向/银行编号 | 全部交易 | 正例 | 原始Lift | 调整后倍数 | 前7天调整倍数 | 后3天调整倍数 | 正例关联账户数 |',
'| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for role,bank in [('from','70'),('from','272142'),('from','293268'),('to','11'),('to','20'),('to','2408'),('to','1213')]:
    r=b[role].loc[bank]
    lines.append(f'| {"付款" if role=="from" else "收款"} / {bank} | {int(r.total):,} | {int(r.positive):,} | {r.lift:.2f} | {r.observed_expected_ratio:.2f} | {r.discovery_adjusted_ratio:.2f} | {r.internal_validation_adjusted_ratio:.2f} | {int(r.positive_accounts):,} |')
lines += ['',
'例如收款银行20：151笔正例分布在104个收款账户，单个账户最多贡献约5.96%的这些正例；其差异并非由一个账户独占。相反，付款银行272142只有8个正例关联付款账户，头部账户贡献约35.48%，更需要检查账户构成。银行1213在两个时间段均偏高，但倍数变化大，不能说强度稳定。', '',
'再排除所有银行70付款交易，重新估计日期/方式基线，以下收款银行仍然偏高：', '',
'| 收款银行编号 | 剩余交易 | 剩余正例 | 重新调整后倍数 |',
'| --- | ---: | ---: | ---: |']
for bank in ['11','20','2408','1213']:
    r=ex.loc[bank]
    lines.append(f'| {bank} | {int(r.total):,} | {int(r.positive):,} | {r.adjusted_ratio:.2f} |')
lines += ['',
'以上支持“局部银行关联值得继续验证”，并不证明洗钱账户主观偏好这些银行。账户的开户行本来是固定属性，还可能反映账户群、合成生成机制及其交易网络。若想检验主动选择去向，应在同一付款账户的历史中比较其目标银行分布，并控制可选对手与账户活动程度；本轮尚未做这一层匹配。', '',
'## 4. 总体效应与统计边界', '',
f'付款行有{p["bank_stats"]["from"]["bank_count"]:,}个取值，收款行有{p["bank_stats"]["to"]["bank_count"]:,}个取值；类别数很多。全训练期银行与标签的未校正Cramér’s V，付款侧约{p["bank_stats"]["from"]["cramers_v"]:.4f}、收款侧约{p["bank_stats"]["to"]["cramers_v"]:.4f}。整体效应不大，与少数银行存在局部高倍数不矛盾。', '',
'由于高维稀疏银行类别、大量重复账户和图关系依赖，普通卡方p值不适合直接作为保留标准；这里不报告“统计显著就有效”。银行排序属于探索性筛查，存在多重比较和选择偏差。账户正例关联比例也受交易次数影响，不能等同于真实犯罪用户比例。', '',
'## 5. 对特征设计的影响', '',
'- 保留当前付款行、收款行和支付方式作为类别对照，同时比较去掉银行ID与支付方式后的表现。银行编号不作为连续大小处理。',
'- 增加账户历史目标银行数、目标银行集中度、当前目标银行历史份额及新目标银行标志。这比单纯记住某个高风险银行编号更贴近行为。',
'- 支票应与付款账户活跃规模、付款银行和时间行为一起分析；先分别验证银行70与其他来源，不能给所有支票统一高风险解释。',
'- 对银行70这15个高频账户单独报告模型表现和敏感性结果，暂不删除、不认定为错误数据或真实违规机构。',
'- 如果做银行标签率编码，只能用适当的历史/交叉拟合训练信息；本报告的全训练期比例不能直接用作同一批样本的预测特征。', '',
'## 数据表与复现', '',
'- [完整付款行统计](result/train_bank_analysis/banks_from.csv)',
'- [完整收款行统计](result/train_bank_analysis/banks_to.csv)',
'- [支付方式统计](result/train_bank_analysis/payment_train.csv)',
'- [银行70账户、币种与方式明细](result/train_bank_analysis/bank70_account_currency_payment.csv)',
'- [排除银行70付款后的收款行统计](result/train_bank_analysis/banks_to_excluding_from70.csv)',
'- [每日支付方式统计](result/train_bank_analysis/payment_daily.csv)', '',
'来源：冻结Train所含10天正式银层Parquet及静态账户表。scripts/analyze_train_banks.py负责全量统计，scripts/analyze_bank70_sensitivity.py负责来源敏感性拆分。付款侧和收款侧分别覆盖同样的20,318,992筆交易和17,794筆正例；双方计数不能相加当作交易总量。']
(ROOT/'训练集银行与支付方式分析.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Report written; aggregates verified.')
