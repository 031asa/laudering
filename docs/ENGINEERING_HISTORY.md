# 基于 Spark 与图神经网络的交易反洗钱检测项目

> **当前状态（2026-10-10）**：正在进行 lr_only 与 A 各三个新种子的完整训练/Validation 重复实验。用户已授权两配方各四个种子的最终 Test，Test 结果尚未产生。最新实验范围、已完成指标和发布内容见 [实验状态](docs/EXPERIMENT_STATUS.md)。下方保留按日期形成的工程记录，早期“尚未训练”等描述只代表当时状态。原始交易、缓存、权重和本机运行结果不包含在此仓库中。

面向大规模交易风险筛查，使用 IBM AML HI-Medium 合成数据，计划以交易边分类识别高风险交易，并分析局部账户关系。

当前路线（2026-09-15）：银层交易 → 账户历史N、交易属性E、当前交易比较R → 有向历史图 → 双方向两层GINE → 交易分类。候选特征文档只作指标来源追溯。

## GNN首批特征与按需查询

2026-09-20一小时节点缓存试验完成：9月4日12:00–13:00的32,010笔目标、525个固定采样批次，去重得到639,056个账户／时刻状态。图与N一次准备约169秒，整小时缓存读取约24秒；12个批次与原查询方式对照通过。仅限一小时，未扩展全训练集。见 [节点缓存试验](result/gnn_hour_cache_trial/节点缓存试验.md)，入口 `scripts/gnn_features/prepare_hour.py`。

2026-09-20小模型验收：复用 `C:/anaconda/envs/d2l-zh/python.exe` 的PyTorch 2.4.1+cpu，双方向两层GINE在96笔固定训练样例上完成60次更新，各输入路径有梯度、模型保存恢复一致。CPU单步训练中位约82毫秒；不含组图，尚无泛化效果或GPU验收。见 [GNN小批次训练验收](result/gnn_model_smoke_v1/GNN小批次训练验收.md)。模型测试入口：`python -m unittest discover -s tests -p test_gnn_model.py -v`（使用上述独立环境）。

2026-09-20：全训练集20,318,992笔交易特征已通过逐日验收；CPU训练批次加载器已接通已保存的E/R、两跳历史图采样和按需N查询。32笔真实目标的首次组批约1.1–1.3秒，N查询仍为主要瓶颈；尚未启动GNN训练。见 [训练批次接口验收](result/gnn_loader_benchmark_compact_v1/训练批次接口验收.md)，入口 `scripts/gnn_features/loader.py`。

2026-09-19整日试跑完成：9月4日830,607笔交易，4并发Spark、8账户分区，计算及落盘约13分45秒，核验约58秒；首次历史索引构建另需约2分24秒。一小时完整对照、全天24小时分层抽样及17项代码测试通过。详见 [整日实测](result/gnn_day_20220904/整日实测.md)。入口 `scripts/daily_gnn.ps1`，最终54列R位于 `result/gnn_day_20220904/features/`；N仍按需查询，尚未开始GNN训练。

2026-09-19新增账户分区及滚动复用实验：同一小时32,010笔交易，全部N/R对照通过。单进程滚动版约108秒，四并发Spark持续会话约66秒，均含合并核验；冷启动耗时另计。详见 [分区与滚动统计实测](result/gnn_分区与滚动统计实测.md)。入口为 `scripts/benchmark_gnn.ps1`，暂为单机共享索引实验，尚未扩到全训练期。

实现入口为 `scripts/gnn_features.ps1`。本阶段使用现有 Python/PyArrow/NumPy 建立精确时间索引及查询基准，不将本次耗时称为 Spark 作业性能。银层与冻结日期不变；正常和洗钱交易都参与无标签历史统计。

- `data/gnn_v1_dev/graph`：账户映射、交易边档案及外部交易双向时间索引。
- `data/gnn_v1_dev/history_index`：按账户、方向、时间排序的磁盘数组；金额保留六位小数的精确缩放整数，聚合防止整数溢出。
- `data/gnn_v1_dev/metadata/feature_manifest.json`：N/E/R列序、来源、公式、窗口、币种口径、缺失及模型变换。
- `result/gnn_v1_dev/target_features.parquet`：9月4日12:00–13:00目标交易的54列R及覆盖标志；交易ID连接E。
- `result/gnn_v1_dev/target_audit.parquet`：缺失原因和比率分子分母；金额支持值采用百万分之一货币单位的精确整数。
- `result/gnn_v1_dev/verification.json`、`samples.json`：实际运行验收与三类真实交易样例。

N不预生成全量快照表：通过 `(node_id, cutoff_ts)` 查询，缓存限同一截止时点。历史索引包含自转账供all统计使用，外部邻接索引排除自转账。样例图批次只验证CPU数据接口，本阶段未启动GNN训练或前向/反向验收。

在项目根目录运行：

```powershell
./scripts/gnn_features.ps1 -Job test
./scripts/gnn_features.ps1 -Job build
./scripts/gnn_features.ps1 -Job query -NodeId 0 -Cutoff '2022-09-04T12:00:00'
./scripts/gnn_features.ps1 -Job run -Output 'result/gnn_v1_dev_rerun'
./scripts/gnn_features.ps1 -Job verify -Output 'result/gnn_v1_dev_rerun'
```

构建可复用相同版本的完整档案；结果目录拒绝覆盖。开发档案仅覆盖截至9月4日13:00前，超出历史覆盖的查询报错，不能拿截断数据冒充完整历史。类别词表按冻结训练期9月1–10日拟合；银行内部身份键与模型UNK类别分开，避免不同未知银行被统计成同一家。

以下铜层章节保留早期转换记录，其“本阶段”描述指当时的铜层转换，不代表当前工程状态。

## 当前功能与边界

已下载 HI-Medium 原始数据；本阶段实现 CSV 到铜层 Parquet 的无损字段转换与独立全量核验。转换执行器为 Python CSV + PyArrow 21.0.0，本次耗时不能作为 Spark 性能实验结果。后续 Spark 可以直接读取这些 Parquet。

交易主表保留一笔交易一行，仅按原始位置区分付款/收款列。账户收支 OUT/IN 双行流水暂不生成，后续按需由主表派生。没有清洗、去重、时间解析、币种换算、时间分区、账户 ID 统一、训练集划分、特征计算或建模。

原始业务字段全部保存为非空类型的字符串；CSV 空字段保留为空字符串，不转为 null。银行编号前导零、金额原始小数写法、空白、重复行均保留。结构错误会使转换失败，不静默跳过。这里的“无损”指每条记录的原始字段值；CSV 引号和换行编码等文件字节由 raw 原件保留，不以 Parquet 复现。

## 输入与输出

| 输入 | 输出 | 记录含义 |
| --- | --- | --- |
| data/raw/HI-Medium_Trans.csv | data/bronze/transactions/part-*.parquet | 一笔交易一行 |
| data/raw/HI-Medium_accounts.csv | data/bronze/accounts/part-*.parquet | 一条原始账户记录一行 |
| data/raw/HI-Medium_Patterns.txt | 保留原始文件 | 本阶段不解析模式文件 |

每个输出目录带有 `_manifest.json`（Schema 映射、来源、分片与耗时），成功核验后另有 `_verification.json`。下划线文件为元数据，数据扫描读取 `part-*.parquet`。

### 交易字段映射

| 原列位置 | 原列名 | 铜层列名 | 类型 |
| --- | --- | --- | --- |
| 1 | Timestamp | timestamp | string |
| 2 | From Bank | from_bank | string |
| 3 | Account | from_account | string |
| 4 | To Bank | to_bank | string |
| 5 | Account | to_account | string |
| 6 | Amount Received | amount_received | string |
| 7 | Receiving Currency | receiving_currency | string |
| 8 | Amount Paid | amount_paid | string |
| 9 | Payment Currency | payment_currency | string |
| 10 | Payment Format | payment_format | string |
| 11 | Is Laundering | is_laundering | string |

新增 `transaction_id`、`source_file`、`source_record_number`。交易 ID 是完整源文件 SHA-256 加冒号加记录序号；序号从 1 开始，不含表头，指 CSV 逻辑记录而非物理行号。它独立于 Parquet 分片大小，重复内容的原始记录仍拥有不同 ID。更改源文件会改变 ID，不能把这个 ID 当作跨数据版本的业务交易 ID。Parquet 不保证读取顺序，需要恢复原序时按来源和记录序号排序。

账户表将列名映射为 `bank_name`、`bank_id`、`account_number`、`entity_id`、`entity_name`，新增 `account_record_id` 和相同来源字段。账户记录 ID 仅标识原始行，不代表统一账户节点 ID。

## 快速开始

本机已验证的运行环境：`C:\anaconda\python.exe`，已有 PyArrow 21.0.0，本轮未安装新依赖。可复现依赖见 requirements.txt；其他机器可在自己的 Python 3.11+ 环境安装该依赖。

在项目目录执行：

```powershell
Set-Location -LiteralPath 'C:\Users\28118\Documents\ChatGPT\图算法完整项目'
& 'C:\anaconda\python.exe' scripts/bronze.py convert
```

转换命令拒绝覆盖任何非空目标目录；已经生成数据时不重复转换，直接独立核验：

```powershell
& 'C:\anaconda\python.exe' scripts/bronze.py verify
```

可加 `--table transactions` 或 `--table accounts` 单独处理一张表。转换和核验分别输出 `result/bronze_convert.json`、`result/bronze_verify.json`。转换结束时状态是 `converted_pending_verification`，全量核验成功后才更新为 `verified`。

独立核验会重新读取原始 CSV 和所有 Parquet，逐字段比较全部记录，检查 ID、记录顺序、分片清单与行数，并核对源文件 SHA-256。小样例自动检查可独立执行：

```powershell
& 'C:\anaconda\python.exe' -m unittest discover -s tests -v
```

## 目录设计

```text
图算法完整项目/
├── data/
│   ├── raw/                       # 官方原件、下载来源与校验记录
│   ├── bronze/
│   │   ├── transactions/          # 无损交易 Parquet
│   │   └── accounts/              # 无损账户 Parquet
│   ├── silver/
│   │   ├── transactions/          # 预留：清洗交易
│   │   └── accounts/              # 预留：统一账户表
│   ├── features/
│   │   ├── account_features/      # 预留：账户历史特征
│   │   └── pair_features/         # 预留：账户对历史特征
│   └── graph/
│       ├── nodes/                 # 预留：图节点表
│       └── edges/                 # 预留：有向多重交易边
├── result/                        # 转换与核验结果
├── scripts/bronze.py              # 转换/核验命令入口
├── utils/bronze.py                # 无损转换与核验实现
├── tests/test_bronze.py           # 边界样例与错误检测
├── requirements.txt
└── README.md
```

## 数据来源与下载记录

- [IBM 官方项目](https://github.com/IBM/AML-Data)
- [官方指向的 Kaggle 数据集](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
- 数据许可：CDLA-Sharing-1.0；数据为合成交易。

2026-09-12 下载三个原始文件，均通过官方大小与 MD5 比对；合计 3,179,071,636 字节（约 3.18 GB）。`data/raw/download_manifest.json` 记录 MD5、SHA-256、来源和验证时间；`source_integrity.json` 保存官方存储校验信息。

本地 Git main 分支未提交、未设置远程、未推送；原始数据、铜层数据和运行结果默认不纳入 Git。文件完整性核验不代表已完成数据质量审计。

## 本次全量运行结果（2026-09-12）

| 数据表 | 原始记录数 = Parquet 记录数 | Parquet 大小 | 转换时间 | 独立全量核验 |
| --- | ---: | ---: | ---: | --- |
| transactions | 31,898,238 | 893,274,364 字节 | 404.835 秒 | 通过，285.511 秒 |
| accounts | 2,087,786 | 54,690,353 字节 | 17.682 秒 | 通过，12.925 秒 |

两张表 Parquet 合计 947,964,717 字节（约 948 MB），包含原始字段、记录 ID 和来源字段。账户表行数是原始记录数，尚未去重或确认唯一账户数。

全量核验比较了 361,319,548 个原始字段值，所有值均一致；同时验证原始文件 SHA-256、行数、顺序、ID 和来源信息。4 项独立小样例测试均通过，覆盖重复行的独立稳定 ID、前导零、金额格式、空字符串、含逗号/换行的字段、损坏检测和拒绝覆盖。

实际运行报告：[转换结果](result/bronze_convert.json)、[全量核验结果](result/bronze_verify.json)。这不是数据质量审计；未判断异常日期、重复交易的业务含义或确定 Train/Validation/Test。

## 基础审计与清洗规则 v1（2026-09-12）

已使用本机既有 Spark 4.1.2 环境完成两张铜层表的全量基础审计。当前仍未生成银层数据。

- [统一清洗规则](清洗规则.md)：银行编号规范化、精确金额类型、日期尾部待审、重复交易保留等处理口径。
- [本次中文审计报告](result/cleaning_audit/基础审计报告.md)
- [完整机器可读统计](result/cleaning_audit/audit.json)
- [每日交易与标签统计](result/cleaning_audit/daily_counts.csv)
- [汇总一致性核验](result/cleaning_audit/consistency_checks.json)

关键发现：银行编号去前导零后全部账户关联成功；金额最多 6 位小数；20 组重复内容的交易先保留；9 月 17—28 日尾部仅 6,987 笔交易、正例率约 58.45%，保留待审，不直接当作普通测试集。

重跑只读审计（PowerShell，输出 audit.json）：

```powershell
& .\scripts\audit.ps1
```

脚本复用 `C:\Users\28118\Documents\hive_spark_project` 的既有 Python、Spark 和 JDK，只设置当前进程环境。可用 `-RuntimeRoot` 指定具备相同内部目录结构的运行环境。Windows 文件系统兼容依赖及激活路径已记入技能本机环境说明册；没有自动安装依赖。审计使用临时 shuffle/warehouse 目录并在结束后清理。读取中文数据路径使用普通 Unicode 路径，不传入百分号编码的 file URI。

本次独立汇总核验确认：所有日期/类别分组的行数和正例数与全表一致，且两张表的行数与铜层清单一致。金额百分位是按币种计算的近似值，其余本报告计数均为全量精确统计。

## 完全重复去重与官方说明核查（2026-09-12）

按用户明确要求，清洗规则已更新为 v1.1：比较全部 11 个原始字段，包括时间戳、双方银行和账户、两侧金额及币种、支付方式和标签；每组保留来源序号最小的一条。源 CSV 与铜层不改动。

- 去重前：31,898,238 条；20 组、每组 2 条；从派生表排除 20 条。
- 去重后：31,898,218 条；洗钱正例仍为 35,230 条，移除的 20 条均为标签 0。
- 输出：[仅完成去重的派生交易表](data/silver/transactions_deduplicated/)。银行编号、时间和金额仍保留铜层字符串，其他银层标准化步骤尚未执行；账户表未改动。
- 可追溯记录：[保留/移除 ID 映射](result/deduplication/decisions.json)、[输出核验](result/deduplication/result.json)。
- 115 个未改动分片复制后 SHA-256 一致；13 个改写分片逐列与原表扣除指定记录后的值及元数据一致。总行数和标签数量核对通过。

执行入口（输出非空时拒绝覆盖）：

```powershell
& .\scripts\audit.ps1 -Job find_duplicates
& 'C:\anaconda\python.exe' scripts/materialize_deduplication.py
```

Spark 4.1.2 用于精确分组与定位；PyArrow 21.0.0 用于局部重写和校验。之前的基础审计报告记录的是去重前铜层数据，其“先保留重复”是历史决定，当前以《清洗规则.md》为准。

官方资料：[IBM README 本地副本](data/reference/IBM_AML_README.md)、[Kaggle 发布说明本地副本](data/reference/Kaggle_DATASET_DESCRIPTION.md)、[官方来源与时间切分判断](数据说明与切分判断.md)。发布者说明日期范围为主要活动期、明确承认期外交易，并推荐按时间顺序 60/20/20 划分。具体尾部成因仍未核实；未执行时间筛选或训练集划分。

## 按日分区（2026-09-12）

已将去重后的交易写入 [按日分区交易表](data/silver/transactions/)，目录格式为 `transaction_date=YYYY-MM-DD/`。2022-09-01 至 2022-09-28 共 28 个连续日期，一日一个逻辑分区，同一天的交易只出现在对应日期目录中；同一目录内允许多个 Parquet 文件。

全量核验通过：输出 31,898,218 条，正例 35,230 条；日期与目录不一致的记录为 0。每天的行数、正例数、原始列内容校验和及来源序号范围均与分区前一致。标签缺失或非法直接排除，本次命中 0 条。详见 [分区核验报告](result/date_partitioning.json) 和《清洗规则.md》v1.2。

本步保留全部日期，未划分训练、验证、测试集。后续按完整日期目录选取连续时间范围，不能把同一天拆到不同集合。银行编号、金额和原始时间列的类型标准化尚未执行。

执行入口（输出非空时拒绝覆盖）：

```powershell
& .\scripts\audit.ps1 -Job partition_dates
```

## 必要字段标准化完成（2026-09-12）

当前分析入口为 [标准化交易表](data/silver/transactions/) 和 [标准化账户表](data/silver/accounts/)。交易两端银行编号及账户表银行编号统一去前导零并保持字符串；账户编号保持原样。时间转为 `TIMESTAMP_NTZ`（不附加或转换时区），两侧金额转为 `DECIMAL(38,6)`，标签转为 INT，日期分区列为 DATE。币种、支付方式、实体信息和溯源字段不改写。

全量运行和读回核验通过：31,898,218 条交易、35,230 条正例、2,087,786 条账户；保留 2022-09-01 至 2022-09-28 的 28 个连续日期分区。非法标签排除数、日期目录错误数、账户重复键数及交易账户未匹配键数均为 0。每日标签计数与标准化内容校验和一致；内容校验和比较不等同于精确集合比较。转换前已检查时间格式、金额小数位数与范围，避免静默舍入。

独立使用 PyArrow 读回了来源记录 959992：银行 `026 → 26`，与铜层账户表的 `26 / 81A4B0100` 匹配；金额为 decimal128(38,6)，标签为 int32，时间不带时区。银行函数的 `020`、`000`、`0`、`20` 边界样例也通过检查。

详见 [标准化报告](result/standardization.json) 和 [清洗规则 v1.3](清洗规则.md)。原分区表保留于 `data/silver/transactions_before_standardization/`，仅去重中间表仍在 `transactions_deduplicated/`。分析应读取正式交易目录，不能将整个 silver 目录合并读取。未生成额外质量提示列，也未划分训练、验证、测试集。

执行入口（只接受标准化前输入，拒绝覆盖已发布输出）：

```powershell
& .\scripts\audit.ps1 -Job standardize
```

本入口遇到未预期的非法时间、银行或金额时会停止发布、保留原表，并在临时输出目录记录失败统计；尚未实现未来异常数据的自动隔离表。缺失或非法标签按约定直接排除。

## 数据集划分已固定

已按完整日期完成逻辑划分，见 [数据集划分](数据集划分.md)、[冻结配置](config/dataset_split.json) 和 [逐日分配清单](config/date_assignments.csv)。发现期为 09-01～09-07，内部验证期为 09-08～09-10，两者组成 Train；模型验证集为 09-11～09-13，最终测试集为 09-14～09-16，尾部观察集为 09-17～09-28。28 个日期目录覆盖完整且互不重叠，复用现有 Parquet，不复制数据。独立入口为 `scripts/split_dataset.py`，只执行划分与核验，不运行 EDA。按用户最新要求，本轮暂停 EDA，尚未形成特征发现报告。

## 发现期统计看板

已生成 [单文件 HTML 看板](dashboard/发现期数据分析看板.html)，直接用浏览器打开；不依赖联网。支持发现期七天内全部28个日期区间，展示币种/支付行为、分币种金额、账户频次与实体关联规模。仅使用发现期交易；完整账户元数据用于身份和登记规模。全量统计与金额抽样估计在页面内分别标明。见 [生成和统计说明](dashboard/README.md)、[汇总核验及交互检查](result/dashboard/verification.json)。日期划分不变，尚未进行内部验证或训练模型。

## 外部反洗钱指标研究

新增 [反洗钱候选特征矩阵 v1.2](反洗钱候选特征矩阵.md)：沿用原16族编号扩展至55族行为候选和8族基础字段，加入银行偏好，并按基础行为T、图增强G登记在 [特征分组目录](config/feature_groups.json)。当前为设计文档，尚未批量计算或验证。


[反洗钱指标复用研究](反洗钱指标复用研究.md) 汇总监管与原始研究来源、16族候选指标、本项目可计算性、历史窗口定义、缺失字段和验证顺序。当前仅完成研究与设计，未新增特征计算或改动看板。Snap ML GFP 官方文档注明不支持 Windows，因此先复用定义，不直接接入库。

## 发现期流水整理

后续按用户要求分析了完整训练期（09-01至09-10）的银行与支付方式关联，见 [训练集银行与支付方式分析](训练集银行与支付方式分析.md)。报告包含日期/支付方式调整、银行70付款来源敏感性分析和前7天/后3天分段统计，未读取主验证集或测试集。完整训练期已用于探索，内部验证期不再视为未观察数据。

已整理发现期全部11,360笔洗钱标签交易，以及8个选定账户的完整发现期收付背景。案例合计662行账户视角，对应630笔独立交易，包含标签0与1。见 [流水阅读指引与真实案例](发现期流水阅读指引.md) 和 [Excel流水](outputs/discovery-ledger-20260915/发现期洗钱标签交易流水.xlsx)。仅用于回顾性EDA，案例不是随机样本，账户视角合表不能直接当作独立交易计数。

提取入口为 `scripts/extract_discovery_cases.py`；两轮发现期扫描、每账户完整性及来源ID核验见 [提取核验](result/discovery_cases/verification.json)。CSV/Parquet保留精确金额和来源信息；不读取验证、测试或尾部交易，不改动现有看板。

## GNN GPU 环境（2026-09-26）

项目独立环境为 `.venv-gnn-gpu`，基于已有 Python 3.11.15，使用 PyTorch 2.7.1+cu126；对应本机 RTX 4050 Laptop 6 GB。CUDA 轮子来源为 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/#v271)，依赖固定在 `requirements-gnn-gpu.txt`。不修改 Anaconda base、原 CPU 环境或 Spark 环境，也不修改永久 PATH。当前 GINE 使用原生 PyTorch，不需要 PyG 扩展。

首次重建环境（已有环境无需重复执行）：

```powershell
& 'C:\anaconda\envs\recbole311\python.exe' -m venv .venv-gnn-gpu
& .\.venv-gnn-gpu\Scripts\python.exe -m pip install --upgrade pip --index-url https://pypi.org/simple
& .\.venv-gnn-gpu\Scripts\python.exe -m pip install -r requirements-gnn-gpu.txt --index-url https://pypi.org/simple --resume-retries 10
```

真实 Train 小批次 GPU 验收（一个命令，默认生成新的结果目录）：

```powershell
& .\scripts\check_gnn_gpu.ps1
```

该入口使用已保存的三个批次、96 笔目标交易，执行60次参数更新和检查点读回；默认明确要求 CUDA，不可用即报错。仅作环境与计算链路验收，不启动全量训练，也不接触验证/测试集。性能计时同步 GPU，排除历史特征计算；显存峰值仅代表这些小批次。

2026-09-26 实测通过：GPU 参数更新60步、全部 N/E/R 数值输入路径有梯度、检查点读回在 float32 容差内一致；显存分配峰值约78.6 MiB、预留峰值118 MiB。带数值检查的训练步中位数37.3 ms、前向中位数6.37 ms，均为数据已在 GPU 上的小批次计时，不能直接外推全量训练吞吐。见 `result/gnn_gpu_smoke_v1/acceptance.json` 与同目录完整依赖记录。`pip check` 通过；新环境中的模型、加载器和滚动统计相关9项测试通过。

原 CPU 复现方式仍保留：使用 `requirements-gnn-cpu.txt` 对应环境，设置 `PYTHONPATH=scripts` 后执行 `python -m gnn_features.smoke_model --device cpu --output <新结果目录>`。Parquet 特征准备目前仍是 CPU 实现，本次 GPU 环境配置不会自动迁移其计算。

## 节点缓存 CPU／GPU 对照（2026-09-26）

已完成同一训练期小时（2022-09-04 12:00–13:00）的三组试验：32,010笔目标交易、525个固定采样批次、639,056条账户时间状态。将41项N中的23项基础指标改为批量边界查找和整数累计统计，18项复杂指标继续由4个CPU进程维护。CPU不重复计算已拆出的指标；既有冻结引擎和已验证数据未修改。

| 方案 | 节点计算、合并与Parquet写入实测 |
| --- | ---: |
| 原CPU逐账户实现 | 280.38秒 |
| 批量CPU＋CPU复杂统计 | 84.19秒 |
| GPU基础统计＋CPU复杂统计 | 86.79秒 |

基础批量统计本身（包含GPU传输）为CPU 2.81秒、GPU 1.40秒，但复杂统计仍占约76–80秒，因此GPU没有显示出总耗时收益。该样本优先采用批量CPU思路；GNN模型训练继续用GPU。GPU特征阶段分配显存峰值约185.3 MiB。

三组全部63列模型节点输入与原已验收缓存精确一致，Parquet读回精确一致；新增CPU/GPU边界与空历史4项测试通过。共用采样阶段另行重跑，525个采样计划和节点时间键均与原结果一致：加载14.16秒，采样、去重及图计划Parquet写入49.21秒。分阶段相加约为原CPU343.75秒、批量CPU147.56秒、混合160.15秒（混合包含约9.99秒CUDA冷启动）；这是阶段测量相加，不是三次完整冷运行。每组只测一次、顺序固定、未清系统文件缓存，小差异不能视为稳定性能差异，也不将倍数直接外推全训练集。

代码入口：设置 `PYTHONPATH=scripts` 后，以 `.venv-gnn-gpu/Scripts/python.exe -m gnn_features.hybrid_cache_trial --output <新目录>` 复现。结果在 `result/gnn_hybrid_cache_trial/benchmark.json`；三个子目录保留试验Parquet和中间数据。当前只完成一小时试验，未启动全量缓存或正式训练。

## 全训练集 Parquet 缓存生成（2026-09-26 已启动）

正式目录为 `data/gnn_train_cache_parquet/`。仅处理冻结Train的2022-09-01至09-10，共20,318,992笔目标交易；采用上述批量CPU算法和4个工作进程，不运行模型训练。

- 按 `transaction_date=日期/hour=小时/block=块号/` 保存；每块最多512个批次，每批最多64笔且交易时间戳一致，块不跨小时。高交易量小时拆成多个块以控制内存和临时空间。
- `nodes.parquet`：账户编号、历史截止微秒和63列模型节点输入（41项变换后的N、19项有效性标志、3项历史覆盖标志），不是未变换的原始统计表。
- `plans.parquet`：批次编号、截止时间、目标交易及历史节点/边编号列表。银行类别、交易E、关系R和标签复用原表，不复制。
- 数据临时文件也用Parquet，只保留一个活动块；核验后删除临时分区，仅原子发布两份正式Parquet及manifest。不同块可能有同一账户同一时刻状态，按块读取即可；不全局展开成全部账户×全部时刻。
- 每块核对全量写入读回、图时间边界和身份完整性，并最多抽256个节点状态与原引擎比较；不声称逐行重算全部原始N。通过后才发布，恢复时校验已发布块的输入、配置和Parquet哈希。
- `config.json`冻结输入、代码、运行环境、采样种子和训练日期。`status.json`记录实际运行pid、阶段、已验证目标数、正式Parquet字节数；启动器pid可能与实际Python pid不同。低于20GiB剩余磁盘会在下一块前停止，不覆盖已验证块。

启动或从检查点恢复（隐藏后台进程，不另加守护程序；重复运行由进程锁拒绝）：

```powershell
& .\scripts\start_gnn_cache.ps1
```

查看进度：读取 `data/gnn_train_cache_parquet/status.json`；失败细节见同目录最新 `run_*.stderr.log`。完成标志为状态及根manifest均为`completed`，并有10天逐日核验清单及目标总数20,318,992。先前约12.6GB的全量缓存体积仅为一小时样本外推，实际以状态与完成清单为准。

上线前7项相关测试通过，第一块32,768笔已完成真实核验；随后后台恢复已跳过已验证首块并继续生成。已有半小时检查任务已改为检查本缓存目录，正常时不重复通知，失败或完成时通知。

## 全训练集 GPU 初版训练

全量缓存已于2026-09-27完成：797块、20,318,992笔，正式Parquet共8,934,266,085字节。
正式模型入口为 `gnn_features.train_model`，与特征生成入口 `gnn_features.train` 分开。
模型只使用2022-09-01至09-10的Train，在CUDA上按时间顺序训练1个epoch；主验证集和测试集不参与。

首次启动前，在项目GPU环境运行实际缓存、模型更新和断点恢复验收：

```powershell
$env:PYTHONPATH = 'scripts'
& .\.venv-gnn-gpu\Scripts\python.exe -m gnn_features.train_model --mode acceptance
```

也可以直接启动或恢复隐藏后台任务；首次运行会先完成上述验收，通过后在同一进程中重新初始化模型并开始正式训练：

```powershell
& .\scripts\start_gnn_training.ps1
```

结果位于 `result/gnn_train_gpu_v1/`。`status.json` 的 `pid` 是实际Python进程，
`committed_targets` 是已保存检查点的样本数，`processed_targets` 还包括当前未提交块；
中断后从最近完整块恢复，未提交块从该块开头重放，对已保存模型不重复累计。
只保留最近和上一份恢复检查点及最终模型。配置冻结已验证输入、代码和训练超参数；
非有限损失/梯度、身份错配或CUDA不可用均报错停止。没有另加守护程序。

模型沿用双向两层GINE、width=64、dropout=0.1，AdamW学习率0.001、weight_decay=0.0001、
梯度裁剪5、种子20260920。BCE正例权重依据完整Train的核验计数计算，不使用小样本权重。
训练逐块复用Parquet节点状态和固定采样图，交易E、关系R和标签读取既有已核验数据。
最终 `training_result.json` 和 `全训练集初版训练结果.md` 仅报告训练过程，不能替代独立验证效果。

## 第一轮 GNN 调参

使用完整 Train（2022-09-01至10）训练，完整 Validation（09-11至13，共4,689,006笔）选参数；
Test（09-14至16）不参与。先在独立目录扩展历史归档并生成验证 R、Parquet 节点缓存和固定采样图，
再评估已有初版及两个预声明候选。候选学习率均为0.0003，正例权重分别为完整Train负正比和该比值的平方根，
其他设置保持相同，从同一种子重新训练完整Train一轮。按全量验证AP排名，ROC-AUC仅用于同分；
验证集选择的阈值不等于独立测试效果，不直接横向比较不同正例权重的训练BCE。

```powershell
& .\scripts\start_gnn_tuning.ps1
```

单个顺序流水线的状态在 `result/gnn_tuning_v1/status.json`；长阶段的详细进度由其中 `child_status` 指向，
日志由 `log` 指向。中断后使用相同入口继续，各阶段保留已核验产物和检查点，原Train和初版模型保持只读。
新归档为 `data/gnn_v1_validation`，验证R为 `result/gnn_validation_v1`，
验证节点/采样缓存为 `data/gnn_validation_cache_parquet`，正式特征缓存与预测均为Parquet。
归档仍使用必要的NumPy历史检索索引，未知原始币种保持独立身份，送入模型的未知类别按Train词表映射为UNK。
结束后在 `result/gnn_tuning_v1` 输出 `comparison.json`、`best_model.json` 和 `调参结果.md`。


## 第三轮：金额与时间分布统计GNN

独立入口为 `scripts/start_gnn_tuning_v3.ps1`，顺序执行四方向24小时分布侧表、金额GNN、金额＋时间GNN及对照报告。Train固定20,318,992笔，Validation固定4,689,006笔；Test封存，v1/v2不改。两个GNN分别为N63/R221和N63/R277，各重新初始化后完整训练1个epoch。

分布统计为同币种、严格更早的外部交易，不使用历史标签。

```powershell
& .\scripts\start_gnn_tuning_v3.ps1
```

首次启动要求 `result/gnn_tuning_v3/engineering_tests.json` 和 `early_gpu_acceptance/acceptance.json` 均通过且与当前源码绑定。正式侧表完成后，还会进行侧表绑定的GPU验收。

进度看 `result/gnn_tuning_v3/status.json` 及其中的 `child_status`；分块与模型/优化器/RNG检查点保留。普通中断使用原入口恢复；显式暂停后须用户要求继续才使用 `-Resume`。身份错配、非有限值、CUDA不可用、磁盘低于20GiB均停止。

最终输出 `comparison.json`、研究用途 `best_model.json` 和 `调参结果.md`，并更新GPTPro交接文档。研究排名按全量Validation最大F1、AP、P80召回；业务达标条件单列，不替换冻结lr_only。Validation已参与诊断，单种子提升不代表统计显著，也不代表超过论文Test基准。
