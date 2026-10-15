# GPT 阅读入口

1. 先读 [实验状态](EXPERIMENT_STATUS.md)：数据划分、配方、已完成指标和结论边界。
2. [工程历史记录](ENGINEERING_HISTORY.md)保留原README中的按日期进展；其中旧状态只代表当时。再读根目录 `GPTPro交接_GNN交易反洗钱项目_2026-10-02.md`：完整沿革与审计。按日期理解旧的 Test 封存说明；2026-10-10 已授权最终 Test。
3. 本目录 `results/` 保存发布时已完成实验的比较报告和种子结果，便于直接引用数值。全部种子重复与最终Test已完成；不能把单次胜负当作稳定结论。
4. `scripts/gnn_features/model.py` 是 GINE 架构，`v2_adapter.py` 定义 A 的输入扩展，`repeat_common.py` 与 `repeat_train.py` 定义固定配方种子重复。
5. `scripts/gnn_final_test/` 是独立最终 Test 流程；共八个预先指定模型，各用自身 Validation 阈值。没有 Test 阈值寻优或最佳种子选择。

## 数据与模型附件

[2026-10-10 Release](https://github.com/031asa/laudering/releases/tag/research-snapshot-20261010) 包含：

- `completed-models-and-results-20261010.zip`：打包时已完成的 lr_only/A 权重、配置、训练计数与结果；约30 MiB。
- `IBM-AML-HI-Medium-original-data.zip`：未修改的原始 HI-Medium 交易、账户、模式文件及来源哈希；约733 MiB。
- `release_assets.json`：附件字节数和 SHA-256。各包内部还有逐文件 SHA-256 清单。

原始数据由 IBM / ealtman2019 发布，数据许可为 CDLA-Sharing-1.0；来源和原许可链接保存在数据包内。大体积图和特征缓存不上传。GPT 阅读代码及报告不需要下载原始数据和权重。部分 GPT 环境可能无法直接打开大型压缩包，必要时单独附上所需报告。

## 当前最终 Test 状态

2026-10-11全部八模型Test完成且核验通过。详见[最终报告](results/final_test/最终测试结果.md)与[当前状态](EXPERIMENT_STATUS.md)。Release新增final-models-and-results-20261011.zip，含全部八模型权重及最终报告，SHA见final_release_assets.json。旧20261010快照保留，原始数据包不变。此处记录原最终Test交付；后续尾部追加分析见下文，部署不替换。

## 尾部追加分析（已完成）

用户随后授权09-17..28的6,987笔尾部数据追加评估。阅读[尾部报告](results/tail_test/尾部追加评估结果.md)、[完整比较](results/tail_test/comparison.json)与[核验](results/tail_test/final_audit.json)。实现位于scripts/gnn_tail_test，复用全部8模型及原Validation阈值；不训练、不寻优。原Test、尾部、合并分别报告。尾部正例比例异常高，合并F1约69%—70%属于数据构成改变后的事后敏感性结果，不能替代原Test或证明模型改善。
