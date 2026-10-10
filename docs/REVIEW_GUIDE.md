# GPT 阅读入口

1. 先读 [实验状态](EXPERIMENT_STATUS.md)：数据划分、配方、已完成指标和结论边界。
2. [工程历史记录](ENGINEERING_HISTORY.md)保留原README中的按日期进展；其中旧状态只代表当时。再读根目录 `GPTPro交接_GNN交易反洗钱项目_2026-10-02.md`：完整沿革与审计。按日期理解旧的 Test 封存说明；2026-10-10 已授权最终 Test。
3. 本目录 `results/` 保存发布时已完成实验的比较报告和种子结果，便于直接引用数值。种子重复尚未全部结束，不能把单次胜负当作稳定结论。
4. `scripts/gnn_features/model.py` 是 GINE 架构，`v2_adapter.py` 定义 A 的输入扩展，`repeat_common.py` 与 `repeat_train.py` 定义固定配方种子重复。
5. `scripts/gnn_final_test/` 是独立最终 Test 流程；共八个预先指定模型，各用自身 Validation 阈值。没有 Test 阈值寻优或最佳种子选择。

## 数据与模型附件

[2026-10-10 Release](https://github.com/031asa/laudering/releases/tag/research-snapshot-20261010) 包含：

- `completed-models-and-results-20261010.zip`：打包时已完成的 lr_only/A 权重、配置、训练计数与结果；约30 MiB。
- `IBM-AML-HI-Medium-original-data.zip`：未修改的原始 HI-Medium 交易、账户、模式文件及来源哈希；约733 MiB。
- `release_assets.json`：附件字节数和 SHA-256。各包内部还有逐文件 SHA-256 清单。

原始数据由 IBM / ealtman2019 发布，数据许可为 CDLA-Sharing-1.0；来源和原许可链接保存在数据包内。大体积图和特征缓存不上传。GPT 阅读代码及报告不需要下载原始数据和权重。部分 GPT 环境可能无法直接打开大型压缩包，必要时单独附上所需报告。

## 当前最终 Test 状态

2026-10-10 已通过10项工程检查并启动历史归档准备，与种子重复训练独立并行。完整 Test 为6,883,233笔、6,932正例。尚无 Test 指标；正式全量特征核验及真实CUDA推理验收会在流程中执行。09-17以后尾部继续封存。下一阶段是否改架构，等待全部结果后讨论；当前没有启动新架构实验。

