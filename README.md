# GNN 交易反洗钱研究

基于 IBM AML HI-Medium 合成数据，使用 Spark 构建严格历史特征，以双方向两层 GINE 进行交易分类。

## 当前状态

截至 2026-10-10，正在进行 lr_only 与 A（features_only）各三个新种子的完整 Train/Validation 重复实验。lr_only 是 GNN 配方名，并非逻辑回归。原开发种子另外列出。完整范围及已完成指标见 [实验状态](docs/EXPERIMENT_STATUS.md)。

已授权两种配方各四个种子的最终 Test。Test 固定使用对应模型的 Validation 阈值，不用于选择种子或再次调参。当前尚无 Test 结果。树模型实验已取消。

## 目录

- `scripts/gnn_features/`：历史索引、因果特征、图采样、GINE、训练、评估及断点恢复。
- `scripts/`：数据工程入口、工程验收和实验启动脚本。
- `tests/`：数值、时间边界、模型与恢复验证。
- `config/`：冻结日期划分与特征定义。
- `dashboard/template.html`：不含交易数据的看板模板。
- `docs/`：公开的汇总说明。

研究报告和交接文档已包含在仓库。原始合成数据、已完成权重及配置通过 GitHub Release 提供；图和特征缓存、日志与本机环境不进入 Git。请从 [GPT 阅读入口](docs/REVIEW_GUIDE.md) 查看报告顺序、附件及校验说明。

## 环境和入口

数据工程依赖见 `requirements.txt`；GPU 训练使用独立 Python 3.11 环境与 `requirements-gnn-gpu.txt`（PyTorch 2.7.1 CUDA 12.6）；早期 CPU 验收依赖另见 `requirements-gnn-cpu.txt`，不要把三个依赖清单装进同一环境。

这是已有研究工作区的源码发布。数据与已完成模型在 Release，预计算缓存不包含，不能克隆后直接启动完整训练。需按来源哈希校验原数据、遵循原数据许可，再建立银层及冻结索引。Windows Spark 入口还需要本地 JDK 和 Hadoop 本地文件系统兼容 JAR；路径参数见 `scripts/daily_gnn.ps1`。

已有完整工作区的种子重复实验入口：

```powershell
powershell -File scripts/start_gnn_seed_repeats.ps1
```

运行前检查脚本参数与配置。恢复只使用原入口，保留模型、优化器、随机状态和已验证的数据块。源码、配置和输入绑定 SHA；身份变化、非有限数值或 CUDA 不可用时停止，不绕过校验。至少保留 20 GiB 磁盘空间。

## 实验边界

完整 Train 为 20,318,992 笔，完整 Validation 为 4,689,006 笔；固定图采样、批序、一个 epoch。历史特征仅使用当前时刻之前的无标签记录，同刻历史排除。Validation 结果属于开发观察；少量种子不证明统计显著，也不代表重新采图的随机性。最终 Test 结果发布前不声称独立泛化效果或超越论文。
