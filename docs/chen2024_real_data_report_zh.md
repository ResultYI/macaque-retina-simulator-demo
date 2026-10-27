# Chen2024 真实公开来源数据：手动 ZIP 后的核验与阻塞报告

2026-10-10 已批准门禁勘误：作者复现模式不再用瞬时50k作拒绝条件；默认工程封套保留，正值高峰记录不含漂白的生理适用性警告。Fig.3/Fig.4及作者kernel不支持普遍瞬时hard cap，也不支持机械套用全段均值上限。真实负值及预历史/相位仍未解，本轮真实模型对照未运行。见 docs/chen2024_gate_correction_report_zh.md。

MAT/README 的官方字节身份已核验、结构已读，全部六个 primate cone 记录已清点并展示。**固定模型来源数据对照未运行**：刺激含未解释的负光强，缺失预刺激历史和明确相位/预处理依据。≥50k 点数不再作为作者复现拒绝理由。本报告更新此前下载阻塞；不把真实文件取得写成生理复现成功。

## 真实资产

原 ZIP 保留且未改动：`C:\Users\PC_User\Downloads\doi_10_5061_dryad_q2bvq83vg__v20241101.zip`。ZIP 大小 41492089 bytes，SHA256 `c7e39748f11cf2903167968b610f360844ce4a91254d14823b8953572a878de6`。仅含 Figure2-SourceData1.mat、README.md 两个普通文件。导入工具拒绝绝对路径、盘符、..、反斜线、重复成员及 symlink；不执行包内代码，只在新项目 data/source/ 下写入核过哈希的目标文件。

| 原文件 | 实际字节数 | 实际 SHA256（与官方完全一致） |
|---|---:|---|
| Figure2-SourceData1.mat | 41490462 | `6724e2edd8e8708456f32900a14efae88192a5ba597365046d0031eb543d1331` |
| README.md | 1357 | `f25cb72716327f5fc08cfe26cc423bc3364f8f331900dd1972eeb664c1911d03` |

路径：data/source/chen2024_dryad_v3_325663/。来源包路径、CRC/member 属性及提取清单见 reports/chen2024_archive_inventory.json；真实完整字段清单见 reports/chen2024_data_inventory.json。

官方论文已读取并保存：`data/source/chen2024_dryad_v3_325663/elife-93795-v1.pdf`，3298490 bytes，SHA256 `78409a7d5f1246e99fae149147090f11d4e94982e4eb724ddb4f49aa5d1a6966`。[官方 PDF](https://cdn.elifesciences.org/articles/93795/elife-93795-v1.pdf)。历史仓库树与 README 另保存为 data/source_snapshots/chen2024_1bcf7c9a/repository_tree.json 和 repository_README.md，README 已核 Git blob 6811443815e57ce113a08af5bb70ef9617b6c9bf。未重新尝试旧 Dryad 403 下载路径。

## 实际 MAT 结构与单位

| 变量 | MATLAB shape/class | 用途 |
|---|---|---|
| MouseConeResponses | [1, 6] / cell | 只清点结构，不替代 primate cone、不运行模型 |
| MouseRodResponses | [1, 8] / cell | 只清点结构，不替代 primate cone、不运行模型 |
| PrimateConeResponses | [1, 6] / cell | 本模块审计全部记录 |
| PrimateRodResponses | [1, 6] / cell | 只清点结构，不替代 primate cone、不运行模型 |

PrimateConeResponses 为 1×6 cell，每个元素内为 1×1 struct，只含 AdaptingStepStim 与 AdaptingStepResponse，两个字段均为 1×80000 double/float64，所有数值有限。字段顺序可不同，按同一 struct 中的字段名读取，不按不同 cell 同行号配对。每条刺激由其自己的 struct 读取，额外逐值核验六条刺激相同，不据此假定其他变量的细胞对应。

单位依据为实际包内 README 的 cone 段落，并经原文与历史 kernel 调用约定核对：刺激 R*/s；采样 dt=0.0001 s；响应为 voltage-clamp current 按饱和闪光估计的暗电流归一化后的无量纲数组，为 4–5 次重复的平均。具体每条重复数、原始 trial 与逐 cell 暗电流不在包中，不估 noise ceiling。

文件没有显式 time、cell ID、逐 cell species、精确 eccentricity、L/M 分类、holdout、滤波/重采样说明或预刺激历史字段。标签 PrimateConeResponses{1}…{6} 只是 MATLAB 数组位置，不冒充生物学 cell ID。论文的群体 Macaca/L/M/>20° 描述不能替代逐 cell 元数据。

绘图横轴定义为 (MATLAB sample−1)×0.0001 s：0–7.9999 s，共80,000点/名义区间时长8 s。该相对索引轴来自采样，不声称是实验绝对起点、刺激物理开启时刻或已确认 kernel 相位。

## 全部六条记录的实际统计

| 数据标签 | normalized response min | max | mean | population std | 负刺激点 | ≥50k 点 | 模型 RMSE |
|---|---:|---:|---:|---:|---:|---:|
| PrimateConeResponses{1} | -0.989765788 | -0.314607044 | -0.720867347 | 0.148191183 | 1026 | 12538 | NOT_RUN / null |
| PrimateConeResponses{2} | -0.976038248 | -0.30227009 | -0.732386333 | 0.129857886 | 1026 | 12538 | NOT_RUN / null |
| PrimateConeResponses{3} | -0.956662634 | -0.466928923 | -0.743862284 | 0.119149331 | 1026 | 12538 | NOT_RUN / null |
| PrimateConeResponses{4} | -0.982615601 | -0.303111296 | -0.691684551 | 0.14858393 | 1026 | 12538 | NOT_RUN / null |
| PrimateConeResponses{5} | -0.974598415 | -0.416813305 | -0.725255951 | 0.142656566 | 1026 | 12538 | NOT_RUN / null |
| PrimateConeResponses{6} | -1.0224269 | -0.627845622 | -0.827503933 | 0.0960179283 | 1026 | 12538 | NOT_RUN / null |

所有六条刺激范围均为 −767.7310526189535 至 93782.5675340274 R*/s；1,026/80,000=1.2825% 点为负值，12,538/80,000=15.6725% 点 ≥50,000。完整异常区间以原 MATLAB 1-based sample 和相对索引秒保存于 all_cell_metrics.json，不裁掉失败例。

全部逐记录统计与空误差字段：reports/chen2024_real_data/all_cell_metrics.csv / .json。MSE/RMSE、MAE、max error、variance explained、Pearson r 与 noise ceiling 均未计算，JSON null / CSV 空白不是 0，也不是合格结果。

## 固定模型无法对齐的具体点

已批准修正：仅显式 component_reference + source_reproduction=True 取消瞬时50k拒绝，允许非负有限正值高峰并记录不含漂白/再生的适用性警告；默认参考/numeric接口保留保守工程封套，完整生理模式仍禁止此组件。作者 kernel 无该 cap；Fig.3/4 的高峰及p15语境不支持硬裁剪或机械均值上限。六条真实刺激仍含未解释负值；不能据字段名改成contrast、加背景、取绝对值、rectify或平移。

MAT 没有前置背景或实验/导出时间相位信息；首点刺激为0而响应已是非暗态值，并不足以唯一推断起始状态。历史 demo 的背景前置只是独立示例，不是本 MAT 的初始化证据。没有运行暗态初始化、猜测预适应时长、按响应拟合起始状态、循环预热、去掉首尾或优化时间平移。

如输入与历史后续由来源材料明确，固定模型的无量纲预测可定义为 I_model/|I_dark_model|，当前作者默认分母428.75 pA；它不是测量 ground-truth pA，也不调 gain。逐 cell 暗电流缺失只阻止恢复真实 pA 和逐 cell G_D，不构成所有归一化比较都不能定义的理由。本次仍由输入语义和历史缺口阻断。

论文 Figure2/p5 与 Methods/p19 原作者曾根据逐 cell 暗电流设 G_D，并允许 gamma 不同；本任务固定所有参数、不重拟合，因此不能宣称复现了论文逐 cell 拟合曲线或其方差解释率。参数21条和profiles2份未改变，只新增三个资产来源条目，当前来源19条。

## 图与完整曲线

![全部真实记录及原刺激](../reports/chen2024_real_data/all_cells_source_audit.png)

PNG 与 PDF：reports/chen2024_real_data/all_cells_source_audit.png / .pdf。选择规则在画图前写入 metrics：全部六条记录、完整80,000点范围，另各记录相同首0.5s索引窗口；未根据误差挑记录或时间。纵轴是无量纲发布平均响应，上方为未修改刺激和当前上界参考线。图明确标“model comparison blocked”，没有捏造模型曲线。

完整未改数值副本：reports/chen2024_real_data/full_published_traces.npz，12条真实数组＋相对索引轴＋published_source_repeat_mean 标签。每条数组已在回归中与其原 MAT 字段逐值验证相同。原 MAT 与 ZIP 始终保留。使用现有 bpsnn matplotlib，不安装依赖或改变环境。

## 回归与限定独立检查

数据清点交付当时的回归 A54、B8、C36 全通过，四个检查退出码0，AST无错误。新增真实数据保存/空指标守门与 zip-slip/symlink拒绝回归；没有新增生理拟合。日志目录 `reports/20261010T121617896210Z`；原 MATLAB source-parity synthetic 最大绝对误差仍7.389644451905042e-13 pA，此误差不是本真实数据拟合误差。

独立代理只读确认哈希、字段、单位/时间边界、输入异常和拟合来源性质，支持停止；没有找到明确允许修正刺激的原文依据。详见 docs/chen2024_real_data_review_zh.md。

实际命令与退出码（结构化完整记录另见 reports/chen2024_real_data/execution_record.json）：

```text
C:\Users\PC_User\anaconda3\envs\retipath_v1\python.exe -B D:\macaque_retina_simulator\tools\ingest_chen2024_archive.py C:\Users\PC_User\Downloads\doi_10_5061_dryad_q2bvq83vg__v20241101.zip
```
退出码 0；证据 `reports/chen2024_archive_inventory.json`。

```text
C:\Users\PC_User\anaconda3\envs\retipath_v1\python.exe -B D:\macaque_retina_simulator\tools\inventory_chen_data.py
```
退出码 0；证据 `reports/chen2024_data_inventory.json`。

```text
C:\Users\PC_User\anaconda3\envs\retipath_v1\python.exe -B D:\macaque_retina_simulator\tools\register_chen2024_real_sources.py
```
退出码 0；证据 `sources/registry.json`。

```text
C:\Users\PC_User\anaconda3\envs\bpsnn\python.exe -B D:\macaque_retina_simulator\tools\plot_chen2024_real_data.py
```
退出码 0；证据 `reports/chen2024_real_data/all_cells_source_audit.png`。

```text
C:\Users\PC_User\anaconda3\envs\retipath_v1\python.exe -B D:\macaque_retina_simulator\tools\run_checks.py
```
退出码 0；证据 `reports/20261010T121617896210Z/verification.json`。

```text
C:\Users\PC_User\anaconda3\envs\retipath_v1\python.exe -B D:\macaque_retina_simulator\tools\coverage_report.py
```
退出码 0；证据 `reports/chen2024_real_data/coverage_refresh.log`。

```text
C:\Users\PC_User\anaconda3\envs\retipath_v1\python.exe -B D:\macaque_retina_simulator\tools\extract_packer_cone_density.py --verify
```
退出码 0；日志 `reports/20261010T121617896210Z/check_0.log`。

```text
C:\Users\PC_User\anaconda3\envs\retipath_v1\python.exe -B -m pytest -p no:cacheprovider D:\macaque_retina_simulator\tests\test_stage_a.py -q
```
退出码 0；日志 `reports/20261010T121617896210Z/check_1.log`。

```text
C:\Users\PC_User\anaconda3\envs\retipath_v1\python.exe -B -m pytest -p no:cacheprovider D:\macaque_retina_simulator\tests\test_stage_b.py -q
```
退出码 0；日志 `reports/20261010T121617896210Z/check_2.log`。

```text
C:\Users\PC_User\anaconda3\envs\retipath_v1\python.exe -B -m pytest -p no:cacheprovider D:\macaque_retina_simulator\tests\test_stage_c.py -q
```
退出码 0；日志 `reports/20261010T121617896210Z/check_3.log`。

## 当前状态和下一项来源依赖

只解除“下载/字节身份/结构不可读”阻塞；A02 外周源码数值子项仍 SOURCE_PARITY_VALIDATED，新增真实资产身份子项 EVIDENCE_IDENTITY_VERIFIED，来源数据对照子项 BLOCKED_INPUT_SEMANTICS_AND_HISTORY。父A02仍BLOCKED；完整需求20项、完整模式15项阻塞保留。可读覆盖表中的损坏中文说明已修正，20项 gate 不变。

公开来源数据复现/对照：NOT_RUN；独立外部生理验证：NOT_RUN。该包是作者定参数来源，即使后续对齐成功也不能用作独立验证。中心、S cone、电压转换和HC耦合未实施。

继续所需证据：这些发布刺激的负值/背景定义及处理规则；高亮度曝光条件用于生理适用性说明，实验实际 delivered/raw stimulus 或明确导出/滤波规则，预刺激历史、相对时间/相位映射。若要按论文逐 cell 原设置复现，还需实际暗电流及 gamma/G_D 记录；不能靠重新拟合补这些值。收到来源依据后再解除对应门禁，不需要重新下载当前已核文件。

门禁修正验收另见 docs/chen2024_gate_correction_report_zh.md；原图未重作，图中50k虚线是当时默认工程封套的标注，不表示作者要求裁剪。
