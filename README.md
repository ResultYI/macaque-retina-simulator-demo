<!-- local-platform-entry -->
## 本地可视化模拟器

双击项目根目录的 `run_retina_gui.cmd`。也可在本目录运行：

```powershell
& 'C:\Users\PC_User\anaconda3\envs\retipath_v1\python.exe' -B -m retina_platform
```

导入图像或视频，设置时间段与有效分辨率后运行。支持暂停/继续、停止并保留已完成窗口，查看所有已配置层的真实细胞坐标与所选细胞记录，保存/载入配置、载入平台及已有核心结果、导出图/CSV/完整结果目录。详见 [统一说明](docs/core_design.md#本地可视化平台)。

实际默认由 `configurations/retina_platform_v1.json` 选择；细胞来自所选冻结图的真实依赖视图，不再用 64 cone / 96 rod 代表来源密度。来源借用、功能性连接、工程近似和实际背景适应状态分别记录在 [统一说明](docs/core_design.md)。旧 2 ms 预运行与低分辨率 fixture 保留为工程预览，不构成稳态或生理标定。固定眼位为默认，DeepGaze / SC 探索继续暂停。

示例媒体位于 `examples/platform_media`，明确标记为 synthetic。界面与核心端到端验收及实际截图：[`reports/platform_acceptance/20261011T022329305607Z`](reports/platform_acceptance/20261011T022329305607Z/acceptance.json)。
<!-- /local-platform-entry -->

项目位置：`D:\macaque_retina_simulator`。已从旧项目目录整体迁移；历史 manifest、来源及执行记录未改写，旧绝对路径通过 [迁移映射](migration_map.json) 对应当前文件。目录外独有的 DeepGaze 代码、官方来源/权重及两条件410 ms原始结果已归档到 [exploratory/deepgaze_exploration](exploratory/deepgaze_exploration/PAUSED.json)，保持暂停；迁移不构成恢复实验的授权。原暂存目录保留。

<!-- current-all-target-entry -->

历史全层工程入口 `tools/run_full_target.py` 及其 synthetic 运行记录继续保留。当前可视化主线使用上面的 GUI 和单一主配置；最新来源范围、实际运行结果与尚未验证部分以[统一说明](docs/core_design.md)为准。



# 猕猴中心—周边连续视网膜模拟器



本项目的完整目标是从标定光输入到显式细胞及连接、持续状态和逐层输出，同时覆盖中心、过渡区与周边。当前交付包含阶段 A/B 工程基础与隔离的 Somaratna–Freeman Gaussian-HC 线性参考，以及阶段 C 的 Chen2024 外周 >20° L/M cone 非线性外段 pA 模块。完整生理模式仍被阻断。



完整需求包括非均匀 L/M/S cone 采样、类型与位置、HC/BC/AC/RGC 和 ON/OFF、soma 与 RF 坐标及位移、稀疏连接、区域依赖动态、给定眼动、真实单位光输入及独立生理验证。局部参考内核和 synthetic 单测不替代这些目标。



- [阶段 C 中文交付报告](docs/chen2024_delivery_report_zh.md)：原 MATLAB 源码数值对照、最终 A/B/C 检查和真实 Dryad 下载阻塞。

- [手动 ZIP 后的真实数据报告](docs/chen2024_real_data_report_zh.md)：MAT/README 官方哈希已匹配，全部六条真实记录已清点；输入语义与预历史仍阻断固定模型比较。

- [已批准的作者复现门禁勘误](docs/chen2024_gate_correction_report_zh.md)：显式作者复现允许正值高峰并记录漂白适用性警告；动力学、参数及真实数据不变。

- [全部真实记录图](reports/chen2024_real_data/all_cells_source_audit.png)、[逐记录 CSV](reports/chen2024_real_data/all_cell_metrics.csv)、[完整轨迹](reports/chen2024_real_data/full_published_traces.npz)、[真实数据限定检查](docs/chen2024_real_data_review_zh.md)。

- [阶段 C 实施合同](docs/chen2024_implementation_contract_zh.md)、[来源审计原文](docs/chen2024_source_audit_zh.md)、[独立限定审查](docs/chen2024_review_report_zh.md)。

- [阶段 A/B 历史交付报告](docs/delivery_report_zh.md)：保留当时测试、误差和限制。

- [已确认 A/B 实施合同](docs/implementation_contract_zh.md)。

- [Pro 三稿与独立审查档案](docs/pro_consultation_and_review_zh.txt)：保留错误原文，仅作咨询记录。

- [原会话三份答复](docs/pro_original_message_archive.json)：工具返回的原始文本，保留会话引用标记。

- [逐项覆盖表](docs/requirements_coverage.md) 与 [机器需求表](requirements.json)。

- [来源规则](docs/evidence_review_rules.md)、[来源注册表](sources/registry.json)、[schema](schemas/source_registry.schema.json)。

- [实际验收记录](reports/verification.json)：每个命令、退出码、日志及数值误差。



`retina_sim` 提供单位与语义类型、参数及物种门禁、源资产校验、带方向的 CSR 图、密度表查询、光积分、联合 ODE 求解及 checkpoint。主线生理配置没有默认参数。参考配置包含作者的混合物种来源，必须显式选择，不能进入完整生理模式。



60 行 ISETMacaque 表属于 `published_code_tabulation`，并非原始实测细胞计数。保留原文件、固定 commit、逐行位置、原数字面量与 SHA256。不填补 e=0、不外推、不把鼻颞表当二维分布；插值与倍率换算必须显式选择。



阶段 B 参考输出是固定参数的 mV generator 状态及整流 Hz rate，没有 AC 或随机 spike 机制。Somaratna–Freeman 作者 MAT 的字节身份已核，MCOS 数值表尚未解码，作者原阵列及论文结果复现仍阻断。



阶段 C 明确选择作者三 Macaca 物种混合来源，仅支持外周 >20° L/M cone。默认参考/numeric 接口保留 0≤u<50k 工程封套；仅 component_reference 且 source_reproduction=True 的作者复现接口允许非负有限高峰，并记录不含漂白/再生的适用性警告，不认证高亮度生理有效。固定历史参数、逐点离散更新和完整 R/P/G/C/S 状态缓存；输出绝对负 pA，不连接电压或 HC/BC。五组 synthetic_fixture 已与真实执行的原 MATLAB kernel 对照。用户提供的 Dryad ZIP 已安全提取、真实 MAT/README 大小和官方 SHA256 匹配。真实负刺激含义及预历史/相位未解，来源数据模型对照仍未运行；独立生理验证未运行。≥50k 点数不是来源数据不可复现的单独依据。当前局部子项不能替代连续全视网膜主线。



使用已存在解释器，不安装依赖：



```powershell

Set-Location 'D:\macaque_retina_simulator'

& 'C:\Users\PC_User\anaconda3\envs\retipath_v1\python.exe' -B .\tools\run_checks.py

```



该命令只运行来源一致性及 synthetic 数值验收，写入带时间戳的测试日志。不运行训练、生理实验或服务。细节见中文总报告。

