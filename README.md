## 当前研究与桌面入口

双击 `run_retina_gui.cmd` 打开当前 **69#6 内部状态浏览**，读取已有结果，不启动模型计算。支持分层状态、对象曲线、刺激同步、中日英切换、滚轮缩放和图像导出。上游连续状态只覆盖已保存的 5.0–5.5 s 短窗口，其他时段按真实记录显示。操作说明见 [桌面回放说明](docs/response_viewer_zh.md)。

- 当前科学模型入口：`RetinaBuilder.build_research_ac`，配置为 `configurations/research/AC_functional.json`；建模范围与参数来源见 [当前建模原理](document/当前建模原理.md)。
- 当前 69#6 结果：`reports/GIN_69_6_HC2_BC1_SUPPORT_V2/current_ac_source/`；[实验报告](reports/GIN_69_6_HC2_BC1_SUPPORT_V2/FINAL_REPORT.md)记录科学状态与限制。
- `retina_sim/` 放模型与数值实现，`retina_platform/` 放桌面读取和显示，`tools/` 放数据准备及实验入口，`tests/` 放仍需维护的回归检查。
- 现有旧入口可能被实验脚本、源码身份或测试夹具依赖，名称中的版本号不表示可删除。清理与检查记录统一维护在 [项目整理记录](document/项目整理记录.json)。

已确认停用的历史工具及两个合并后的重复入口共 **109 个文件（97 个 Python、12 个 PowerShell）** 已移到项目外的 `D:\macaque retinal simulator archive`；`tools/` 的 Python 文件由 441 个减至 344 个。归档中的 `ARCHIVE_INDEX.json` 保留原相对路径、大小和 SHA256，`RESTORE.txt` 说明恢复方法。历史文档中的旧工具路径可在此索引定位；不要直接在归档目录启动实验。原始数据、有效缓存、科学结果和当前源码绑定仍留在本工程。

五份修复前源码备份的项目内原件已按用户确认删除，校验一致的外部归档副本全部保留；明细见整理记录中的 `tools_archive.deletion_candidates`。当前仍使用的旧名共享工具和源码身份绑定文件继续保留。

<!-- response-package-desktop-v1 -->
## 完整响应包与历史回放

完整响应 ZIP/目录使用 `python -B -m retina_platform --response-replay --result <结果路径>` 打开，保留原有九层回放。已有 4 秒示例为 `reports/result_packages/retina_113_4s_298fa2ef1524.response.zip`。旧 Cell6 回放使用 `--cell6-replay`。以下其他入口和早期阶段记录保留各自实验范围，不替代上述当前模型与结果。
<!-- /response-package-desktop-v1 -->

## 研究室光学依赖迁移

固定 ISETCam/ISETBio 工作目录和 101 个 OTF 缓存已随主工程交付；保留 data、external 与许可证，排除依赖自身的 Git 元数据。版本、逐文件 SHA256、Gaussian 孔径语义及 Linux 独立配置要求见[研究室迁移说明](docs/lab_optical_migration_20261102_zh.md)。


<!-- local-platform-entry -->
## 本地可视化模拟器

双击项目根目录的 `run_retina_gui.cmd`。也可在本目录运行：

```powershell
& '.\.venv\Scripts\python.exe' -B -m retina_platform
```

Windows 启动脚本会在当前进程未设置时读取用户环境中的 `PIP_CACHE_DIR`、`TORCH_HOME`，使缓存迁移后从旧终端启动也能使用新目录；当前进程显式设置的路径仍优先。

仅做输入前端的 cone mosaic 对照图（不运行后续电路）：

```powershell
& '.\.venv\Scripts\python.exe' -B tools/cone_mosaic_demo.py --output reports/cone_mosaic_demo_new
```

该入口直接复用当前主线冻结查询域的 cone 位置、ID 和 L/M/S 标签，以及已配置的 ISET 显示/光学捕获函数；不编译电路、不进行光转导。默认分别使用项目 RGB 色块/渐变与 ISETCam 水果/色卡自然图，可用 `--natural` 更换自然图。输出目录须为新目录，包含两版英文 PNG/PDF、10 ms 原始期望光异构化数、逐细胞 CSV、统一色标、图源/模型假设及哈希；图内无副标题或图注。合成图还用于检查 x/y 方向；需要现有冻结资产和 MATLAB，缺失时不生成替代几何。最新结果说明见 `reports/cone_mosaic_demo_20261028_v5_english/README.txt`。

本次只显示约 1.17–2.14 deg 偏心度的已有局部，未覆盖中央凹中心到远周边。图中圆盘为按完整冻结格点六近邻间距中位数 × 0.79 得到的模型采光孔径（本例直径 2.89–3.74 um），不是实测内节/胞体直径。按用户选择，仅使用 ISETBio 自带依据，不强加缺乏对应猕猴资料的 L/M/S 直径比例。相同微米比例的放大窗口及孔径—偏心度检查见 `reports/cone_mosaic_demo_20261028_v5_english/aperture_size_check.png`；该检查只读取保存结果，不重跑光捕获或电路。

当前生产光学入口共用 `cone_optics_mosaic.m` 和 `capture_cone_optics.m`：逐 cone 使用其实际孔径的离散空间积分核，外节衰减按各类型的吸收谱/光密度计算；保留 ISETBio 标准外节模型的 400:10:700 nm 波长范围。ISETBio CVRL 外节长度与既有光密度保持一致，不重复乘入 S-cone 长度衰减。原有晶状体、黄斑滤光和固定全局参考继续使用。实现范围、证据和未启用项见 [光学输入证据核查](docs/cone_optical_input_evidence_20261028.txt)；旧输出保留，模型/源码哈希使新捕获与旧缓存分开。

自然视频光捕获可运行 `& '.\.venv\Scripts\python.exe' tools/cone_video_demo.py --media 'D:\macular\examples\video\video\113.AVI' --duration 4 --output reports/cone_video_demo_new`。默认时长为 1 秒，可用 `--duration` 改为 4 秒或其他帧边界对齐的时长；30 fps 的 4 秒片段对应 120 个 33.33 ms 窗口。可用 `--start` 指定帧边界上的起点；本入口要求整数帧率的恒帧率视频、可核对的解码时间戳和足够长的完整素材，不循环补足短片，并保留原有内存与逐次导出时间上限。输出英文三面板 MP4、代表帧 PNG/PDF、逐帧逐细胞 CSV、原始捕获和来源记录；视频全程色标固定。仅计算光学捕获，不运行时间适应、光转导或后续电路。

追加 `--fixational-eye-movements --eye-seed 20261028` 可加入固定版本 ISETBio 默认的漂移和微扫视。30 fps 每帧使用 134 个约 0.249 ms 的中点积分子步；轨迹来自原生 1 ms 模型，并按原生规则线性插值。保留原有帧均眼位黄斑近似，cone 几何不变。输出增加眼动轨迹 CSV/图、参数及九个指定 cone 的细时间捕获数据；绘图输入面板显示帧中点眼位，响应面板显示整帧积分。该路径是模型生成而非实测眼动；兼容层和验证依据见 [眼动输入说明](docs/cone_fixational_input_evidence_20261029.txt)。

追加 `--central-2deg` 可使用现有猕猴母格点中以中央凹为中心、半径 2°（直径 4°）的完整圆域，共 45,480 个 cone；该选项要求关闭微动。沿用全局 ID、完整母格点孔径及既有分型规则，不重新采样格点，也不修改局部电路配置。视频等比例放大填满视野，居中裁掉左右多余部分，视野内无灰边；屏外光学支持区仍在源图像外使用参考灰。原生光学输入按四帧分批，保留原有资源限制。绘图输入的降采样只影响展示，逐细胞捕获使用原生光学网格。

中央视频入口在本机使用四个独立 MATLAB 进程，每进程最多八个计算线程；各帧输入与捕获仍遵守原有逐次导出上限。`--central-geometry-from <已有结果目录>` 可复用已保存的中央分型，读取时检查母格点、全局 ID、孔径、分型来源及规则文件哈希；规则变化则要求重新准备。绘图采用实际物理采光圆盘的四倍超采样，只加快显示，不改变逐细胞捕获。

导入图像或视频，设置时间段与有效分辨率后运行。支持暂停/继续、停止并保留已完成窗口，查看所有已配置层的真实细胞坐标与所选细胞记录，保存/载入配置、载入平台及已有核心结果、导出图/CSV/完整结果目录。详见 [统一说明](docs/core_design.md#本地可视化平台)。

默认为 64 cone / 96 rod 的显式有效分辨率、2 ms 工程适应与 6 ms 记录，覆盖现有 20 机制层 / 39 组件；不是全解剖密度，也不代表生理标定完成。RGB 输入采用明确工程显示假设，固定眼位为默认。DeepGaze / SC 探索继续暂停。

示例媒体位于 `examples/platform_media`，明确标记为 synthetic。界面与核心端到端验收及实际截图：[`reports/platform_acceptance/20261011T022329305607Z`](reports/platform_acceptance/20261011T022329305607Z/acceptance.json)。
<!-- /local-platform-entry -->

当前项目位置：`D:\macaque retinal simulator`。历史 manifest、来源及执行记录未改写，旧绝对路径通过 [迁移映射](migration_map.json) 对应当前文件。旧迁移记录中的 DeepGaze 探索保持暂停；本项目的代码整理不构成恢复该实验的授权。

<!-- current-all-target-entry -->

早期全层工程入口：`tools/run_full_target.py`。[架构与交付说明](docs/core_design.md)及[当时状态](reports/core_architecture_status.json)保留其 rod/RBC/AII 路径、逐参数证据和 synthetic 工程运行记录。该入口和旧 fixture 用于历史复现，不代表当前 69#6 模型或生理标定状态。



# 猕猴中心—周边连续视网膜模拟器



本项目的完整目标是从标定光输入到显式细胞及连接、持续状态和逐层输出，同时覆盖中心、过渡区与周边。以下是阶段 A/B 工程基础、隔离的 Somaratna–Freeman Gaussian-HC 线性参考，以及阶段 C 的 Chen2024 外周 >20° L/M cone 非线性外段 pA 模块的历史记录；各自的来源限制和未完成项仍按原报告保留。



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

