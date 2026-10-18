# 台式接续：笔记本代码与验证快照（2026-10-18 UTC）

这是笔记本 `D:\macaque retinal simulator` 当前已完成工作的代码交接。仓库为
`ResultYI/macaque-retina-simulator-demo`，工作分支为
`review/physiology-wip-20261017T2356Z`，本次上传前提交为
`db7a3ca5cd0259b230ce762c0d2a3d2a5cb509d5`。
它不代表家中台式机尚未提交的修改已被收集，也不是完整生理验收。

## 台式机先保留自己的工作

在家中实际项目（此前为 `D:\macaque_retina_simulator`）先只读核对：

```powershell
git status --short
git branch --show-current
git remote -v
git log -1 --oneline
git fetch origin review/physiology-wip-20261017T2356Z
git diff --stat HEAD...origin/review/physiology-wip-20261017T2356Z
```

先单独保存本地未提交和未跟踪文件，再逐项比较。不要直接用笔记本文件覆盖，
不要自动 reset、强推、合并 main、重贴模型哈希或跳过检查点身份校验。
如有版本分歧，先确认台式机未提交工作的意图再整合。
不要误入另一个项目 `D:\next saccade`。

## 本次实际代码

- `retina_platform/grouped_disk_capture.py`：显式分组独立光学曝光，保持原导出器、
  冻结几何/光学/参考与100µs输入映射；仅分组缺失请求，保留旧缓存来源。
  `bounded_iset_batch.py` 同时修正 Windows 超长 `.ready` 路径的进度判定。
- `tools/video_background_start.py`：背景默认仍为 strict；只有显式 transient
  加明确来源才接受诚实标记的 `NOT_REACHED` 功能/吞吐测试。
  原身份、状态、单位与历史检查保留，不能把它称为充分适应。
- `retina_sim/lossless_video_storage.py`：单次检查点内规范JSON子树只编码一次，
  有界磁盘暂存；原JSON协议、校验及哈希不变。观察NPZ可显式取消DEFLATE，
  数值及元数据不变，代价是更多磁盘空间。
- `tools/prepared_video_input.py`：只准入已完成、同视频字节/时间段/配置/背景/
  实际模型/几何的输入；仍由原严格加载器校验每个分片。错误媒体或视线会拒绝。
- `retina_sim/no_delay_publication.py`：只在空 link-delay 缓冲区的四处发布前避免
  无消费者的全状态拼接。有缓冲区时原发布顺序保留；HC自身的观察延迟历史不跳过。
  原数值源字节不变，helper pin 原 `hybrid.py`，执行过程SHA显式写入运行身份。
- `tools/video_diagnostics/`：从本地报告目录保留的真实短窗计时、配对、验证和
  首次异常见证脚本。见该目录README；它们不是开箱即跑的数据包。

`tools/run_video_05s.py` 的新选项均显式启用：

| 选项 | 默认 |
|---|---|
| `--background-mode strict` | strict；transient须另给 `--background-source` |
| `--group-optical-frames` | 关闭 |
| `--prepared-input-from <completed-run>` | 不复用指定旧输入 |
| `--fast-lossless-storage` | 关闭，保留原写入 |
| `--skip-empty-link-publication` | 关闭，保留原advance |
| `--restore-diagnostics` | 关闭 |

没有新默认生理图，没有更换积分器或精度。新缓存/缓冲区优化仍暂停，
按用户要求放到家中台式机再做：本轮没有实现它，也没有启动其测试。

## 独立环境

已验证的是独立 Windows x64 Python 3.11.15（MSC 1944），不使用原混合`.venv`。
基础依赖仍见 `requirements.txt`；下面是实际验证版本，不是自动升级命令：

| 依赖 | 实测版本 |
|---|---|
| NumPy / SciPy | 2.4.6 / 1.17.1 |
| Pillow / OpenCV headless | 12.3.0 / 5.0.0.93 |
| Numba / llvmlite / TBB | 0.67.0 / 0.49.0 / 2021.8.0 |
| PyTorch / CUDA runtime | 2.14.0+cu126 / 12.6 |
| MATLAB | R2025b Update 5，另需本地已批准ISETBio/ISETcam资产 |

Python来自独立conda-forge环境；Torch等按该环境实际来源安装，所需Windows运行库
与线程运行时也须匹配。仓库没有打包此环境、安装器或所有二进制依赖。
台式机先验证自己已有环境，不要全局安装或混用旧site-packages。
CUDA型号、Torch/CUDA版本、执行后端等参与身份；不同GPU不能保证直接恢复笔记本检查点。

启动前清空 `PYTHONPATH`、`PYTHONHOME`，设置 `PYTHONNOUSERSITE=1` 和
`OPENBLAS_NUM_THREADS=1`，再启动新Python进程。CUDA原路径设Torch CPU线程1，
FP64且TF32关闭；不改变原FP64 DOP853的 `rtol=1e-6`、`atol=1e-9`、`max_step=100µs`。
原路径保留CPU/Numba内层、CPU事件/history/RNG和CUDA联合DOP853。

## 验证边界与证据

精简统计和哈希见 `docs/validation_20261018.json`，不包含状态、RNG内容或原始数据。

- 实际完整0.5秒的存储/严格输入复用比较：端到端3441.192→2750.175秒，
  减少20.08%；推进/记录/滚动检查点2843.567→2394.160秒。两轮均复用已有光学结果，
  不包括此前MATLAB光学准备；求解器本身耗时基本不变。
- 全部501个观察时刻、每时刻37,449个记录状态逐位一致；首末完整检查点一致，
  覆盖4,967,023个注册状态及辅助时钟/RNG/history。不是每1ms全图快照。
  跨进程的两个原诊断计时字段单独分类，其余科学/事件元数据一致。
- 去掉NPZ压缩使记录约0.892→7.356GiB，增加约6.464GiB。
  检查点仍按前次写完后满60墙钟秒、在原2ms窗口边界写入，触发规则和内容不变；
  实际滚动次数35→33，因为运行更快，不能称为相同模拟时刻检查点。
- 实际20–40ms互斥profiling跨33.4ms光学输入切换；没有乘25冒充完整运行。
  主要求解器外开销是构造复制/有限值检查、CPU信号求值与全状态打包替换。
- 空发布修复另作一次无嵌套计时钩子的同20–40ms配对：86.329→81.783秒，
  少4.546秒（5.27%）；求解器55.759→55.742秒，求解器外30.570→26.041秒。
  逐记录及全部当前元数据相同；原版/补丁/上轮独立40ms末态文件SHA相同。
  两轮各225次积分、3510次RHS。有delay小回归98次发布与完整结果一致。
  这不是新的完整0.5秒提速测量，也不能把两个百分比直接相加。

原图为旧HC v7，879,437依赖节点、77状态块；human ISET cone/optics为借用来源，
不是完全macaque模型。此次背景为1.012秒 `NOT_REACHED` 瞬态。
新的functional HC全域图仍未验收，默认未切换。
首次几何门拒绝的根因仍未查明，后续完整原检查通过；没有用绕过检查代替修复。

## Git不含的资产

仅clone不能恢复完整实验。以下都留在本地，台式机需复用自己已批准、身份匹配的资产，
或另行手工迁移；本次没有执行数据迁移或上传：

- 独立环境、旧`.venv`、`vendor_pinned/`及MATLAB/ISETBio/ISETcam本体和外部依赖；
- 被忽略的 `data/source*`、大atlas/人源借用输入、原始视频和论文/源码资产；
- 本机光学profile配置、背景/20ms/完整检查点、光学cache和输入分片；
- 整个 `reports/` 内的GB级窗口记录、日志和本机诊断配置。

历史复现脚本引用这些相对目录：
`reports/human_disk3mm_floor_transition_v7/transition`、
`reports/performance_real_prepared_A_v4/legacy_bridge_case`、
`reports/notebook_transient_source_20261018T093006Z`、
`reports/notebook_clean_geometry_observed_video05s_20261018T162158Z`、
`reports/advance_exclusive_profile_20261018T230442Z`。
它们都不在Git。路径匹配不能代替内容哈希、几何和模型身份匹配。
原v7编译器由 `tools/benchmark_compiler_snapshot.py` 从已有提交
`3333811d2dd737cacbb30bf746b19ad0ebdfb0e0` 实际源码恢复；此旧提交不是今天的新代码。

后续工作先在台式机确认真实版本，再审计安全别名生命周期。不能把不同RHS stage/
trial state、事件commit、输入变化、RNG/history变化下的有效权重当静态量缓存；
保留原校验、观测顺序、只读快照隔离和检查点身份。当前没有接受任何新缓存收益结论。
