# Cone–HC 简化候选与笔记本迁移

本说明只对应显式的 **解析 R/E＋w=A z1＋FP64 指数中点＋CPU inner** 候选。它是新的 B 类数值后端，不替换普通 GUI、compact、resident 或原 split 默认，不把原缓存重标为新后端产物。

实施依据：主方案 `reports/chatgpt_latest_reply_review_20261024_v1/assistant_dbdc7ee4-254a-4605-8a9a-4178aaf3f515.md`；最终补充完整规范化归档为 `reports/cone_hc_reduced_implementation_20261024_v1/SUPPLEMENT_ORIGINAL.md`。报告目录保留本地证据，按现有 `.gitignore` 不上传整份运行资产。

## 候选保留什么

- 原 150 Hz 刺激、光学、空间矩阵 A/B、细胞身份、研究输出范围、完整 Cone↔HC 反馈和 inner 全部原连接。
- FP64；native 100 μs 基础步，在真实输入边界短步截断；原 CPU inner 动力学与 SciPy/Numba 路径、100 μs rod 递推、1 ms BC/RGC 历史/事件、PCG64。
- 五组核心观测：I(当前 nX=1，I=X)、Vis、C/Vbar、release、h/Bh。gi、R/E、z2、q 继续有真实状态或解析诊断语义。
- 原指定 HC 观察的 2.82 ms 延迟、prehistory、100 μs 队列网格和剪枝。反馈仍用当前 h，不能把观察延迟插入反馈回路。

native 数值向量为五组 Cone `X,C,Vis,g_i,Vbar` 和三组 HC `w,z2,Vh`。R/E 从真实合法初态保留解析锚点，在输入边界刷新，不是删除其时间记忆。w 的合法初态是完整旧 z1 的 `A@z1`；当前 gt 为标量、tau1 统一固定、nX=1、A/B 运行内固定等前提均须校验。

指数中点采用同一起点快照产生全部半步状态，再以同一半步快照产生终点。二次连续延拓为：

`x(t)=x0+theta*dt*f0+theta**2*(x1-x0-dt*f0)`。

这里 f0 是原方程真实起点导数，从第一轮系数复用。半步延拓值无需等于指数中点预测值；monitor 在查询态重算真实 RHS，不能用多项式导数替代原方程。参数 tau 为 ms，执行时间为 s，必须保留 1000 换算。

## CPU 桥与 z1 能力边界

桥接选择实际 inner 消费 release 的 Cone 集合 C*，以及这些 B 行所需 HC 集合 H*；保留 CSR 权重、行内顺序与 ID，不裁剪或重新归一化。每个接受 segment 只发送这些 Vis/Vbar/h 的 `(x0,f0,x1)`，FP64 payload 为 `8*(6*c+3*m)` 字节，CPU stage 本地执行二次状态读出→B*h→原非线性 release。实际原图测得 c=966、m=998、payload=70,320 字节/segment。双 pinned 槽位及 ready 事件保护未消费数据，四类消费水位和 checkpoint 保留完整因果所有权。跨 chunk 返回的状态先取得未消费 packet 的独立不可变副本，再归还 pinned lease；GPU 完整区间继续由消费水位保护。

默认不再保留全量逐 Cone z1；**w 不叫 z1，也不能由 w 反演逐 Cone z1 历史**。确需 z1 时，在实验开始前冻结少量 Cone 集合 K，使用真实旧值/原合法初始化，同步推进辅助 z1。checkpoint 记录 K 的 ID、顺序、值及版本；运行中不能新增 K 后猜造过去。需要全部旧 z1 的实验继续用原参考后端。

本轮不新增原来不存在的钳制或干预 API。修改 A、逐 Cone tau1 异质化、任意全量 z1 历史查询等超出该候选表示支持范围，必须明确拒绝，不能以零值、别名或假旧状态兼容。

## 入口与验证状态

普通 GUI 仍用 `run_retina_gui.cmd`，或项目根目录下 `python -B -m retina_platform`；这不是新候选的自动入口。启动器支持显式 `RETINA_PYTHON`，其次项目 `.venv`，最后当前用户的 `anaconda3/envs/retipath_v1`。

新候选的显式入口为 `tools/validate_cone_hc_reduced.py`。先用 `--prepare-only --output <新报告目录>` 封存方案和原资产身份，再对同一目录使用 `--execute`。每个目录只接受一次执行；失败和 `.writing` 目录留存，不能清空旧结果重新刷通过。候选不自动进入正式拟合缓存生产、长电影或默认 GUI。

runner、evaluation、monitor、checkpoint 已有独立模块，并接到原 CPU Numba 内层及事件合同。原参考 evaluator 文件保持原字节；新 evaluator 内保留同一段 CPU gather/observe/gap/RHS 源码，并有实际调用测试和源码顺序检查。新公开状态为 `ReducedPoint(native, inner, compact_release)`，不再导出旧全量 native 向量。已接受区间的四类同步消费者完成后才可回收；跨 chunk 的未消费 native 区间、CPU packet 和水位全部进入新 checkpoint。

本轮最小验证范围为小单元/接口检查，再同图、同输入、同合法起点的一次有界真实短段，复用已有电影资产和资格起点，不重跑光学/背景。预定区间 6.4→6.5 s，原图 30 组件、12 RGC；在初段非事件时刻保留未消费区间并保存/恢复新 checkpoint，随后仅从恢复态续算。检查 R/E、核心 native 量、w 对原 A*z1、选中 aux z1、同新轨迹 CPU/GPU release、inner/实际事件/RNG/history、2.82 ms 延迟、完整 monitor 及分相时间。严格门与既有候选 B 门分别报告，w 误差由原 z1 预算经 |A| 传播，不另放自由阈值。实际 B 事件门首个分歧即停止，不补跑另一输入或长段。

新方法没有自动继承 DOP853 的容差保证。失败须保留原始证据，不能改 100 μs 步长、参数、随机种子、刺激或误差门追求通过；未通过时保留显式候选，不替换默认。单个短段通过也不证明 65 秒稳定、拟合质量或生理有效性。

## 笔记本软件条件

当前未连接或检查笔记本，OS、CPU/GPU、RAM、驱动、MATLAB 和许可证均未知。以下是现台式机实现的要求，不是笔记本已通过的验收。

| 路径 | 条件 |
|---|---|
| 普通 GUI | Windows/Python 3.11（台式机实测 3.11.16），带 Tcl/Tk；现有 `requirements.txt` 的 NumPy/SciPy/Pillow/OpenCV |
| CPU 小单元检查 | 原库与适用的 pytest；不等于可在 CPU 跑完整新 GPU 候选 |
| 新 reduced native/桥 | NVIDIA CUDA GPU；现有开发构建 PyTorch 2.14.0+cu130、Numba 0.67.0、llvmlite 0.49.0，NumPy 2.4.6、SciPy 1.17.1 |
| 原保留 CSR ALG2 | 当前适配从 `torch/lib` 加载唯一 `cusparse64_*.dll`，按已核验 CUDA 13.0 ABI；Windows DLL 路径，Linux/macOS 未实现等价加载 |
| 光学生成 | MATLAB R2026b 与冻结 ISETBio/ISETCam；若只消费已绑定完整输入资产，是否能省掉生成依赖以实际入口为准 |

`requirements-reduced-windows.txt` 是可选候选版本清单，包含原 GUI 基础依赖和上述 Numba/Torch 版本；它不选择未知下载源、不安装 GPU 驱动或 MATLAB，不保证对应 wheel 对笔记本可用。不要盲目升级现有环境或把其他 Torch/CUDA 版本当成已认证相同实现。不同 GPU/库/驱动的身份与数值范围需明确记录。

本次台式机硬件为 RTX 4070 Ti SUPER（16 GiB 档），驱动查询为 610.88；实际装载 `torch/lib/cusparse64_12.dll`，不能仅根据 DLL 文件名猜 CUDA ABI。其 SHA256 为 `a155d5bfd5bff68305557be21486470b9949cf6a45a3acda80b27e1670e671e2`。测试框架实测 pytest 9.1.1；GUI 依赖实测 Pillow 12.3.0、OpenCV headless 5.0.0.93。这些是本机记录，未认证为所有笔记本的安装组合。

## 源码与大资产分开迁移

当前分支 `review/physiology-wip-20261017T2356Z`，远程仓库 `https://github.com/ResultYI/macaque-retina-simulator-demo.git`。本轮实施前本地 HEAD 为 `6612e00`，有 42 个 tracked 修改和 348 个 untracked 条目、无暂存内容；这些不是都属于本轮，作者也不能仅凭 Git 确定。提交须逐文件选择本轮代码和必要 WIP 依赖，不 force push，不覆盖未知修改，不全量 `git add`。

本轮新增代码提交不等于以下资产已传到笔记本：

| 资产 | 当前相对位置/来源 | 迁移要求 |
|---|---|---|
| 冻结实际图与矩阵 | `data/atlas/human_disk3mm_functional_hc_v1/`，由配置引用其 `circuit/manifest.json` | 复制实际依赖的完整文件集合及原 manifest，保持字节和 ID/顺序 |
| 冻结输入几何/计划 | `data/atlas/human_disk3mm_input_plan_v2/` | 保留 aperture/row ownership/参考定义和内容身份 |
| VR 原输入与 teacher | `tools/vr_teacher_handoff_v1/intake_drive_20261023_v1/` | 保留 stimulus/index、gray8 chunks、task/teacher 清单和原切分，不另归一化输入 |
| 当前生产进度/光学 | `reports/first_useful_split_capture_20261023_v1/` 下实际 optical、linear input、source/qualification/checkpoint/receipt | 只迁移需要的合法前缀及完整引用闭包，未提交目录不能冒充完成；默认不上传仓库 |
| ISET | 台式机旧 `D:/retipath/.../vendor_pinned/` 下 ISETBio `d2924c2f`、ISETCam `17146a9f` | 冻结来源与许可独立传输，不能因为路径旧就改用旧工程模型 |
| 本机光学配置 | `configurations/iset_lcd_apple_mw_v1.json`，Git 忽略；模板为同目录 `.example.json` | 笔记本设置其实际 MATLAB/vendor 路径，科学字段/固定参考不猜值 |

项目 `.gitignore` 排除 reports、machine-specific ISET 配置和若干大 atlas/vendor/source 目录；另有未跟踪 output/artifacts/tmp/MAT，不应因忽略不全而加入提交。`.gitattributes` 的 `* -text` 保留字节身份，不能让换行转换修改被科学 source hash 绑定的文件。

历史 `docs/notebook_setup_zh.md` 和 `reports/notebook_transfer/runtime_assets_20261013.zip`（62,758,517 字节）面向旧34组件局部图；其“尚未初始化 Git”等状态已经过时，**不能用这份包宣称当前30组件新候选已可迁移**。本轮不重写历史证明或把它改名成当前包。

应由明确的当前运行/依赖 manifest 选择资产，独立通过用户允许的文件传输复制，保留相对路径，并在笔记本检查大小/内容身份；不上传完整研究数据、缓存、凭据或无关来源包。本轮尚未做大资产打包、传输或笔记本检查。

## checkpoint 与路径身份

原完整 checkpoint 必须原样留存。显式单向迁移从真实旧 R/E、z1 和其他状态构造新解析锚点、w 与固定 K 辅助值，保留原 PCG64/history/延迟/时钟及来源身份。新 checkpoint 具有新的模型、状态、monitor、solver 和桥接版本；不能伪装成旧模型 checkpoint，也不能自动复用旧源缓存为新候选产物。

路径应由项目根和显式配置解析，不能引入新的 D 盘绝对路径。历史清单中的绝对路径属于证据，不能为迁移随意重写哈希。实际跨根目录续算必须使用实现提供的可验证重绑定/迁移规则；若 loader 拒绝，先说明身份冲突，不能跳过验收或删除 manifest。

光学输入和 teacher 在输入政策/内容身份不变时可复用；改变算法后，旧背景资格与源缓存是否仍适用必须按新的资格合同判断，不能从“状态已转成功”推导已数值合格。笔记本上首次运行前还需确认 GPU/RAM 是否满足守卫；不能把现台式机12 GiB资源上限照抄到更小硬件后承诺安全运行。

## 本轮交付边界

2026-10-24 已完成六个 reduced 模块、独立真实验证入口及必要依赖接线；科学状态为 **`FAIL_CANDIDATE_RETAINED`**。它保持显式候选，不切换默认、不具备拟合资格。原生产 `REQUEST_STOP` 的 SHA256 保持不变。没有运行光学、长预热、长测试、拟合或笔记本计算。

CPU 数学与接口检查共 45 项通过，覆盖 native、桥接、事件、完整 codec、monitor、原 split 接口和原图绑定。最终真实证据位于 `reports/cone_hc_reduced_real_ab_20261024_r4/`，协议 SHA256 为 `79eaea3b14b208d4cf54d803b0044f51aae0654458202d1bc6aa6c54cb2b9b54`。先前两次准备阶段停止，以及第三次参考臂仅推进 31 μs 后暴露的 `SimulationResult` 字段错误，均保留原报告；修复接口后才完成本次 6.4→6.5 s A/B。它们不算已通过的数值对照。

| 实测阶段 | 秒 |
|---|---:|
| 原图与合法起点冷准备 | 422.511 |
| 参考完整神经 advance，含完整 monitor/选定读出 | 234.131 |
| 候选完整神经 advance，含完整 monitor/选定读出 | 108.068 |
| 候选 native 指数中点推进墙钟 | 7.814 |
| 候选 CPU inner 处理墙钟 | 26.173 |
| 候选 monitor/比较快照回调墙钟 | 60.276 |
| 候选 D2H CUDA event 复制时间 | 0.02749 |
| 候选桥提交 / 等待墙钟 | 0.15522 / 0.01668 |
| 候选 checkpoint 保存、校验、恢复 | 52.666 |
| 候选含 checkpoint 的诊断包络 | 160.787 |
| 本次完整验证 worker，含冷准备和全部证据 | 852.965 |

相同 100 ms 的神经 advance 本次比值为 2.1665；计入候选一次 checkpoint 后的诊断包络比值约 1.46。不能据此声称完整电影、光学管线或其他 GPU 达到相同倍数。阶段墙钟是代码区域计时，不保证相互独立；CUDA copy 与提交/等待计时重叠，不重复相加。原参考臂传输了 138,848,976 字节 compact release；新桥共 1006 段、70,741,920 字节。原 native 210 个新接受区间、3180 次 RHS；候选固定 100 μs 加真实输入截断，未改步长。

本次执行覆盖 24 个比较时刻、44 个时刻/输入侧组合，所有指定输入边界左右侧均实际存在，保存的 point 文件 SHA256 已复核。数值门每类 7832 行：严格门失败 2004 行，B 门失败 132 行。B 失败集中在 `w`、`z2`、`q`、`release`；第一个 B 失败是 6.406666666666633 s 的 z2（最大绝对误差 5.46326e-5，最大预算归一化误差 1.3653）。最坏 B 项是 6.4733333333333 s 的 w，绝对误差 1.92409e-4、预算归一化误差 4.8085。w 的预算仍为原 z1 预算经 |A| 传播，没有放宽。

同一候选轨迹的 CPU/GPU release 50 项比较全部通过，最大绝对差 9.43690e-16；1005 次、每次 5 个 HC 的延迟数值读出通过 B 门，使用原保存数组重算严格门也全部通过，最大绝对差 1.26811e-10。两臂各 1200 次真实 PCG64 抽样在严格/B 原公式门均通过，实际事件身份、pending 事件身份、最终 RNG/history、component ticks 和 sample index 一致。**最终完整延迟队列的时间节点不完全一致**，因此不能宣称历史队列合同全部通过。该差异和上述数值失败均保留，不另改观察政策或阈值刷通过。

checkpoint 在 6.400031 s 保存并恢复一个尚未消费完成的 native 区间；R/E 锚点、w、固定 K、CPU inner、原事件/RNG/history、延迟和 packet/四类水位的完整 payload 位级一致，后段实际使用恢复态。两臂完整 monitor 分别覆盖 72/73 个状态组，均无域失败；动态电影的静态背景窗口 `passed` 标记为 false，不能将它当成新背景已获资格。资源守卫无停止：采样全 GPU 峰值约 7.83 GiB，进程峰值 RSS 约 6.09 GiB。

不追加实验、不改步长、参数、种子、输入或冻结误差预算。数值速度收益、科学验收、生理有效性、大资产传输和笔记本可运行性分别报告；本候选尚不能作为原后端的科学等价替代。

## 延迟队列修复（2026-10-24，独立于上述 r4 失败记录）

`cone_hc_observation_queue.py` 定义显式队列合同 `cone-hc-explicit-observation-queue-1`：只在精确请求时间及原外层区间终点入队。外层终点包含原观察网格、事件和真实输入边界；额外 native/CPU 求解器接受节点不因靠近网格而入队。两个相邻 ULP 若分别属于真实请求或输入边界，仍保留为两个节点；同一边界的左右侧观察不重复入队。原 2.82 ms 延迟、线性插值、剪枝及即时反馈均未修改。

旧 `cone_hc_split_runner.py` 保持原字节，使合法生产 checkpoint 仍先通过原严格加载。A/B 入口随后调用显式 `upgrade_split_reference`，产生包含新队列合同和旧模型来源的新模型身份；继承历史逐项保留，不删去旧近网格节点。旧状态不能直接作为新身份使用。新参考仅用于此验证入口，生产默认没有替换。

新验证协议为 `cone-hc-reduced-real-short-protocol-2`，新 reduced checkpoint 为 `cone-hc-reduced-complete-binary-checkpoint-2`，封存并检查队列合同。旧 v1 reduced checkpoint 不伪装兼容；历史 r4 报告和 checkpoint 保留。每臂新增 `*_queue_times.json`（完整秒值与 float.hex）；完成比较时新增 `delay_queue_comparison.json`（两臂完整列表、数量、首个不同节点和缺失项）。异常退出时标明文件只代表最后成功返回的状态，不声称取得异常发生瞬间的内部状态。

相邻 ULP 回归另发现并修复候选请求漏记：只有实际将执行事件的外层终点才延后读出，其旁边的真实请求不再被 `1e-10` 邻近判定吞掉。未改变事件触发顺序、native 积分、参数、误差预算或随机数。

本次共 51 项针对性 CPU 检查通过（含必要复测；没有复跑 GPU A/B）。新检查覆盖相邻 ULP、真实历史的 38 ULP 额外节点、输入边界左右侧、显式模型身份迁移、完整时间戳失败证据，以及 3 ms 合成调度下 checkpoint 恢复后穿越输入边界、队列/延迟读出与不间断执行一致。实际 codec 使用 Torch CPU tensor；合成 native packet 不代表 GPU 或科学数值验收。初次测试有临时目录缺失的 setup error，随后 ULP 读出回归暴露上述代码错误，原日志均保留于 `reports/cone_hc_queue_repair_20261024_v1/`。

`w/z2/q/release` 的 r4 超预算仍是未解决的科学失败，候选保持非默认、不可拟合；本次没有声称修复后的真实全链队列已通过。

## 复核与笔记本运行命令

以下命令在项目根执行，`python` 必须指向使用者明确选择的 Python 3.11 环境。不要调用旧生产入口，也不要删除生产 `REQUEST_STOP`。

```powershell
python -X utf8 -B -m pytest tests/test_cone_hc_reduced_native.py tests/test_cone_hc_reduced_bridge.py tests/test_cone_hc_reduced_interfaces.py tests/test_cone_hc_observation_queue.py -q -p no:cacheprovider
python -X utf8 -B tools/validate_cone_hc_reduced.py --prepare-only --output reports/laptop_reduced_preflight
# 只有本机实际硬件、完整外部资产和 DLL 环境满足条件时才执行：
python -X utf8 -B tools/validate_cone_hc_reduced.py --execute --output reports/laptop_reduced_preflight
```

前 19 项 native/桥 CPU 检查不导入 CUDA；新增接口/codec 检查使用 Torch 的 CPU tensor，不能替代真实 CUDA 检查。`--prepare-only` 仅核对小 manifest 并封存协议，不构建模型、不初始化 CUDA。新入口的实际推进只覆盖同一真实输入 6.4→6.5 s；候选事件门失败会提前停止。`neural_advance_seconds` 统计实际 advance 调用及其完整监控/选定读出，checkpoint 和冷准备另列。CUDA event 单独测量紧凑 D2H 复制；CPU submit/wait 时间也保留，两者不可重复相加当成新总耗时。

迁移所需的外部身份闭包至少包括以下内容，不能仅复制某个 NPY：

- `data/atlas/human_disk3mm_functional_hc_v1/` 的实际 circuit manifest 及其完整引用；`human_disk3mm_typed_tables_v1/`、`human_disk3mm_input_plan_v2/` 和被图引用的固定几何/连接文件。
- `tools/vr_teacher_handoff_v1/intake_drive_20261023_v1/received_asset_binding.json`、其 `local_intake_root` 所指的 stimulus index、实际使用的 gray8 chunks，以及 handoff 绑定的清单。teacher 不会在此次验证中重算。
- `reports/first_useful_split_capture_20261023_v1/` 的 `round_budget.json`、既有 walltime-policy/approval 文件、`input_policy_approval.json`、`actual_global_reference.json`、`linear_inputs/frontend_receipt.json` 及该 receipt 列出的完整文件；`source_execution/source_graph_binding.json`、`movie_input_definition.json`、`movie_chunks/session.json`、`chunk-000000/commit.json` 与其完整 checkpoint。checkpoint 所有 NPY 和原 public JSON 都须保持原字节。
- 由 `tools/vr_exact_input_catalog.py` 的 `require_block(6.4,6.5)` 实际返回的光学/转换 receipt 和数组；复制其 manifest 引用的完整块，不能把未完成的未来 packet 当成可用输入。
- 原本被忽略的机器配置 `configurations/iset_lcd_apple_mw_v1.json` 及输入政策绑定的显示/全局计划。只消费缓存时不启动 MATLAB；这些来源身份文件仍用于校验。新的光学生成另需 MATLAB 和 ISET，但不属于本入口。

原图绑定和 intake 清单包含绝对路径。当前严格 loader 没有经过跨盘/跨根目录重绑定验证，不能手改路径后重算哈希绕过身份检查。不同根目录或非 Windows 机器可先做 CPU/小 manifest 检查；真实后端须另有可验证的路径重绑定和 cuSPARSE 适配后才能运行。笔记本 GPU、驱动、可用 RAM/VRAM、wheel 和 Windows DLL 是否满足要求均未验证，未做资产传输。

提交只带代码、小型必要配置和本文；上述研究资产、报告中的原始数值、缓存、MAT、凭据及机器配置不进入 Git。必要既有运行依赖保留其当前字节，不能把这些历史改动描述成本轮重新实现或全部回归验证。
