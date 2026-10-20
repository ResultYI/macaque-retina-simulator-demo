# 猕猴中心—周边连续视网膜模拟器：当前架构与交付说明

## 等价执行性能修正与真实配对结果（2026-10-18 UTC）

A/B 已在真实旧HC过渡区图完成顺序测量：RF框(.96,1.04,-.04,.04)mm，879437个依赖节点、34组件、530929个锥细胞、344287个HC、4967023个注册状态标量。细胞、图、权重、方程、FP64、原DOP853设置、事件/监测时刻、历史、延迟、RNG和停止条件保持原样。没有OPL卷积替换、多速率、分裂或降阶。默认仍未切换，新功能HC中央图的生理/背景验收仍未完成。

| 配对范围 | 原路径 | 修正路径 | 本次速度比 |
|---|---:|---:|---:|
| 已收敛完整状态，1ms背景推进，schema未缓存→全部A优化 | 7.362557s | 5.381850s | 1.368× |
| 同上，schema已缓存，仅state-only增量 | 6.132197s | 5.381850s | 1.139× |
| 同一已收敛状态，2ms刺激与完整所选记录，固定ID索引缓存 | 13.887739s | 6.335494s | 2.192× |
| 相同工厂初态，100µs完整GPU推进与记录，仅B紧凑桥接 | 0.514364s | 0.583345s | 0.882× |

A复用原2.099999999999879s收敛检查点，不重跑适应。1ms三臂均为25次原监测、12次solver、180 nfev；完整状态、所有监测结果、时钟、RNG、history/delay/input均完全相同。2ms记录的状态、输出、单位、ID顺序和元数据也完全相同。A先丢弃一次完整窗口预热，再各计时一次；B每个进程均先丢弃一次完整100µs窗口，再各计时一次，保留全部17 nfev和原观察时刻。B还比较原始/扰动两组完整RHS及全部实际native外向消费者驱动，初态/终态每个4967023标量按float64位模式独立复查，完整因果辅助JSON逐字节相同，全部所选输出及单位/ID相同。两套大网从未并存。每方另一次同态100µs cProfile复算已核验相同，profile开销不计入上表。单对数据没有统计分布，不能相乘或外推完整背景速度。

实现为：`engine._state_schema_hash`只缓存同一深层不可变schema；`background_advance.StateOnlyBackgroundRunner`在源码固定且读出无副作用的机制上跳过空trace/SimulationResult/元数据，其余积分与完整监测照常；`recording_indices.IndexedRecordingPlan`缓存固定不可变ID索引，保留顺序/单位/返回隔离，可变结构回退且缓存有界。GUI与匹配响应调用者已接入，并分别记录本次准备、同身份恢复与刺激耗时；原本已有的背景缓存不重复算作新收益。38项联合回归通过（新A/B18、此前schema7、已有时钟/前端13；不是38项新生理验证）。小模型1.44/1.22×为较早初步数据，真实结果以上表为准。

B是显式可选`cuda_compact_native_joint_dop853_float64_v2`：所有实际编译外向release消费者形成650列确定性并集，私有inner CSR只改列地址，系数和行内求和顺序不变。每RHS外向release由4247432 bytes降到5200 bytes；完整cone/HC及A/B反馈仍在GPU求解，监测、history和记录仍用完整模型。light每次仍验证原端口合同，按实际float64位比较复用缓冲，支持值变更/连续输入而不依赖对象ID或ZOH标签。传输字节比不是墙钟倍速。资源候选为`configurations/disk3mm_validation_resources_v4_cuda_compact.json`，预算与v3相同，未切默认。B有独立执行/模型身份，旧CPU/v1检查点仍拒绝，不迁移或重标。

严格恢复曾正确拒绝当前编译器：AII空和修复使`assembly.py`从旧SHA4fed0da9…改为bbe18564…，而assembly身份包含编译器文件原始字节。仅核对8个积分文件不够判断旧检查点兼容。本次在独立诊断进程中实际执行审阅提交3333811中留存的原编译器源码，路径`reports/current_wip_review/frozen_v7_compiler/assembly.py`，完整SHA为4fed0da9eee54bb91b8e0d7823b9e1c9e34204d95bfb2a754e3b72589ebbd7c7；没有伪造hash、改写检查点或覆盖当前源码。另18个原运行时/CUDA文件与该提交逐字节一致。原检查点全部绑定差异为空、完整严格恢复通过，文件SHA仍与此前profile记录一致。B沿用同一冻结编译器，自己初始化原工厂值，初态全值及图/参数/schema/clock/input/solver绑定均与原路径相同；没有把旧检查点导入v2。

装配/恢复仍明显影响周转：A装配199.65s，runner与绑定52.16s，严格完整恢复50.81s，均单列，历史四小时准备不计入本次恢复耗时。1ms state-only的积分内层为1.614s，整窗5.382s，其差额包含原监测、事件、状态处理等，不能全归为可删元数据。B未profile计时中原积分内层0.157s→0.139s，整窗如表；全部profile函数和交叠关系保留在case.json，不能把inclusive时间相加。监测/记录的完整CPU求值、GPU同步、装配及完整检查点序列化仍是残余成本。 本次B整窗增加13.41%，因此未证实整体提速，显著收益来自A；B继续保持可选候选。B未profile计时的非积分差额约占整窗76.2%，含记录/事件/状态处理等必要工作。诊断profile中57次Tensor.item同步、5次完整network.evaluate（包括native evaluation_context）突出；这些inclusive时间相互交叠，profile自身也改变GPU发射节奏，不能据此直接推算常规吞吐。

用户明确授权后，核对PID/父链/创建时刻，仅关闭Androws46220→ABoxSVC44236→ABoxHeadless45324。均无可正常关闭的主窗口，普通Stop-Process按授权结束这三个目标，未提权绕过OS权限，未动其他腾讯程序、浏览器、启动项或安全设置；延迟确认退出。关闭后物理可用约17.08GiB、提交余量约12.50GiB。原15GiB准入为单进程7GiB工作预算（历史RSS峰6.062GiB）+8GiB物理余量，并新增至少7GiB提交容量检查；两轮开始前均满足，运行保持原余量。

证据：`reports/performance_real_prepared_A_v4/result.json`、`background_binding_differences.json`、`legacy_bridge_case/`；`reports/performance_real_compact_B_v1/case.json`、完整初末状态NPZ/辅助JSON、两组RHS/驱动和输出、同态profile；`reports/current_wip_review/handoff_status.json`汇总独立文件复查。关闭收据为`user_authorized_androws_close.json`，较早内存阻塞/失败尝试及旧结果全部保留。固定公开审阅提交3333811未覆盖，后续本地修改尚未推送。

节点选择仍先于工厂、状态、ISET和积分。非零有向BFS证明旧过渡区879437、新功能HC中央894425节点均无冗余选择；仅第一个midget ON也分别需875557、890625节点，因为原native反馈分别为875216、890559节点的单一强连通分量。这是当前有限图的依赖事实，不是小ROI生物学必需全网的证明，也不授权裁剪反馈。功能HC部分背景保留在绝对t=.124s，未收敛、未重启；134个无RBC化学输入AII的项目空和（118有gap、16无gap）不等于真实无支配的测量结论。人锥细胞借用、有效动力学/亚型、AII支配等科学缺口保持明确。

## 严格恢复唯一热点与反序复测（2026-10-18）

本轮只消除严格恢复中两次大型JSON往返。`engine.restore_payload`复用原动态状态检查；`checkpoint_loading`仅对原HybridRunner.restore源码固定且适用的分支使用已解析对象，保留外层/内层hash、模型/参数/schema、输入相位、有限值/shape、绝对时间及收敛检查。可变RNG/history/delay/输入容器仍深复制，返回envelope与state互不别名。自定义/未审阅restore或RK2回退原路径，坏文件仍直接拒绝。`BackgroundCheckpointStore.load`默认auto，回退可设置`RETINA_CHECKPOINT_DECODER=legacy`或显式`decode_mode='legacy'`。没有改状态/检查点格式或模型身份。

真实311445018-byte原v7检查点，单进程先丢弃一次完整legacy载入预热，再按parsed→legacy反序各测一次：50.844s→38.638s，时间减少24.01%，速度比1.316。文件读取仅0.071/0.068s，首层JSON解析约3.89s，原几何校验约4.7–4.8s、envelope hash约14.7s仍保留。状态解码/校验27.43s→15.12s；旧路径额外生成并解析约328MB及97MB JSON，新路径只保留首次文件解析。不得把整个严格恢复叫作磁盘I/O。

本次装配204.49s，其中两次顶层RetinaBuilder.build分别45.16/45.57s（25组件→34组件，都含native）；不能据此直接删掉中间构建，当前基准传递仍依赖它。执行器15.36s、初态1.44s、完整绑定34.72s、既有光输入缓存严格加载6.81s。一次性实际setup共263.06s；加上预热后的parsed恢复为301.70s的分段预算，非新视频冷启动时间。2ms记录也按新indexed→旧ordinary反序，每方先丢弃一次同输入完整窗口：15.160s→6.145s（2.467×），均23 solver/334 nfev。全状态位模式、身份、RNG/history/delay/input、输出单位/ID及元数据一致。基准进程总497.10s包含预热、重复核验和工件IO，不是生产推理时间。33项相关回归分两日志通过（31+新增2）。没有再跑B或长背景。

CPU monitor/readout暂未合并：compiled RHS不直接调用Python evaluate；监测sampling=True与普通record不同，同一时刻历史转换前后也不是同一状态。仅按时间缓存会误用结果。本轮唯一建议路径为既有后端（该图为GPU-v1）+A+auto严格解析恢复，B继续可选且不默认开启。证据`reports/performance_strict_restore_reverse_v1/result.json`及`reports/parsed_checkpoint_regression_v1.log`、`v2.log`。

## 0.5秒真实视频、有界输入与内部状态回看（2026-10-18）

用户已授权实际0.5s视频试跑；目前仍等待真实本机路径和片段起点，尚未运行该视频。项目现有0.12s/50fps/6帧/480×270合成QA和8Hz光斑不能充当真实素材；历史引用`C:/Users/PC_User/Videos/原视频/第零回广播.mp4`为ObjectNotFound。没有下载、循环或替换输入。

原输入的19.87GiB是仅0.5s的保守预算估算，未实际分配。`disk_frontend.py:130–132`硬编码0.0001s，得到5000个窗口、5001个边界，按5001×(530929 cone+2306 rod)×8=21,333,665,880B检查；真实原始数组分别为float64 `native_counts[5000,530929]`及`rod_values[5001,2306]`，合计约19.8646GiB，转换时还会产生副本。128MiB来自`iset_frontend.py:13`。MATLAB先按视频帧/注视去重计算OI，随后`export_disk_mixed_reference.m:73–79`将相同计数重复写至每个100µs窗口。导出器明确只做捕获，无上游光转导、光子噪声或跨帧动态状态；100µs是项目输入格点要求，不是该ISET光学过程要求。

新增显式`StreamedDiskMixedReferenceFrontend`与`streamed_input.py`保存每个实际(frame,gaze)的单窗口原始计数、完整原100µs索引和分块原rod采样。固定注视、从0秒开始的50fps/0.5s例子为25个光学曝光×148个固定tile=3700任务，在原batch执行器的一次tile批次中处理；不重复成37000任务，不创建all-cone×all-time矩阵。连续变化的注视不会被合并或降采样。严格参考/背景绑定继续使用原one-window全局packet读取器，额外分片及曝光provenance均冻结hash并校验，数值LRU合计上限64MiB，每分片仍受128MiB预算。原光学导出器、`disk_frontend.py`、LMS系数、全局增益、图、状态方程、求解器和背景身份均未改动。新存储源身份单独记录。

实际通道是ZOH窗口均值。特别保留原`np.diff(edges)`的float64窗口宽度，再按原顺序执行counts/width/alpha；提前除名义100µs并复用速率会产生微小差异。所有原breakpoint、searchsorted(right)、最终窗口夹取、杆通道sample端点及现有背景重定位/左右极限规则均保留。神经状态跨所有帧和块持续，不逐帧重置。

验证：13项直接回归通过（`reports/streamed_input_regression_v1.log`，14.88s），覆盖QA视频所有bin/中点/边界两侧、注视key、坏hash/布局/值/预算拒绝、完整准备与严格重载、背景重定位、恢复保留和完整checkpoint协议及状态读取。一次真实pinned ISET小对照在2 cone+2 rod、20.3ms、203窗口、两帧上逐位一致，前端及背景签名一致，42.8656s；证据`reports/streamed_input_actual_ISET_QA_v2/result.json`。这是合成QA几何/视频的输入存储等价检查，不是大图推进、0.5秒视频、物种标定或新functional-HC验收。沙盒首次MATLAB启动遇MathWorks5201，未进入光学计算；一次获准的沙盒外同安装运行成功，未改环境或安装软件。随后仅增加分段计时，无数值改变。

专用入口：`python -B -X utf8 tools/run_video_05s.py --media <真实视频路径> --start 0 --output reports/<新目录>`；`--resume`只继续同目录、同素材/片段/源/旧编译器及严格相符的完整checkpoint。该入口保留原v7过渡区879437-node图及2.099999999999879s收敛背景，实际加载留存旧compiler，任何身份不符立即停止，不启动长背景；不是新functional-HC中央图验收。入口及其状态记录/读取和checkpoint函数经过小fixture回归，完整大图视频执行尚待真实素材。独立主GUI默认未切换到此实验入口。

记录为正式观察队列650 cone、616 HC及全部已选内网/接触状态，共37449个注册标量×501点×8=150095592B（143.1423MiB原始数据），1ms采样且分块不重复端点。状态key及单位/ID写入结果manifest，可由现有ResultStore/GUI历史读取。原timestamped spike events保留在窗口metadata；完整起点、终点及60s wall间隔滚动checkpoint保留全依赖图动态数组、RNG、clock、history/delay和输入位置。恢复时超出最近checkpoint的窗口保留归档后重算；`STOP`文件可在块边界停止。没有逐时刻保存全网辅助缓冲，也没有实现全图任意连续时间checkpoint replay viewer。其他点或细胞不能由143MiB观察数据直接推断。

一次前向可缓存同视频/同输入/模型/记录设置，之后回看读取结果；换视频仍需新推进。入口分别记录装配、ISET准备/缓存、runner、严格绑定/恢复、推进和记录、结果写盘及checkpoint写盘；前端还区分decode/crop、MATLAB tile batch和校验/绑定，decode为准备子项不可重复相加。没有冷ISET大图或0.5秒神经wall-time测量，不按42.9s小QA、100µs GPU或历史668.282s静图整体交付线性外推。预检`tools/check_video_recording_budget.py`当前仅剩真实素材路径阻塞。

## HC等价热点修正与短窗口实测（2026-10-18）

按新Pro主体方案仅做一个等价工程改动，不改HC方程、A/B权重/接线、FP64、DOP853、时钟、RNG/history、监测准则或记录。未重复SCC/闭包搜索，未加HC-HC gap、局部域截断、Schur/线性化、低秩或分区积分。旧0.514→0.583s是GPU v1与compact GPU比较，不是GPU对CPU，本次另做同进程新配对。

实际原图仍为530929 cone+344287 HC、4967023状态。一次完整100µs窗口的诊断保留原数值监测和记录，17次RHS；三次SpMV的CUDA event总量分别Bh10.5635、AVbar9.4567、Az1 10.5276ms，合计30.5477ms，约占native GPU算子时间17%，不是主要成本。对同一保存trace按External id/父CPU算子归属分解，纯device kernel分别为9.2393/9.1382/10.2014ms，native逐元素数学112.1320ms，native域/输出检查34.9247ms，重复joint全向量finite检查11.8000ms；此kernel分解与event跨度不是同一量，不重复相加。HtoD116.16MB/6.02ms、DtoH195.63MB/13.45ms；设备内部拷贝另记，不能称全是PCIe流量。136次cudaStreamSynchronize的CPU跨度162.06ms包含等待前序GPU工作，不是纯同步调用成本。

带profiler的全窗口1.1978s中，原数值monitor4次0.4069s、CPU model.evaluate9次0.3176s、CPU monitor RHS4次0.1284s、CPU inner17次0.0232s；这些包含关系重叠。diagnosis/trace只定位成本，不当吞吐。CUDA/CPU源码算子已在构建时绑定；无每RHS重新归一化。实际Numba执行数组内A位于两个不重叠的CSR区域，权重、indices、indptr均为独立副本，单份60800288B（57.984MiB），整个编译计划数组296670976B。共享有确定内存价值，但本轮未混入第二个布局改动或强制CPU后端，也未把它当成巨大提速。

新增显式可选`cuda_native_joint_cpu_finite_validation_float64_v1`（`gpu_joint_validation.py`）。原pinned CudaNative已完整验证native导数及release；内网导数本来在CPU/Numba以float64计算，因此在CPU检查全部inner导数，再原样float64上传，即可覆盖原joint最终检查的全部分量。省去拼接后对4967023标量重复做GPU isfinite/reduction及每RHS一次主机判定；保留native原有两次检查、全部release传输、端口校验和求解器原检查。内网external端口/期望channel集合预绑定，无跨RK stage动态缓存。源码pin失配、自定义native、非有限值或错布局均拒绝。普通make_runner工厂可显式选此候选；默认配置、v1及compact旧路径保留。

一次完整warmup/臂丢弃，候选先、原GPU后，同一模型实例的静态算子可顺序复用。均100µs、1 solver、17 nfev、相同4个observer点、原全状态数值monitor及同一indexed记录：

| 本次未插桩窗口 | 原GPU v1 | CPU finite检查候选 |
|---|---:|---:|
| 完整推进、原monitor和记录 | 1.1297707s | 1.0552959s |
| 其中积分 | 0.3685838s | 0.1700224s |
| 原数值monitor | 0.3948836s | 0.4066298s |

完整窗口约1.0706×、wall减少6.59%，只有一对，不作为统计分布或长背景/0.5s吞吐预测，也不把74.5ms差值全部归因于11.8ms的GPU kernel。全部4967023个最终状态bitpattern、原monitor报告、输出/单位、完整clocks/RNG/spike/history/delay/input辅助数据及执行点一致；插桩结果也与未插桩原路径一致。9个最终不同回归case通过；日志保留最初QA错误选择R作负值拒绝样本、后改为源显式约束X的修正及复测，未更改模型域规则。记录/非backend metadata与候选自身checkpoint往返由小图回归覆盖。

本进程峰值RSS5232607232B（4.874GiB）、GPU peak allocated1764893184B（1.644GiB），最后可用物理约12.10GiB，原8GiB余量未变。两臂共享静态算子，未测独立每臂峰值或实际CSR内存节省。装配162.6615s、runner/init16.6341s、既有输入严格加载6.4814s，未做ISET或长背景。

本次fresh pair未设置OPENBLAS_NUM_THREADS，旧记录为1；此字段进入现有执行身份。科学binding与保留旧GPU case逐字段相同，但本轮原model08644122…及候选96e22c12…应按本次独立身份解释，未恢复旧背景、未拼接旧耗时。候选有独立执行/model hash，旧背景checkpoint应拒绝，不迁移。另修视频入口在NumPy载入前从留存v7执行记录复现OPENBLAS=1；若NumPy已载入且设置不符则要求新进程，防止将错误进程设置套上旧身份，未升级或改变conda环境。

结论：停止本例SpMV融合调优；本轮交付的是小幅实测收益的验证/同步优化，保持显式可选。当前旧背景/真实视频仍走原v1；更大成本在必须的CPU监测/读出及native逐元素数学，后续若继续，只考虑同一次同状态计算中的复用/融合并保留完整校验，不能跨stage缓存或偷换模型。真实视频路径仍待提供，0.5s未执行。证据统一在`reports/hc_equivalent_short_v1/result.json`及同目录diagnosis/trace/两臂完整数组与辅助数据；测试`reports/gpu_joint_validation_regression_v1.log`。

## HC计算诊断快照：后续方案边界（2026-10-18）

旧图A为344287×530929、B为530929×344287，各3627874个非零；functional中央图A为344901×545658、B为545658×344901，各3577939个非零。每cone有R/E/X/C/V/gi/Vbar/z1八状态，每HC有z2/h两状态；每RHS计算Bh、A·Vbar、A·z1三个稀疏乘法。反馈为q=sigmoid((V−Bh−Vk)/Vn)，HC池化适应为((A·Vbar)/VI)^mu，z2/h沿原tau2/tauh演化；没有独立HC↔HC电耦合算子，内网AII等另有电耦合。S/H2仍共享H1项目动力学，不能称独立实测校准。

现闭包用预存分量加速再逐route反向追溯。独立非零有向BFS/SCC核查旧/新图无冗余选入节点，native分别形成875216/890559-node单个强连通分量；只请求一个midget ON输出仍需875557/890625节点。因此仅将无向分量换成SCC不能缩小这两个具体图，不代表生理上小ROI必需全盘。源证据在既有`closure_and_AII_finite_support_audit_v2.json`。

CPU已有Numba并行逐行CSR和native行融合，无fastmath/跨行归约；GPU以float64常驻A/B和native状态，保留DOP853，内网逐阶段经CPU/Numba。权重和归一化只在构建时确定。CPU Plan.csr为A两次乘法重复存储权重/indices/pointers，属于潜在等价内存/构建优化点，不能误称每RHS重复归一化。GPU每阶段有效性.item同步及内网传输、全量监测/读出仍有成本。

既有旧/compact桥接100µs实测全窗口0.514/0.583s、积分0.157/0.139s；compact整体未提速。诊断profile的57次.item共0.381s包含此前GPU工作的等待，inclusive不可相加；三次SpMV的后续实际分解见上节短窗口实测。背景监测要求全注册状态，正式记录仅观察队列；旧长背景非积分约79%不可当成HC纯算术。后续严格等价候选为CSR常量复用、保留逐行加法次序的融合、明确同步边界及读出复用；需短对照和完整状态/causal辅助校验，已实施的唯一验证/同步候选及其适用边界见上节；未改HC数学。局部域+边界/衰减误差、降阶、改方程/积分器、删边或高斯替换均属科研取舍，等待具体方案及用户授权；未开新长背景。

## Current result: human cone reuse candidate blocked (2026-10-17 UTC)

The single main configuration remains unchanged: `configurations/retina_platform_v1.json` still selects the earlier finite local macaque route `configurations/atlas_isetbio_source_stop_shared_v1.json`. The new candidate is `configurations/atlas_human_borrowed_disk3mm_v1.json`; it has **not** passed neural acceptance or replaced the default.

At the user's instruction, the macaque generator and old automatic waiter were stopped after a complete iteration5855 checkpoint. Current/best coordinates, all5855 history rows, initialization, RNG information and implementation identity were saved and verified in `reports/disk3mm_build_20261016/macaque_paused_20261017T140036Z/manifest.json`. The checkpoint SHA256 is `f7eab3454e757ce6b9a12dd1705d6884154fa3a87fcdeb80aabf2d35cf98fc95`. This is a paused, resumable result; the source stopping condition was not reached. No automatic fallback/resume is active.

The candidate imports the locally verified official ISETBio right-eye nominal58-degree human cone lattice (actual MAT FOV58.5), pinned commit `d2924c2fe1d41c7dc5af1006d8081735b08f1b35`, official asset SHA256 `54eb0a9ef90511ca72782a194a7c6f17d5404e05794f7e4872c808d138f45728`. Its original coordinates are negated and converted from microns to millimetres using the official importer convention, then cropped to the existing complete support. There is no warp, density resampling, Packer fitting, jitter or gain correction. Original source row identities are retained. The complete square support contains705259 cones; the owner RF disk R=3mm contains505162; the fixed active optical/connection support contains545658. All active aperture neighborhoods are inside the complete support. Existing0.79-times-median-six-neighbor apertures were recomputed on actual imported positions.

This is an explicit **human-borrowed approximation** on the composite macaque project frame. HC/rod/RBC/AII/parasol masters and existing physiological/effective coefficients remain. Cone-dependent IDs, BC/midget geometry, connected RF centres, finite contacts and optical tile/cache identities were rebuilt. Parasol source RF seeds and identities remain; their BC-dependent contact anchors are rebuilt by the existing rule. S labels retain macaque absolute macro quotas and legacy rhoS/rhoPacker selection scores on actual human candidates, with the named L:M1:1 rule. These scores are project selection weights, not measured human subtype probabilities. The actual S provider is the frozen `deMonasterio1985_mulatta_effective_S_v1` binding; an unused inherited `S_partition_rule` description in the atlas profile is historical metadata, not the executing S operator. Packer is also retained in the existing Wool temporal iso-density adapter; it does not generate the imported human coordinates. The earlier fixed-position source audit found human/Packer density ratios around0.57-0.72 at1mm and0.62-0.99 at3mm across meridians; these are source-function comparisons, not a physiological equivalence test.

The actual frozen graph fails the unchanged global gates:

- **7770 owner cones have no native HC feedback partner**, including78 directly in the central0.02mm owner. The exact central dependency view contains1630 unsupported native feedback rows. The fixed HC master, per-ID finite candidate arbors and no-nearest-fill policy are retained.
- **533 owner midget candidates per ON/OFF polarity have `INCOMPLETE_MIDGET_CONVERGENCE_FINITE_SUPPORT`**, at RF radii1.1122-2.7576mm. These are interior finite-pool deficits, not a halo-only count. Targets and support were not relaxed.
- The central(0,0)/0.02mm view has5975 nodes and is unsupported. Transition(1,0)/0.08mm and near-peripheral(2.85,0)/0.1mm views pass graph qualification with879437 and879720 nodes, respectively. This establishes graph qualification only. New human-route MATLAB/ISET capture, background preparation and neural responses were **not run** because the global scientific gates already failed.

No partner filling, arbor enlargement, convergence-target change, cropped-weight renormalization or biological absence claim was used to hide the gaps. A scientific decision is required for this combination of geometry and retained rules. Continued macaque relaxation alone is not evidence that these compatibility gaps will disappear. The local effective/engineering routes remain distinct from this candidate.

Authoritative results and exact bindings: `reports/human_disk3mm_build_20261017/final_delivery.json`, `feedback_gap_diagnosis.json`, and the frozen global graph manifest SHA256 `617247a7819cb1c6334ec7bff34ba0946034f4639ebe758c7c291fabb34d9074`. The final diagnostic image `geometry_blockers.png` plots actual geometry gaps; it is not a neural response figure. The old default SHA256 remains `054c057a17f6f66165f4a923dd2edf714b47454f3b9d26076c83c3c4af557006`. Fixture checks passed:13 existing ISET geometry cases,1 new explicit human species/asset/transform rejection case and15 disk structural/input/identity cases. The earlier focused background clock/RNG/continuation checks passed2 cases. These checks do not certify physiological calibration or whole-disk dynamics.

The Pro semantic audit remains applicable: BC `activation` is the dynamic `terminal_release` of `HistoryBipolarTerminal`, which feeds AC/RGC; `generator_activation` is diagnostic. This implementation audit is separate from the unrun new-human-route dynamic acceptance. `conditional_rate` is hazard lambda, Bernoulli-bin expectation is `(1-exp(-lambda*dt))/dt`, and `window_rate` is realized count/window. vH C is author-scale calcium concentration in Supplement1 equations7-9; `I_os=X^nX` is current. Legacy serialized C labels remain for checkpoint compatibility and do not establish physical micromolar units. The2.82ms delay applies to `observed_Vh`; BC input uses pre-readout `outer_native.release`. Earlier1.2ms smoke results do not establish full pathway function.

Wool Table1 uses D=Rc=sigma as radius with `exp[-d^2/(2*sigma^2)]` truncated at Rc. The existing project kernel `exp[-(d/Rc)^2]`, support `min(40um,Rc*sqrt(log100))`, finite candidate selection, fractional weights and normalization are separate project rules. Borrowing n/scale does not establish source-equivalent kernel geometry. Source references: [vH Supplement1](https://pure.rug.nl/ws/files/2938731/2005JVisvHaterenSupp1.pdf), [Wool article](https://pmc.ncbi.nlm.nih.gov/articles/PMC5815352/), [official ISETBio mosaic asset documentation](https://github.com/isetbio/isetbio/blob/d2924c2fe1d41c7dc5af1006d8081735b08f1b35/docs/sdr-mosaic-data.md).

RBC=rod/18, AII=5000/mm^2, HC candidate geometry, shared S/H2 kinetics, BC/AC/RGC effective dynamics and physical soma displacement retain their documented project limitations. Whole3mm neural dynamics, stochastic stationarity, independent physiological validation and selective pathway causal controls remain unverified. No default adoption, new Git commit or push is represented as completed here. Historical sections below remain the record of earlier implementations and runs.


本报告更新同一份 `docs/core_design.md`。当前主线是独立的猕猴视网膜模拟器，依据已确认的 FINAL v0.1 和随后核验的 rod–RBC–AII 合同继续实现。没有恢复旧 RetiPath、变化盲或下一注视点预测；没有使用 CNN 代替细胞和连接。没有安装软件、训练模型、拟合动物数据或启动服务。

**2026-10-14历史 GUI 基线**采用 `atlas_population_revision_v1` 与同一份冻结局部 .35mm 人口图，34组件/20层、77状态块/53,759标量的参数、人口、连接、gap、时钟和solver绑定均未改变。日常图像/视频输入现已实际接入获批的 `iset_lcd_apple_mw_v1`：GUI Run → 真实 MATLAB/ISET 导出或身份匹配的捕获缓存 → Python 输入 → retina solver。新背景在2.10秒、1707.00秒墙钟内按原判据收敛；默认1.2ms真实GUI刺激运行通过，复用的是本轮新背景。2026-10-12旧RGB背景与20ms亮暗结果仅保留为历史，不作为新前端的适应/响应证据。一个AII行仍有RBC配额2/3缺额。

2026-10-11 的旧 `retina_contact_geometry_effective_v1`、旧source ROI/工程预览与其结果保持独立历史身份；GUI历史结果可读，但不会覆盖当前Run的混合参考前端，无RGB cone fallback。原VH/Chen等内部机制与生理参数未改，独立外围LM物理参考不拼入本次native主干。全猕猴光学/动力学校准、随机稳态与独立动物验证仍未完成。

## 当前默认与新 cone 全链迁移（2026-10-16）

唯一主配置仍为 `configurations/retina_platform_v1.json`，现选择 `atlas_isetbio_geometry_v1`，经 `circuit_configuration_path` 绑定 `configurations/atlas_isetbio_source_stop_shared_v1.json`。已冻结的5677步、121811点cone master的manifest SHA256为 `5ab2429aa920e9bd07f7a7d1a335edeffbbf8f263ddce731352809e7d6218a3a`。本轮没有重新生成mosaic，没有假造旧新cone逐细胞映射；新命名空间实际贯穿人口图、接触与权重、typed几何、ISET输入和背景身份。

本次神经运行仍限于原owner `[.34,.36,-.01,.01] mm`、RF输入 `[.30,.40,-.05,.05] mm`及40µm halo。活动网络1590 cone、1053 HC、4586 rod、165 RBC、357 AII；34组件、20层、77状态块/58756标量，五类RGC owner均存在。固定HC和原独立parasol ON/OFF master保留；旧rod/RBC/AII master不能覆盖新cone依赖（rod上界相差约1.8µm），仅这三类按原已批准密度和采样规则建立新有限资产。接触、权重、内层表、parasol IPL锚及gap配方重新计算后冻结于 `data/atlas/circuit_source_stop_shared_v1_20261016`；生产装配只加载核验。owner必需输入行缺额0；RBC→AII仍有5/357行实际2/目标3联系，没有改40µm半径或补增益。有限开放外边界不变。

实际验收经Tk Run按钮→固定版本MATLAB/ISETCam/ISETBio→Python网络完成，目录 `reports/platform_runs/20261016T071417135148Z_0348ea84`。本次真实导出新输入，未复用旧几何输入或背景。背景1.5s、538.895s墙钟按原判据数值收敛，77/77状态块覆盖；随后记录1.2ms、1窗口，保存输出均有限，完整运行578.562s。完整背景checkpoint由重新装配的同一几何成功加载，未再次推进时间。随机burn-in完成但统计仍为 `STATISTICS_INSUFFICIENT`，不声明整网随机稳态。混合来源光学、估计孔径、S/H2共享动力学、有效内层动力学及原跨物种/区域转移保持原近似身份，没有新增生理校准结论。

构建核验、转换、GUI及新增无界面入口 `tools/run_retina.py` 读取同一RunConfig和冻结几何绑定。`tools/build_isetbio_model.py --verify-only --config <RunConfig>`可核验；构建仍需显式指定新atlas和输出。基础回归86项、最终共享人口绑定30项、默认切换4项通过。GUI证据 `reports/cone_master_migration_20261016/gui_final/check.json`，同一保存结果的可读图 `reports/cone_master_migration_20261016/response.png`，统一交付索引 `delivery.json`。最初启动受偏好目录写权限和许可通信阻塞，分别通过项目内运行目录及获准的沙箱外MATLAB执行解决。误重采样独立parasol的中间候选已停止并标为superseded，没有成为默认。

默认在候选验收和checkpoint身份校验通过后原子切换。旧main配置保存在 `reports/cone_master_migration_20261016/before/retina_platform_v1.json`；还原此文件即可恢复旧默认，代码仍保留旧人口装配绑定。旧cone、人口图、typed几何、输入、背景和结果未删除或覆盖；旧缓存继续受身份检查，不能强行混用。

用户已批准后续第一阶段目标为**中央凹为中心、半径3mm的二维圆盘，包含中心到近周边**，不代表全视网膜/远周边或新的生理规则。本轮没有扩域。现master的完整可查询方框仅为 `[-.512,.512]² mm`，生成椭圆半轴约 `.799370/.978989 mm`，不足以覆盖3mm圆盘。扩域须建立更大的固定cone总体及映射后依赖的HC/内层/rod总体、typed图和输入分块；不能复制本小patch或降低密度冒充覆盖。原获批Packer中央桥/方向插值、S跨种有效分配、Schein有界桥/外侧恒等、有效动力学和混合光学可保留原近似身份，不应再次当作新增假设。另需决定的是现有profile不支持处的补法，例如中央HC仍零接触时是否改变arbor/密度/接触规则，以及若强制精确r=0细胞时的位移方向策略；范围批准本身未授权这些改变。若要求各类型和各偏心位置均获生理校准，还需补相应动力学及形态来源；现有有效参数全域复用不能证明该验证。缺soma映射仍单列，RF/pedicle/IPL坐标不改名充当soma。

## 有限 Benardete–Kaplan 功能时序比较（2026-10-14）

独立参考在 `exploratory/benardete_kaplan_reference.py`；固定提交 `08aea38003db41c4753990038ed8ac9a75b2706a` 的两份 CSV、SHA 和原文定位保存在 `data/literature/benardete_kaplan_v1/sources.json`。P 的 `NLTL` 除以 `NL` 得到 `TL`；M 使用直接列出的 `TL` 中位数，不强制中位数乘积相等。时间由 ms 转秒，A 为 Hz/单位对比度的输出功能增益；各列中位数组成的向量不是实测单细胞或联合分布。M 原文 p.357 Eq.(3)、p.360 Eq.(4)、p.363 Table 2 已由父任务审阅者核对；`Ts=T0/[1+(c/C半)^2]` 保留平方，c 固定为本次幅值。Vanni 无平方实现另行注明；P 公式高清逐字符核验仍缺失。D 是电刺激到记录端的传播延迟，未赋给原生细胞内部机制。

一次原生轨迹完成750ms的8Hz局部圆斑正弦刺激，后500ms/四周期计分，实际中心5个L/M cone capture基频幅值为3.6162%。按原128MiB预算拆成十个75ms输入快照，34组件/20层的同一状态、时钟、RNG和历史连续推进，无参数、连接、ID或默认配置改变。复用2.10s新ISET背景时，原checkpoint在原上下文通过全部15项身份检查；前端保存位置补丁逆向恢复旧SHA完全一致，且每段新背景向量与旧向量逐值完全相等。本次成功轨迹流程约588.9s，146,156次RHS、原生积分墙钟以manifest记录为准。两个前置失败与绘图库缺失记录保留，未计作原生响应结果。

读数为事件生成用条件强度λ(t)，保留原生spike history。原v1把中心capture对比度当作B–K外部刺激对比度，直接增益叠图有口径错误；旧manifest、图与数组已完整保存于 `superseded_capture_normalization_v1`，明确标为superseded。v2仅后处理原存档：校验实际输入MAT/NPZ、固定LCD gamma/SPD与光度数据，按原gamma线性插值及光谱能量积分恢复光斑外部亮度，均匀外部参考约14.1471088cd/m²，与原记录相对差1.91e−7；未按capture比例估算或新标定。共同分母现为 `Lspot/Luniform_reference−1`，8Hz幅值4.86185%，中心capture的3.61615%仅作前端中介。原生增益依次为midget ON/OFF 41.99/41.00、parasol ON/OFF 371.97/128.02 Hz/单位外部亮度对比度，相应B–K参考为52.94/41.49、323.99/345.62。参考使用同一恢复的外部波形，Eq.(4)中的c固定为该外部8Hz幅值。

原始λ、膜电压、输入快照和最终checkpoint未改变。因分母波形改变，四类原生计算相位统一减1.6288°；P参考相位不变，parasol ON/OFF参考相位分别增6.7999°/8.2906°，来自固定实验c的修正，而不是平移、翻转或拟合。逐周期幅相变化及单轨迹限制保留；仅一次8Hz采样不能估平均放电率传递函数或脉冲核。局部圆斑中心/周边未标定，当前100td/3mm混合参考与原约60cd/m²（P 5mm、约1178td）仍不同；重建的是实际ISET模型场景亮度，不是新增硬件测量。修正版 `comparison_analysis_v2.npz`、`manifest.json` 与 `comparison.png` 为当前分析，原 `native_response.npz` 的native数据仍有效，其旧reference字段由v1归档标为superseded。一致性核验通过，原生求解没有重跑。

## 当前获批的混合参考光输入（2026-10-14）

生产路径为 `retina_platform.backend.SimulationSession` → `MixedReferenceFrontend` → `tools/export_mixed_reference.m` → `load_offline_inputs` / `rebase_offline_inputs` → `HybridRunner.advance`。`configurations/iset_lcd_apple_mw_v1.json` 是唯一活动前端配置；界面简洁显示“混合参考输入 / LCD-Apple / human-MW”。缺MATLAB、源资产、身份或时间/单位合同不匹配时正常报错，不静默切回旧RGB cone点采样。独立作者参考/历史fixture继续保留，不代表本次日常前端。

| 项目 | 实际采用与边界 |
| --- | --- |
| 显示输入 | ISETCam固定版 `displayCreate('LCD-Apple')` 的实测SPD（380–780nm、4nm采样）与1024×3 gamma表。视频数字RGB直接驱动gamma一次，cone路径不先做sRGB逆变换，也不改成线性gamma表。源图像/视频FOV、帧选择与时间戳保持；局部场采用明确的RGB重建与既有halo有限光学支持，属于数值近似。 |
| optics | 显式 `oiCreate('human mw')` 与 `opticsCreate('human mw',0.0015,f_native_m)`；第二参数为3mm瞳孔的半径米。`f_native_m=223e-6/tan(1deg)=0.012775661443659351m`，保持项目223µm/deg。该项是按猕猴布局尺度缩放的人类MW近似，不是猕猴实测波前，未采用Thibos、AO或缺失的M838波前。 |
| aperture | ISET `positionsToSpacings` 在冻结完整支持集2226 cones上取局部六近邻距离中位数，按ID取1465运行cones，`D_i=0.79×spacing_i`。每个cone保留独立pillbox集光面积，导入空间blur为`mean(D_i)`；本例均值3.2074938µm、范围0.9588927–6.7884120µm。位置、ID、类型均不生成/移动，直径是几何估计而非测量。 |
| 光谱/filter/OS | ISET Stockman–Sharpe intrinsic absorbance；OD `[0.5,0.5,0.4]`，peak efficiency均2/3；human Lens密度1在optical image中应用一次，Macular密度0.35及其human偏心度规则在capture中应用，标准ISET OS衰减保留。本局部OS因子约0.589–0.689。人类借用与标准模型convention明确记录，未使用工程Gaussian吸收谱。 |
| reference/alpha | 既有`u=0.5 linear`参考经对应数字RGB `[0.735356983]*3`进入LCD，计算一个固定谱辐亮度增益（本例0.2223670242），所有帧共用。3mm pupil的100td参考亮度为14.1471060526cd/m²。三个同条件L/M/S参考cone使用固定均值孔径与patch中心，不按每个运行cone重新标定。`alpha_class=q_ref_class/I_ref`；期望`R*/window`仅按原窗长转为窗均值，再除固定alpha，在原生τR之前驱动native cone/H1。变色光谱的结果称参考等效驱动，不称一般photopic td。 |
| background/rod | 均匀参考场保留各cone的面积/位置差异，本例参考等效背景约9.12–450.54td；不逐细胞归一成100。该向量、前端/source/参数/图/时钟与solver身份共同绑定新背景缓存。Rod保持原独立sRGB加权R*/s输入与原动力学，不宣称共享完整ISET捕获。 |

MATLAB导出的是无上游photon noise的expected capture，没有photocurrent或上游phototransduction；原H1/H2、BC/AC/RGC、杆通路和神经RNG继续由现有模型负责。L/M/S参考率本例为5221.3575/4084.8355/586.0268 R*/s，来自这套混合参考条件，不能冒称原vH实验绝对标定。完整捕获/参考/源码hash、角色和时序记录在运行manifest与输入NPZ中。每个新Run在自己的目录独占创建并校验 `input.npz`，`source_file` 和 `source_sha256` 指向该不可覆盖快照；相同 cone capture 可共享，但不同 rod 背景或同配置再次运行不改写首个Run的来源。旧共享缓存中的包不再由新Run覆盖；若历史来源此前已被覆盖，仅凭此次修复不能恢复其原字节，仍必须按原SHA核验。文件身份检查不等于生理验证。

时间合同保留刺激相对窗口边与绝对状态时钟、作者100µs格及其前史。期望counts按原窗长转窗均值，窗口内ZOH，末端消费最后一个有效窗；计数与原窗口数组不改变。仅在≤1e-12s舍入误差内容许十进制时长与tick乘法末端差，并将rebased末端/重合knots对齐到既有时钟；真实覆盖缺失仍拒绝。离线输入和RGB帧各有128MiB预算，超限明确要求缩短所选记录，不删细胞或降密度。

验证：66项相关唯一检查已覆盖输入/单位/窗口/背景和GUI历史上下文。真实ISET导出已完成；新背景完整77/77状态块在2.10s达到原判据，最后三窗最大残差0.74134、0.49693、0.33310。一次修复后的默认1.2ms实际Tk Run复用了身份匹配的新背景与真实ISET捕获缓存，重新执行retina solver（14次积分调用/250次RHS），34组件/20层输出有限。未复用旧RGB神经结果、未改阈值/生理增益、未跑全域或新机制研究。

证据：`reports/platform_runs/20261014T001258071356Z_a376b1e0/background_initialization.json` 与新完整checkpoint；最终短运行 `reports/platform_runs/20261014T004903003981Z_ba858f3c`；GUI验收 `reports/isetbio_active_frontend/gui_20261014_final/check.json` 与 `response.png`。匹配缓存后本次短运行墙钟11.68s。两个此前窗口舍入失败保留为失败历史，不计为通过。该交付是获批混合参考前端，不是完成全部猕猴生理标定。

## 结构、逐项目标与执行路径

组装接口是 `Population → ComponentSpec → ModelSpec → RetinaBuilder → HybridRunner`。每个细胞/接触有稳定 ID、类型、区域和位置；每个端口有数量、单位和语义；每个状态有初始化和时间合同。连接按真实 ID 定向，化学权重非负，正负作用由受体、膜电流和反转电位负责。电连接单独保存一份共享的物理电导，不能替换成两张独立归一化矩阵。

| full_target 目标 | 实际组件/代码路径 | 实际状态或作用 |
| --- | --- | --- |
| cone outer segment / native synapse / H1 | `outer_native`, `vh2005.py` | 源模型的 R/E/X/C/Vis/gi/Vbar/z1/z2/Vh 及 release；联合 HC 反馈 |
| S cone / H2 | 同一 `outer_native` 的 S/H2 群体 | 显式共享 H1 动力学的项目近似；没有声称是已测 S/H2 参数 |
| peripheral physical OS | `physical_OS`, `components.py` + 原 `chen2024.py` | 固定作者 0.1 ms R/P/G/C/S 更新，负的绝对 OS pA 电流 |
| physical membrane | SI 电流转换 → `physical_membrane`, `physical.py` | 两室被动电缆、固定条件电流参考及独立绝对电位参考 |
| physical terminal | `physical_terminal` ↔ `physical_HC` | 单调 Ca 激活释放代理、显式有效 HC 阈值反馈；没有虚构 gCa/ECa/释放生化测量 |
| midget ON/OFF BC | `HistoryBipolarTerminal`, `rod_pathway.py` | 原定有限因果 BC 历史核，加连续终端 V 和释放状态；中心私线保留明确源 cone |
| diffuse ON/OFF BC | 同一历史—物理终端机制、独立群体与连接 | 各自参数/汇聚；OFF 的默认 glycine 配置是 DB1-like 项目分支，不能代表全部 diffuse OFF 类型 |
| S_ON BC | 独立 `S_ON_BC` | S 输入和 ON 历史分支；不与 LM-OFF 分支合并成色彩减法向量 |
| local feedforward / crossover / widefield AC | 各自 `ResidualAC` | 完整有限核、因果历史、明确采样周期、不同路由与空间汇聚 |
| midget / parasol ON/OFF RGC | 各自 `HistoryConductanceRGC` | 物理 C/gL/反转电位/兴奋抑制电导、连续 V、严格过去 spike history、独立 RNG |
| small-bistratified S_ON RGC | 独立 RGC 与两条输入路径 | S_ON 和 LM_OFF 分别通过非负兴奋电导连接；保留独立抑制通道 |
| rod | `Chen2024PrimateRodCurrent` → SI current → `rod_membrane` | 真正 primateRod 作者化学状态及物理杆终端 V/释放，不是 transport provider |
| rod bipolar | `individual_rod_RBC_contacts` → `rod_BC` | 每个 rod 接触独立 mGluR6/TRPM1 门控，非线性先于汇聚；RBC V、释放池耗竭/恢复和事件率 |
| AII | AMPA → `AII_AC` ↔ ON BC / AII；glycine → selected OFF BC | 连续 AII V、资源受限释放、双向守恒 gap 电流、按 OFF 类型分别配置的抑制电导 |

旧插件 ABI 和其工程 fixture 保留供接口回归检查，但主入口不再使用 `RodMechanismPlugin`、`RodBipolarMechanismPlugin` 或 `AIIMechanismPlugin` 的 transport 演示。主入口的源选择、目标组件映射和执行元数据可逐项核查；仅注册名称不能满足覆盖检查。

## Rod 主通路、受体和电耦合

杆前端直接读取并验证已有固定作者源的 `primateRod` 分支：σ=φ=7.07 s⁻¹、η=2.53、Gdark=15.5、k=.01、n=3、Cadark=1、β=25 s⁻¹、m=4、KGC=.5、γ=4.2，Jdark=37.23875 pA。作者的有效活动/浓度归一化与真实测量分开；rod 的浓度状态明确使用 `Chen_scaled_concentration`。既有 cone 参考沿用作者命名单位，不把它当作独立测得的绝对浓度标定。

公开输入是 R*/s；作者差分步每次只转换一次 `rate*dt`。R/P/C/G 使用旧状态，S 使用新 C。非零恒定背景的化学初态由源方程固定点推导，之后保留连续状态。输入负值和非有限值拒绝，不以裁剪浓度或电压掩盖不稳定。

杆终端是物理电位和单调释放状态。每条 rod→RBC 接触各自计算并保存 mGluR6/TRPM1 有效开门比例，之后才以非负接触权重汇聚电导到 RBC 膜。RBC 和 AII 的资源池满足 `(1-x)/tau_rec - U*r(V)*x`；输出是 `Npool*U*r(V)*x` 的预期释放事件率。AMPA/glycine 是逐化学接触的因果双指数电导，单事件峰归一化；默认使用预期事件模式，没有同时再加随机事件造成重复释放。

AII↔ON cone BC 和 AII↔AII 每条电接触使用一个非负 S 电导，同时向两端加入 `g*(Vother-Vself)` 和相反电流。它们共同进入当前时间的膜 RHS；改变端点顺序不改变物理结果，电流总和守恒。BC 原来的历史生成量没有被冒充 mV：历史核保留，额外终端 V 使用具名的 V/reference/gain、C 和 gL 参数，再接收真正 gap/chemical 电流。所有连续膜状态一起积分，没有先更新一端再用新值更新另一端的顺序效应。

AII glycine 只进入选定 FMB-like 和 DB1-like OFF 终端。0.97/0.80 nS、−76/−73 mV 和电压端点 decay 数字保持，但当前本地资产只有核验摘要身份，运行记录降为明确的未完成 primary 验证来源引用近似，不能放行 `measurement`。连续 decay 插值、双指数 rise 和接触倍数仍是项目近似。没有把实验 event peak 当成总最大电导，默认不使用 rat 数值。历史来源 JSON 未改写。

可选 secondary rod↔cone 已有相同守恒电流原语的实际物理终端路径，以 OPL 距离、10 µm 候选范围和具名临时电导核生成。默认系数为 0，清楚称为 primary-only 配置；没有自动把饱和杆信号切换到虚构的 secondary 路线，也没有把 tracer 的约六杆池大小解释为六条直接 gap 接触。

## 空间、中心—周边与眼动输入

2026-10-14旧冻结人口图的 cone 采用 `piecewise_constant_poisson_tiles_v1`：固定全局分块上的分段常数 Poisson 采样；不能称已替换成等质量 master。HC 采用 joint canonical master；rod、RBC、AII 与 parasol ON/OFF 各用独立等质量 master，其他人口保持各自既有规则。已有 equal-rho-mass 后端并非所有活动人口的实际采样器，也不等于精确 Laguerre。S 标签使用 de Monasterio 源拟合/中央 assay 约束和显式项目桥接；L/M 比率、magnification、种间转移分别声明。此次只纠正说明，不重采样。

坐标接口区分 RF/吸收点、soma、OPL pedicle/dendrite 和 IPL terminal/arbor，缺失角色保持缺失。当前活动图五类 RGC 只有 `rf` 与 `ipl_post_arbor`，没有 soma 坐标；不能由角色可分离推称完整胞体位移已实现，也不复制 RF 伪造 soma。McGregor 等位移原语/独立历史参考保留已有证据与近似说明，但不等于本活动图已接入 RGC soma 映射；RGC soma 不能代替 cone 吸收坐标。新连接查询使用所声明的 OPL/IPL 接触坐标，未测完整形态的部分是明确的点形态近似。rod 终端属于 OPL，BC 终端属于 IPL；合同已经按真实细胞类型区分。

rod 的环状径向形状和角度插值是来源地标约束下的项目近似，不是 digitized Packer 全图。AII 有独立的内视网膜位置，并保留靠近中央的 cone-driven 通路，不能随局部杆/RBC 缺失一起删除。默认有效分辨率的接触半径/汇聚不等于实测 arbor；实际边长、接触数、有效扩张和单位都输出到 manifest。没有悄悄把真实密度降采样后称作动物细胞数量。

中心—周边连续 target 保留。2026-10-10历史source ROI运行的是周边二维 patch：owner `[4.59,4.61,-.01,.01] mm`，halo 40 µm，生成域 `[4.55,4.65,-.05,.05] mm`；尺寸属于计算选择。cone 数量为密度积分四舍五入，HC 在 OPL 独立积分。旧 x=.004..7.2、y=±.04 mm 缩密度走廊和显式有效计数仅供工程预览/历史复现。原 source_patch_v1 按映射后的 OPL halo 选择一个动物的 2–3 或 4–5 mm 区间，其他点锚与中央峰值不暗外推；另已授权实现 hc_joint_synthetic_v1 的 0–7.1 mm 合成场与联合mosaic/有限arbor接触。两者保持独立身份，新HC路径未运行，已跑patch不代表连续全域已经模拟。

新默认 cone 统一 `native_shared_H1_approximation`，S/H2 与区域转移继续明确未标定；不使用 20–25° blend。旧 blend 只在具名历史工程预览中保留。`physical_effective_network` 提供五组件独立 Chen >20° LM + 既有 effective cable/terminal/HC 配置，不以 native 填补其他谱型或中心；物理参考仍不等于完成 release 校准。给定 gaze 只改变输入采样，不预测眼动。

## 时钟、状态、测量与输出

所有连续状态使用联合 RHS。Chen rod/cone author-discrete 状态在精确 0.1 ms 事件更新，期间电流 ZOH 保持；BC/AC 的历史时钟和 spike 时钟独立。新的 `explicit_event_clock` 支持互不整除的周期（默认 widefield AC=.73 ms，BC/RGC=1 ms），不四舍五入周期，也不改写源核。原来的 commensurate 模式仍可用。原有显式逐边延迟、前史、method-of-steps、时间戳缓冲和 checkpoint 继续可用。

checkpoint 保存膜/化学/历史/释放池状态、事件索引、延迟历史、RNG 和完整输入背景上下文；跨窗口不重置。当前GUI使用有界参考场背景准备：由既有名义参考初态开始，以各cone实际ISET捕获对应的参考等效驱动积分；初始化本身不声称已经适应。原数值政策要求完整0.1s窗口、连续3窗通过、最大归一化状态漂移/原RHS/离散缺陷≤1，并检查前史；最低已知时间要求1.25s，墙钟预算1800s，未达到时不开始刺激。该证书只覆盖注册状态的数值判据，不是全生理稳态、随机分布稳态或独立动物验证。旧2ms固定预运行/6ms记录属于历史工程配置。

每个窗口输出所有层的有单位状态/端口，包括 rod photocurrent、接触 gate、RBC/AII 电位和释放池、预期事件率、AMPA/glycine 电导、各端 gap current、BC 原始历史生成量和终端电位、HC 反馈、RGC 条件强度/观察 spike 数。条件强度、实际窗口 firing rate 和 observed event count 分开。当前GUI默认采样0.1ms；历史独立命令保留自己的记录设置。可用 `--sample-step` 指定更细的亚毫秒输出，预算会按实际状态和样本检查。

参数证据由 `EvidenceValue`/profile 导出，含单位、来源、定位、适用域和不确定性。`measurement`、`published_fit`、`author_assumption`、`project_approximation` 与 `software_fixture` 保持区分，来源标签另列 `MEASURED_MACAQUE`、`AUTHOR_MODEL_PRIMATE`、`PROJECT_EFFECTIVE_TEMP`；默认没有隐含 `TRANSFER_RAT`。空间/接触系数也有独立几何证据条目。用于定参数的源值、曲线或拟合不能充当独立验证。

实际运行保存 `manifest.json`（组件、全部参数证据、原始来源记录、坐标、真实连接和近似）、逐窗口 NPZ、`final_checkpoint.json` 和 `summary.json`。NPZ 中记录的是模型在 synthetic 输入下真正执行得到的结果，不是动物实验数据。

## 先前架构检查和历史交付

必要检查实际结果：**41 passed in 104.44s (0:01:44)**，退出码 0。检查日志：`reports/primary_integration/20261010T233747112444Z/check.log`；当前检查状态：`reports/primary_pathway_status.json`。

性能修复前版本、相同空间范围及分辨率的短时全层运行实际保存于 `reports/full_target_runs/20261010T235800158134Z`：20 个必需机制层、39 个运行组件、21 个目标组件声明均绑定，无缺失目标；1280 个逐接触 rod→RBC gate，546 个逐边化学突触，191 个实际 gap 联系。显式适应 0.002 s 后连续记录 0.006000 s，保存 3 个窗口，每窗口 211 个状态/端口信号；全部 633 个数值数组有限，最终 checkpoint 哈希已核验。manifest、summary、NPZ、checkpoint 和 `delivery_verification.json` 均保留。运行代码哈希与 41 项通过检查使用的版本一致，六个既有源合同文件哈希一致。

默认窗口覆盖 x=0.004..7.2 mm、y=±0.04 mm 的连续视网膜走廊，128 个有效 cone、384 个有效 rod；这些是显式降分辨率的项目单位，不是动物真实细胞总数。输入为 synthetic，动物参数拟合次数为 0。此运行只证明当前有限 ROI 的全目标路径执行与记录成功，不证明解剖全视网膜完成或生理标定成功。NumPy 使用 OpenBLAS；成功运行仅在进程内设 OPENBLAS_NUM_THREADS/MKL_NUM_THREADS/OMP_NUM_THREADS=1，未修改系统环境或模型参数。

历史长时运行限制：原默认 0.2 s 适应 + 0.12 s 记录尝试在适应期耗时过长，已明确中止，未取得完整默认时长输出；不能用上述短时运行替代默认时长通过证据。短时入口命令为 `tools/run_full_target.py --warmup 0.002 --duration 0.006 --window 0.002 --sample-step 0.001`，全部模型参数、空间范围与细胞分辨率保持默认。较长时长与更高解剖分辨率仍需运行性能工作及实测资源评估。


必要检查覆盖源差分顺序/一次 rate→bin 转换、化学背景固定点、1.37 ms 非对齐窗口的 checkpoint/事件相位、真实每接触非线性、释放耗竭与恢复、双指数事件峰、双向 gap 电流守恒/端点交换、AMPA/glycine/gap 具名消融、cone OFF 输入在杆通路消融后仍存在、证据类别，以及 `full_target` 的实际路由。原 `contracts.py`、`engine.py`、`provenance.py`、`reference.py`、`chen2024.py`、`sources/registry.json` 的六个既定哈希均保持不变。旧 33 项结果是之前架构证据；当前检查结果以本节最新真实日志为准。

从 `D:\macaque_retina_simulator` 启动：

```powershell
& 'C:\Users\PC_User\anaconda3\envs\retipath_v1\python.exe' -B tools/run_full_target.py
```

同一入口提供 `--duration`、`--warmup`、`--window`、`--sample-step`、`--bounds`、显式密度/数量、seed、secondary strength 和 `--ablate ampa|gap|glycine`。原 `run_full_network_fixture.py` 仍是旧软件 fixture 的入口；当前主线入口是 `run_full_target.py`。必要检查入口是 `tools/run_primary_integration.py`。

当前科学边界是：独立 S/H2 和内层/终端的猕猴数值标定、各偏心度的 rod 接触阈值/释放/膜电导、真实 AII/BC 类型与 footprint/完整数字密度图、gap nS 和适应依赖、native 到物理终端及光学映射的校准、独立实验验证。AII 采用具名被动膜 reduced mechanism；未测 active channel 和杆 Ih/bleaching/regeneration 的影响属于其近似限度。精确 Laguerre 未实现且本轮明确允许保留守恒近似。默认有限分辨率不能证明真实解剖和长期生理稳定性；软件运行成功不能替代这些验证。

## 主要机制来源

- vH2005 [源补充材料](https://pure.rug.nl/ws/portalfiles/portal/2938733/2005JVisvHaterenSupp2.PDF) 与 [勘误](https://sites.google.com/site/jhvanhateren/home/erratum_vh05-html)：保留原生尺度与完整已核验系数，连续转写并未声称 Fortran ARMA 数值一致。
- Chen2024 [论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11537484/) 与 [固定参数源](https://github.com/chrischen2/photoreceptorLinearization/blob/1bcf7c9a6a6814ef39d2c39d82cf8dd1b5b39db2/src/model/initPhotoreceptorParams.m)：primateRod 与 peripheralPrimateCone 各自源分支；源 commit/blob 验证沿用。
- [Grimes2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6218188/)：猕猴 primary rod 通路及 AII/ON electrical、OFF glycine 路由依据，不是全部参数来源或 secondary 解剖缺失的证明。
- [McLaughlin2021](https://www.eneuro.org/content/8/1/ENEURO.0432-20.2020)：FMB/DB1 实验事件峰、反转与电压端点动力学；本地 `mclaughlin2021_verified_anchors.json` 是核验摘要，不是原动物数据文件。
- [Field & Rieke2002，mouse](https://doi.org/10.1016/S0896-6273(02)00700-6) 和 [Graydon2018，mouse](https://pmc.ncbi.nlm.nih.gov/articles/PMC6133723/)：第一突触非线性和资源机制背景；临时数字不是从这些论文伪装提取的 macaque 参数。
- [Packer1989](https://pubmed.ncbi.nlm.nih.gov/2794135/)、[Saha2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10616377/)、[Mills & Massey1999](https://pubmed.ncbi.nlm.nih.gov/10404105/)、[Strettoi2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6219554/)：空间地标和 AII 保留的来源；径向函数、个体位置与默认分辨率另标项目近似。
- [McGregor 原始源文件](https://osf.io/download/x35y9/)、[Wässle2000](https://bpb-us-e1.wpmucdn.com/sites.uw.edu/dist/6/8438/files/2020/09/Wassle-et-al.-2000.pdf)、[de Monasterio1985](https://pubmed.ncbi.nlm.nih.gov/2579042/)：已有坐标/密度/谱型约束与其适用域。
- [Maddox Ca activation 源表](https://cdn.elifesciences.org/articles/94908/elife-94908-fig5-data1-v1.xlsx)：1只 M.mulatta、3细胞、−90/−50 mV holding 的6对 Vhalf/slope。中央 punched 方法与工作表 Peripheral 标记冲突仍明确保留；没有由此编出 gCa、ECa、完整 release 或 HC gain。

其他已有固定源、许可、参数和原始资产身份继续见 `sources/registry.json`、`core_registry.json` 及 run manifest。独立验证状态必须与上述用于确定模型的来源分开。

补充来源适用范围：Chen primateRod 的既有参数来自作者周边 rod 模型；在中心—过渡区域复用该参数组属于尚未独立验证的项目应用近似。不能把参数的作者来源归类等同于所有偏心位置均已验证。

## 2026-10-11 默认入口性能修复与当前交付

原无参数命令是 `python -B tools/run_full_target.py`，模拟 0.2 s 适应＋0.12 s 记录，窗口 0.01 s、采样 0.001 s。第一次从 2026-10-10 23:37:52 UTC 开始，23:53:31 UTC 最后确认仍运行，故实际墙钟至少 15 分 39 秒；精确中止时刻未保存，不能补造。第二次仅设进程内单线程，23:55:42 UTC 开始，23:58:00 UTC 新短时运行之前中止，墙钟约 2 分 18 秒上界。两次目录均无窗口输出，停在完成 0.2 s 适应之前；旧入口未保存模拟时间进度，不能恢复精确中止的模拟 t。第一份目录 `20261010T233752361975Z`、第二份 `20261010T235542417397Z` 均保留，不能当作完成证据。

默认分辨率始终为 128 cone、384 rod；32 RBC、34 AII；H1/H2 共享 native 模块中的 HC 有效单位 32；midget BC ON/OFF 各118，diffuse BC ON/OFF 各30，S-ON BC10；3类 AC各32；midget RGC ON/OFF各118、parasol ON/OFF各32、small-bistratified10。physical cone 的18个实体在外段/胞体/终端表示中复用，rod 的384个在外段/膜表示中复用；1280个第一突触 gate、546个化学突触 edge 是接触状态，不能把27个 population 的数量直接相加冒充独立动物细胞。普通有向连接表51张、连接行14054，独立无向 gap 联系191个。

定位依据：原实现 2 ms 诊断有22次 solve_ivp、818次 RHS；通常100 μs事件间隔，29–41次 RHS，少数跨非整比时钟间隔为30/40 μs，刺激适应边界区间最多161次。cProfile 开销中 canonical_hash 累计13.695 s，占19.308 s约71%；其中对不变状态规范重复展开大批 cell IDs 并序列化两次/每RHS。此统计含 profiler 开销，不能当成无 profiler 墙钟，也不能直接推算整个0.32 s。现有证据主要支持实现开销；未证明所有长时条件下绝无刚性或拒绝步。性能诊断文件保留在新运行目录。

已修复：JointNetwork 初始化时缓存冻结的状态规范哈希、全局/组件状态切片、输入邻接表；每次 evaluate 复用同一状态的组件视图，避免按每个输出重复拆分；内部求解路径在每次导数保留单位、语义、shape、evidence 及有限值检查后直接展平，免去对不变注册表的重复哈希。对外 rhs 仍返回 StateDerivativeVector，原 contracts.py/engine.py 未修改，advance/checkpoint 仍检验注册身份。CLI 复用同一 runner，不再每窗口重建布局；未改变图的数学运算顺序、方程、细胞密度、求解器或容差。固定作者100 μs调度和非整比时钟均保持。

当前标准入口无参数可完成：默认2 ms数值连续适应＋6 ms记录、2 ms窗口、1 ms采样，仍是全部目标机制、同一128/384有效分辨率和连续空间走廊。用途是工程执行及逐层检查，2 ms不代表生理适应完成；6 ms不支持稳态、频率响应或动物符合度结论。仅对本进程给未设置的BLAS线程变量默认值1，尊重调用者已有设定；不修改系统环境。入口一开始输出building，随后输出initialized/warming_up/recording/completed；`progress.jsonl`与`run_status.json`记录墙钟、精确模拟 t、求解器调用量，长运行不再静默。

```powershell
Set-Location 'D:\macaque_retina_simulator'
& 'C:\Users\PC_User\anaconda3\envs\retipath_v1\python.exe' -B tools/run_full_target.py
# 原长时选项保留；本次未宣称该时长通过。
& 'C:\Users\PC_User\anaconda3\envs\retipath_v1\python.exe' -B tools/run_full_target.py --warmup .2 --duration .12 --window .01
```

当前默认命令实际完成于 `reports/full_target_runs/20261011T000323412125Z`：24.839 s（完整阶段日志24.851 s；含构建/初始化、求解、记录、manifest与checkpoint写出），初始化阶段14.615 s；实际ODE积分9.023 s，共89次solve_ivp、2989次RHS，最小事件区间10 μs（非整比0.73 ms/0.1 ms时钟交点导致）、最大161次RHS/区间。DOP853、rtol=1e-7、atol=1e-10、max_step=50 μs保持原值。默认执行不是长期性能承诺，但交付入口已不再默认尝试被中止的0.32 s运行。

仅重复了受改动影响的既有 `tests/test_primary_pathway.py`：**8 passed in 26.74s**；此前41项通过的版本及其日志完整保留，未把旧41项结果伪称为本次修改后重跑。最直接的回归证据：新默认与原2 ms＋6 ms运行的**633个数值数组逐项bitwise一致，最大绝对差0**，包括状态、端口、时间、spike_count与window_rate；所有数组有限，六个冻结源合同文件SHA256保持不变。`performance_delivery_verification.json`保存该结果。无研究对照、参数拟合或动物验证扩展。

必需层的实际绑定如下；无空机制名、未绑定输入或未接必需目标。它们使用的临时参数/共享S-H2动力学仍属于已声明项目近似；实现绑定完整不能替代生理标定。secondary rod-cone gap 默认strength=0是明确可选路径关闭，具有真实电流代码；也不属于必需目标漏接。

| full_target 必需层 | 实际运行组件 |
|---|---|
| `cone_outer_segment` | `outer_native`, `physical_OS` |
| `cone_native_synapse` | `outer_native` |
| `HC_H1` | `outer_native` |
| `HC_H2` | `outer_native` |
| `midget_BC_ON` | `midget_BC_ON` |
| `midget_BC_OFF` | `midget_BC_OFF` |
| `diffuse_BC_ON` | `diffuse_BC_ON` |
| `diffuse_BC_OFF` | `diffuse_BC_OFF` |
| `S_ON_BC` | `S_ON_BC` |
| `local_feedforward_AC` | `local_feedforward_AC` |
| `local_crossover_AC` | `local_crossover_AC` |
| `widefield_AC` | `widefield_AC` |
| `midget_RGC_ON` | `midget_RGC_ON` |
| `midget_RGC_OFF` | `midget_RGC_OFF` |
| `parasol_RGC_ON` | `parasol_RGC_ON` |
| `parasol_RGC_OFF` | `parasol_RGC_OFF` |
| `small_bistratified_S_ON` | `small_bistratified_S_ON` |
| `rod` | `rod` |
| `rod_BC` | `rod_BC` |
| `AII_AC` | `AII_AC` |


<!-- local-platform-detail -->
## 本地可视化平台与默认配置修正（2026-10-11）

入口仍为 `run_retina_gui.cmd`，复用 `C:\Users\PC_User\anaconda3\envs\retipath_v1\python.exe`；未升级环境。GUI 默认“源约束混合”，提供背景、目标位置预设、自定 owner ROI 中心/宽高、halo 与原有运行/暂停/继续/停止、结果视图和导出。停跑后新增的位置选择与区域路径均为 `NOT_RUN`；选择预设只填写几何值，不自动改变动力学。`configurations/source_roi_display_v1.json` 可直接载入。工程 preview 数量控件只用于 preview；不改变 source-density 数量。CLI `tools/run_full_target.py` 默认同一个 `macaque_constrained_native_v1`，可显式选择 `--profile engineering_preview_v1`。直接调用旧 `full_target_network` 的默认参数继续提供历史工程兼容；新调用传 `execution_profile='source_constrained_approximation'`。

配置实际数值从 `effective_dynamics_v1.json`/`legacy_effective_v1.json` 解析，37 个参数族、254 项常量或推导绑定。C/gL、BC/AC 核、五类 RGC 共同数值等未为了看起来生理而换成猜测。均匀广播值和固定向量有单位/语义检查，背景电流 anchor、baseline、接触增益等保持明确推导；最终全部实际向量见每次运行的 `resolved_profile.json`。execution profile 同时声明路线、solver、预算、区域、source/cohort policy 与精确 allowed approximation IDs。内层共享参数代表尚未辨识型间动力学。

冷启动 GUI 直接构造 `RunConfig()`，不需要先载入新 JSON 才进入源约束模式；首次仅需导入刺激媒体。显式载入或保存的用户配置优先，不会被默认配置覆盖。旧 JSON 缺少执行模式时按历史 preview 解释。CLI 不需要媒体或手动配置文件即可使用同一默认 ROI。当前没有自动读取上次配置的机制。

`LightContext` 为不可变、带输入域标签的背景上下文。native td、cone R*/s、rod R*/s 是独立消费端口设定，不推断它们属于同一实物光源。背景进入建网初态、baseline 和驱动；`background_context_hash` 同时存于输入合同、初始化与 checkpoint。vH 使用完整已有背景平衡，Chen cone/rod 使用实际声明 rate 的源固定点；rod 膜参考额外按实际背景电流配平既有 leak/V0。物理 cable 保持 `I(t)-I(bg)` 的参考，而 Vanchor 仍为既有项目值。新 run/reset 后才应用编辑，运行中的状态不会被参数控件暗重置。

RGB 仍是假定 sRGB 的 display approximation。新 `contrast_reference_v1` 实际编码是非负 relative intensity u：`L=Lref*u/uref`，默认 linear `uref=.5`；黑像素是真零输入，参考值映射到声明背景。它不是 Weber c 编码（c 的定义应为 `L=Lref*(1+c)`），也不是颜色/光学校准。L/M/S 与 rod 人工权重保留并逐运行导出。旧 `legacy_rgb_proxy_v1` 的 `.6+.8u` 仅用于历史 preview。`.223 mm/deg`、70° FOV、图外 reference 背景均明确是项目显示约定，不能作通用猕猴光学校准。

内层模式明确为 `transient_start`。全网络 AII/cone/gap 联立平衡求解器尚未实现，不能把有限预运行诊断解释成该求解器尝试后不收敛。可解析的局部闭合已落实：TRPM1 初值取实际 rod 初始释放；RBC 电压按实际汇聚电导配平；release resource 与化学滤波初值按实际源 V0/释放参数一致计算；AC 背景历史取实际 BC 初始输出与稀疏连接；RGC 按实际 BC/AC 初始电导配平；独立物理参考 HC 使用当前 cable 参考电压下的标量固定点。未求解的耦合状态与随机 spike 零前史仍显式保留，不以随机预烧代替确定性平衡。2 ms 只表示均匀背景有限预运行，不表示充分适应。

当前小 ROI 的 cone 为 `round integral rho dA`，没有 requested_count/mass 缩放；Packer 60 行只提供 N/T tabulation，log-PCHIP、角向延拓、central cap 与跨来源 S 组合仍是项目 provider。等质量 greedy-capacitated 算法输出整数/积分、容量和 quadrature residual，未标为精确 Laguerre。HC 的源约束 provider 现在要求 RF halo 经具名 McGregor RF→OPL bounding-box adapter 后完整落在一个动物的有限区间：2–3 mm 取 animal 9.8.99 的 H1=6360/H2=1586，4–5 mm 取 animal 29.7.99 的 H1=2665/H2=947 cells/mm²；随后在该 OPL 物理域独立积分。单区间 stationarity 和跨来源 RF→OPL 矩形适配均为项目近似，没有拼成一只全域动物。3.5/5.6/6.3/7.1 mm 单点行不形成有限密度场；中央 0.5–1 mm 的 18400/4600 是作者估计峰值，只保留参考，不能铺满中央或外推到 pit。H1 仅 LM、H2 LMS，divergence 通过来源锚点插值和平衡整数配额，H2 使用所报范围中点的项目选择；owner 缺配额会拒绝。HC feedback 只返回已有 feedforward 联系。

rod 径向/角向场保持已有 source-landmark project 公式，密度 scale=1；RBC 用 `2*rho_rod/36` 独立二维 mosaic，AII 用 5000/mm² anchor 的独立二维 mosaic。前者是 central rounded-degree 公式转移到周边，后者是 1.5 mm anchor 的常密度区域转移，均为 `source_anchored_approx_density`，不是新增实测全场亚型密度。RBC 不再每 12 rod 取点，AII 不摆 y=0，也不依赖 rod 数量决定是否存在。BC/AC/RGC 未知密度保留 `effective_unit`；膜表示层和化学接触分别计数，编译预算的实体总数不能当动物细胞总数。

默认 owner `[4.59,4.61,-.01,.01] mm` 面积为 0.0004 mm²；加入 40 μm halo 后生成域 `[4.55,4.65,-.05,.05] mm` 面积为 0.01 mm²。以下为停跑前 03:51 完成的配置所导出的 `density_integrals`，属于该次历史运行证据；生成整数数目为该生成域密度积分四舍五入，owner 数目为实际落入其 owner 的细胞中心数。停跑后 HC 域与 owner 改用显式 RF→OPL adapter，新代码未运行，不能把下表当新路径验收。

| 群体 | 生成域积分质量 | 生成数 | 实际 owner 数 | halo 数 |
|---|---:|---:|---:|---:|
| cone | 96.2336861524 | 96 | 5 | 91 |
| rod | 1189.9731839882 | 1190 | 51 | 1139 |
| H1 | 26.6500000000 | 27 | 0 | 27 |
| H2 | 9.4700000000 | 9 | 0 | 9 |

cone 使用未缩放的 Packer 派生二维 provider；rod 使用既有 source-landmark project field，不能解释为已核实的二维原始测量。H1/H2 独立按 Wässle 所选行生成；owner 内没有 HC 中心不表示 owner cone 无 HC 接触，接触可以来自 halo。每次新运行的 manifest 与 resolved profile 都导出面积、质量、整数残差、owner/halo 数。

原source patch及内层OPL/IPL局部选择先实行真正40 µm hard support；新联合HC模式按显式有限arbor支持筛选，再按具名pooling归一化，保存目标 degree、实际边数、缺额、最大边长；不会跨任意远距离补 nearest。midget BC 保留 parent cone ID，central private 分支使用唯一 parent BC，外围使用 Wool fit + 明确 iso-density temporal-equivalent adapter、fractional boundary 权重。有效输入数、整数边数与归一化增益语义分开。gap 候选同样受 hard support，保留共享非负电导和 reciprocal 电流。

halo 是有限计算 buffer。当前边界是 finite induced graph，外部耦合省略，没有 reflection/重复细胞。HC/gap 循环有多跳和无限 recurrent 依赖；40 µm halo 不宣称消除边界误差。扩大 halo 的 owner 响应收敛、Gaussian 未生成域丢失质量和完整严格接线验证仍未完成，当前 profile 只允许明确 finite-boundary 近似。超过 cell/compile/input/recording 预算会报错，不暗缩密度、改 dt 或切换 preview。

midget RGC 的 `parent_bc_ids` 为实际限定 ID，`effective_contact_target<=1` 时按 ID 建立一对一 BC→RGC 矩阵，不依赖索引排序或 nearest；越出声明接触范围拒绝，并更新实际连接审计。中央规则已做乱序父 ID 与越界回归，但当前 4.6 mm ROI 是外围汇聚路径。原HC patch provider保留独立的2–3与4–5 mm区间；新合成HC有限密度场及中央q→pedicle→arbor接触接口已实施，但中央形态跨度未设默认、全层运行未执行，因此不声称中央源约束全层ROI已通过验收。该历史路径私线接触点取限定 parent BC 的 IPL terminal，坐标原语可单独保存 soma、不把 soma 当突触点；当前活动图五类 RGC 的 soma 仍缺失，不能把这一接口能力表述成完整胞体位移已接入；Wool 分支按 iso-density temporal-equivalent radius 判断，不能用 UI 中心标签替代。新默认 native 路线没有 20–25° native/Chen blend；旧 blend 仅留于具名工程 preview。

Maddox `source_operators.maddox_ca_activation` 需精确 `cellN_hold-90/-50` record ID，六对 Vhalf/k 实际与冻结原 workbook K/L 单元格配对核对，接既有 normalized Ca activation 算子。类型是 `published_fit`；未知 region/谱型、gCa/ECa、动力学、HC shift 和 release 不由此补齐。`examples.physical_effective_network.physical_effective_network` 是独立可运行五组件参考。合同补充的 Bryman/Wienbar 特定 foveal cable 原始资料在本项目未提供可核来源资产，本轮未造一个 source-verified peripheral cable；保留原 C/gL/axial/Vanchor effective closure。

static/dynamic measurement 共用来源身份检查，要求 primary verification 等级及具体材料 hash/locator/condition。dynamic published_fit 要求 verified code/table/primary，structured cell type/subtype/eccentricity 域在构建时执行，编译会复查被替换的记录。历史 registry JSON 和结果不改造；v2 静态 schema 保留兼容。Packer expected 来自修改前的独立既有 extraction manifest，不再从当前 CSV 自算；该 manifest 只核实提取一致性，原始测量归属、动物与误差仍未核实。新配置内 Wässle/S/McGregor 派生 JSON 的预期哈希仅是本地内容基线，明确标记 `verification_scope=local_baseline_integrity_only`、`primary_verification=pending`；冻结未经核实的内容不升级其科学证据等级。

| 资料对象与当前定位 | 父合同 SHA256 | 当前原始字节 SHA256 | 身份与核实范围 |
|---|---|---|---|
| Packer CSV `data/published_code/packer_cone_density.csv` | `f87e3674dea002652634e7c9d952d011b5ec427f0275496b02a9b413bcdf6d33` | `897ab2056e5a92cd0180b09ee4b97fe01cf010cc17f269fb15d94e352515ba5c` | 不同；当前值匹配修改前 manifest，不静默接受合同另一版本 |
| Packer MATLAB `data/source_snapshots/isetmacaque_cf22ac9d/PackerConeRc.m` | `4140b1c89eb3c8a7b5c23f5b4a13b7d4ab1404a8593aafd51b2b82d7d3fb2142` | `72408017cb5b6c4996b48c95050963057f61aab6e39154ea095062a8b5c35e9d` | 不同；原 manifest 定位 `toolbox/+simulator/+load/PackerConeRc.m`、commit `cf22ac9d681482ca1fcc365a423ef0c2be7916a3`；只检查过本地 HEAD，未证明整个原 checkout 干净 |
| Wässle 派生 JSON `sources/spatial/wassle2000_HC_density_tables.json` | `e22c0210ae2ccafff6a7566bd5bcdc2e1d0d6916ac92685b674ea1f914e6bf9b` | `852c71ebb9cbc63f126fe30f0ca360fe0ccfb794a95e5e9116e71d7e2d15a2c2` | 不同；原 locator 是 Wässle2000 PDF Table 2，当前 JSON 转录主来源核实待办 |
| McGregor 原始 MAT `data/source/mcgregor2018/fovealTopographyData.mat` | `e6122ffdaf46100d51f1b81f11af7e5d863c354e0a0d4e8ce5e2e27bcf451e78` | `e6122ffdaf46100d51f1b81f11af7e5d863c354e0a0d4e8ce5e2e27bcf451e78` | 一致；来源 OSF 下载。派生 production nodes JSON 是另一对象，不与原 MAT 的 hash 直接比较；跨来源二维适配仍为近似 |
| Maddox 原始 XLSX `data/source/maddox2024/elife-94908-fig5-data1-v1.xlsx` | `df7b88ce170bfc1dce2867937cacd7bd0315f419a2b86d5a5b36fbc4d76a959c` | `df7b88ce170bfc1dce2867937cacd7bd0315f419a2b86d5a5b36fbc4d76a959c` | 一致；旧 source registry 仅 identity_verified。新算子直接读取 Sheet1 K4:L6、K11:L13 核对六组成对 fit，为该有限数值表建立独立 numeric_table_verified 记录，不推断全物理 release |

Packer 原 manifest 的原始路径为 `D:\retipath\experiments\retipath_foveal_v1\exploratory_v2\vendor_pinned\isetmacaque_cf22ac9d`。本次只读逐值比较确认：该路径的原 MATLAB 与项目快照原始字节完全一致；从四个源矩阵重新提取的 60 行（temporal foveal/peripheral 13/15，nasal 13/19）与当前 CSV 的 10 列全部一致，共 600 个字段、其中 120 个核心数值，无数值或元数据字段差异；提取 CSV 原始字节和历史 manifest 完整对象也一致。这是代码数表提取一致性，Packer 原始动物测量表尚未独立核实。

Wässle 当前完整 JSON 与修改前历史 `C:\Users\PC_User\Documents\Codex\2026-10-10\task-2\prepare_spatial_sources.py` 中的转录对象一致：Table 2 的 6 行共 30 个标量字段，其中 24 个数值；中央 author estimate 另 4 个数值，均无差异。默认所取 animal 29.7.99 的 `[4,5] mm`、H1=2665/H2=947 也匹配合同的明确数值约束。历史脚本仍是手工来源引述，项目没有保存该原始 PDF，故其六行数表与中央 author estimate 未独立对原页核实，不升级为 primary_verified。

Packer MATLAB/CSV、Wässle JSON 的 LF 与 CRLF 形式都不匹配合同另一组 hash，已排除纯换行解释；Wässle 另检查 84 种常见无语义变化 JSON 序列化，均不匹配。合同 hash 所对应的另一版原始文件字节未提供，所以不能由 hash 单独判断是其他序列化/元数据差异还是数据值变化。实际结论是本地历史内容的逐值差异为零，合同另一版差异尚缺对应文件可比；没有改写任何来源文件凑 hash。身份一致、提取一致和生理有效性是不同范围。

本轮受影响检查覆盖背景 build/init/drive 联动、均匀背景、真实 profile 数值绑定/单位、CSV 一字节篡改、伪造 measurement、域/类型、hard support 缺额、二维数量/HC/native 路由、六对 Maddox 原表配对、历史配置兼容，并复用已通过的源差分、gap、event peak、history/checkpoint 等旧检查。新增 9 项与一个旧 measurement gate 的最终命令为 `python -B -m pytest tests/test_runtime_profiles.py tests/test_stage_a.py::test_code_not_primary_measurement -q`（10 passed）；此前受影响旧项有 39 passed，修改旧 measurement 预期并修临时目录后，两项旧失败/环境错误均通过。没有重做全部研究源审计、训练或研究对照。

最终端到端目录 `reports/defaults_acceptance/20261011T032037326390Z`：20 层、39 组件，96 cone、1190 rod、27 H1、9 H2、66 RBC、50 AII；2 ms uniform prerun + 6 ms recording，3 个窗口，250 td/2000 cone R*/s/15 rod R*/s，约 9.6 MiB compile estimate。实际 PNG `source_roi_response.png`、`selected_cone.csv`、acceptance.json、run manifest/resolved profile/NPZ/checkpoints 均保留；GUI 字段 roundtrip 使用同一个结果，不重复模拟。独立五组件 physical reference 另仅执行 0.1 ms 接口烟测。第一次出图因环境无 matplotlib 失败，原结果保留，改用现有 Pillow；后续完整合同的几何/输入变更完成后进行上述最终端到端，没有安装绘图库。

最后的局部初始化与输入回归 `tests/test_runtime_profiles.py` 为 **13 passed in 8.22s**，包括实际网络初始输出汇聚后的局部 RHS、black=0、非默认 linear reference 回到声明背景、图外填充值回到背景、显式父 ID 乱序映射。随后针对新增私线 hard support 与源 ROI 的三项回归为 **3 passed in 8.34s**。这些局部检查未进行时间积分或随机预烧。

停跑前最后一次使用当时的代码执行短入口 `tools/check_physiology_defaults.py`，未重跑全套回归或科研对照。其目录为 `reports/defaults_acceptance/20261011T035144035652Z`，run 为 `runs/20261011T035144381425Z_2b93eb51`：20 层、39 组件、2 ms uniform prerun + 6 ms recording、3 窗口 completed；ResultStore 可载入，228552 个已记录标量全部有限，resolved profile SHA256 为 `7a932f26191d1b2c57ffc8257b188d54f4631917cf86693f85e74239a5f7420c`。`transient_start` 与 `INIT_NOT_CONVERGED_TRANSIENT_ALLOWED` 保留；背景最后窗口的最大归一化 drift 为 9502.752977，不表示全网络稳态。`acceptance.json` 记录运行代码版本哈希与图版本 2；`reports/defaults_acceptance/latest.json` 保留最后完成的 run/图指针，并额外声明 `current_code_status=STATIC_REGIONAL_FOLLOWUP_NOT_RUN`，历史目录不改写。停跑后的区域/GUI/观测目录改动没有模拟、测试、replay 或拟合证据；只做源码 AST 与 JSON 读取检查。新图红框及红圈分别表示 owner 区域与其中细胞，其他点为辅助 halo；曲线标注实际所选 cone 0，不能默认将其当成 owner 细胞。

当前 GUI 绘图按端口/类型显示全部已记录细胞，包含 owner 与 halo，没有 owner-only 筛选按钮；默认 RF 范围覆盖生成域 `[4.55,4.65,-.05,.05] mm`，实际细胞中心绘图范围略小。owner 是目标报告区域，halo 是参与计算的辅助细胞，均有记录，`population_owner_mask` 与人口/密度审计可区分它们；OPL/IPL/Soma 图使用各自接触/解剖坐标，不把 RF owner 框套成全部坐标系的范围。

关于中心与周边：4.6 mm 在项目固定 `.223 mm/deg` 显示换算下约 20.63°，正 x 使用 Packer nasal 分支，换算不声称是通用动物光学常数。cone density provider 有中央常值 cap、N/T 节点内连续插值与角向延拓；S provider 有中央 assay bridge/外围 fit；rod provider 有 rod-free 中央及径向/角向 source-landmark 项目场；McGregor 有 RF→解剖位置适配；midget 有父 ID 私线与外围 Wool 汇聚规则。`center<.5 mm`、`transition<3 mm`、其余 periphery 的名称分区是项目约定。S/H2 仍共享 H1 动力学，BC/AC 核与 RGC 型别动力学仍多为共同 effective 参数，不能将不同几何解释成完整型别生理辨识。

GUI 现在可选中心目标（0.15 mm）、中央作者锚点（0.75 mm）、过渡来源锚点（2.5 mm）、周边来源锚点（4.6 mm），或自定二维位置/ROI。前两者仍是未跑的几何目标：source_patch_v1不会补中央场；可另选已实施未跑的hc_joint_synthetic_v1连续合成场，须显式提供有限arbor/形态。0.75 mm不会用中央峰值硬铺常数场。后两者尝试按映射后的 OPL halo 匹配单动物区间，不保证仅凭 RF 中心值就通过全部接线合同。手改预设坐标/尺寸后保存为 custom。中央私线曾做旧版本局部规则回归，新接触点处理和区域 GUI 未运行；旧工程 corridor `.004..7.2 mm × ±.04 mm` 跑过的中央/过渡/周边短动力学只属于工程狭条预览。源约束连续二维全域还受 cohort/domain 与未知亚型密度限制，预算超限不暗降密度。

验收解释：`engineering_pass` 已有短运行证据；`source_adapter_verified` 仅指对应已检查的代码提取、Maddox 配对和既有源数值接口，不包含未独立核实的 Wässle 原页或 Packer 原始动物测量；`constraint_pass` 仅涵盖已执行的数量、域、owner HC 配额与 hard support。`physiology_validation=NOT_RUN`。全确定性内层 equilibrium 求解器尚未实现，扩大 halo 的循环边界收敛是尚未进行的数值实验；这两项不能归为仅缺动物数据。长时间适应、严格全型别接线、全域亚型密度/独立 S-H2/BC-AC-RGC/rod 内层及 absolute 光学/release 标定仍是明确缺口。DeepGaze/SC 继续暂停。


### 停跑后的区域合同与补源绑定（静态修复，未运行）

所有模拟进程已停止。本段描述保存到代码的修改，不描述新的生理验收。`regional_support_v1.json` 保存 UI 导航约定、预设与逐来源适用域；`regional_observations_v1.json` 保存 85 个 typed 观测参考、3 个生成模型候选及配对资产信息。源码中的 `RegionSpec` 按来源自己的 unit/frame/origin/meridian/条件选目标，不能把 `.223 mm/deg`、光学 RF 或 McGregor soma proxy 静默变成另一篇论文的记录位置。未定义精确数字区间的约 30° 样本只可读参考，不能自动选成全周边适用域。未本地主来源核实的条目须显式 reference-only 访问；四类 evidence_class 沿用原合同，参考目录不进入 runtime coefficient registry。

| 路径／来源 | 类型、来源位置与条件 | 当前实际绑定及边界 |
|---|---|---|
| Packer + deMonasterio + McGregor | RF 密度与 S fit、单动物 foveal RF/soma 数据各自的域 | 未缩 cone 密度；跨来源角向、中央 cap、S bridge、pedicle 共享与 RF→OPL border bounding rectangle 均具名近似。HC 不再伪造光学 RF；其 OPL/解剖位置和 owner 区域单独导出，实际 cone pedicle 必须落入 mapped HC 域。 |
| Wässle2000 HC / 联合合成HC | 物理soma/OPL框架；原patch为单动物2–3、4–5 mm，新合成域0–7.1 mm | 原patch独立积分保留。新hc_joint_synthetic_v1分开保存中央/Table2记录和shape/轴对称/跨动物假设，联合配额soft-core并接有限pedicle/arbor。冻结历史转录identity；原页核实pending，新增实现NOT_RUN，形态跨度未配置。 |
| Wool2018 midget | temporal-equivalent `.25..10 mm`；观察 3–9 mm；D 为 radius | 私线按实际 cone→BC→RGC parent IDs，BC 私线应用受原 source 中心范围约束；外围多个 parent BC 汇聚。边数、归一化前后和、`N_eff` 分开审计，没有在显式 HC 后再叠来源 light→RGC surround。新 IPL 点修复 `NOT_RUN`。 |
| Chen 周边 LM | LM `>20 deg`，不含 S/foveal | 无支持细胞时省略独立 physical reference 与输入，不把参数搬到中央。主 native 与 legacy blend 为具名路线；独立五组件参考保留。 |
| rod/RBC/AII | rod source-landmark field；RBC/AII 独立源锚转移 | 二维几何已接；内层数值与区域转移仍 effective。没有全域实测亚型场或 rod 内层校准。 |
| Sinha2017 | ON-midget 记录点 `<1 / 2..4 / >6 mm`；约 4000 R*/cone/s、31..34 C | spike STA、composite excitation filter、QI/QE 各自端口目标；1..2、4..6 不补空。charge ratio 不写成 gI/gE/AC weight；不转移到 OFF/parasol。S/L/M 电流与电压峰时独立保存。 |
| Saha2024 / Baudin2019 | source-defined foveal/peripheral cone 位置、背景和 assay | 背景分组 flash、gain adaptation、S/LM endpoint 参考；已记录转录差异及缺少 local workbook/raw array。适应拟合 τ 不自动成为模型门控 τ，S 不套 LM 后声称校准。 |
| Zhang2011 / Packer–Dacey2005 | HC 空间场、光背景及物种域；D10%来源方程 | 空间 diameter、lambda、light/dark ratio 与方程参考不转成 gap g 或 temporal kernel；混合 Macaca/Papio、未解决的中心实测下界与可疑电阻单位不补猜。 |
| Puthussery2013/2014 / McLaughlin2021 | 分型 BC；后者成年 mulatta `>8 mm nasal`、31..33 C、300 µm slice、low photopic | OFF residual 不称显式 AMPA/KAR 门控。McLaughlin FMB `.97 nS`、DB1 `.80 nS` 已有数值保留；源条件写入对应 EvidenceValue 和导出。−100/0 mV 的 decay endpoint→连续 Vm、周边 slice→当前 ROI 为独立允许近似。10–90% rise 区间不当 tau_rise。 |
| Khuc-Trong2008 | 周边约30° ON-parasol、mixed macaque/Papio、32..34 C | RC约2 ms、绝对不应期2.5 ms、相对阈值4 mV/8 ms、相对阈值4.5 mV为具名作者候选；当前电导 ODE + softplus/history 形式不兼容，全部未绑定。不用 Rin 反推 leak、不造测量C、不将 relative threshold 作绝对 Vm。 |
| Weber2005 Normal | pooled ON/OFF parasol，midtemporal 4..8 mm、36.5 C、sharp electrode，n=61 | Vm、Rin、tau、rheobase、AP宽度/幅度和最大记录率作为 Iinj→Vm/spike 目标。当前主线无注入/数字 AP waveform 端口；不把最大率倒数作 refractory，未做 ON/OFF 分裂或 C 推算。 |
| Latimer2019 / CBEM | 6个 ON-parasol paired stimulus/EPSC/IPSC/spikes | 委托核实 Cell_6.mat 大小/MD5/schema已登记；本地无副本，dt、单位、padding、per-cell species未解，replay未准备好。CBEM忽略C、g_l=200为不可辨识 model scale；light→g filters不塞进 BC residual，simulated示例不称实验数据。 |
| Cafaro2013 / Crook2011/2014 | typed midget/parasol；各自位置、背景、温度、clamp internal | gE/gI相关 lag、source I–V和干预只作 typed postsynaptic 目标。−62mV driving force、dynamic clamp −90mV E_i、Crook pipette −65mV分别保存；不合成通用E_inh。NMDA voltage fraction有候选，但主线没有 branch，未扩展。Cs/QX-314 Rin/rest不当天然 leak/rest；整细胞nS峰值不当单条边g。 |
| Crook2009 | central `<4 mm/<20deg` small-bistratified，35..36 C | S40±3/LM46±3ms输出相位 latency与低光rod177±36ms分别保留；不是BC/cone/spike intrinsic tau。不取消AC或反转独立cone结论。缺 intrinsic 参数与原始数组。 |
| Wu2023 / Percival2022 AC候选 | 具名 ON-starburst 多区室／AII puff whole-cell响应 | 不能替泛AC，4.4nS/+0.9mV不能作单RBC→AII g。只登记候选，不实施新多区室/active channel/gap 架构。 |

上述内层补源已完成本轮静态登记，未继续扩展文献范围。五个 RGC 类型×中心/过渡/周边的 intrinsic spike 参数矩阵仍未齐备；已识别非-starburst AC 的本体门控、release 与耦合也未齐备。当前五类 RGC、BC/AC residual、S/H2共享动力学仍明确 `NOT_CALIBRATED`，不能因参考目标充足而宣布系数已辨识。每次未来构建的 regional audit 会记录实际 component/typed population/port、参数profile、背景hash、允许近似、candidate未绑定状态和数据asset未准备状态；没有自动 STA→tau/kernel 转换。

<!-- /local-platform-detail -->

## HC 连续合成几何与接触路径（已实施，未运行）

用户随后明确授权写代码；[HC空间分布方案与实现范围](hc_distribution_proposal.md)保留Pro讨论、其他模拟器比较和科学缺口。新增具名`hc_joint_synthetic_v1`已接入既有项目，不自动替换`source_patch_v1`，不改变HC/RGC动力学。工程preview继续独立选择，不能把其缩密度走廊解释为新组织实例。

实际调用链为RunConfig/GUI或CLI→`read_hc_geometry`→`source_roi`→`HCDensityProfile`/`generate_joint_hc`→同ID crop→`build_hc_arbor_contacts`→`hc_contact_runtime_view`→既有原生组件。总HC密度与H1比例分开；中央250→23000 cells/mm²的log-smoothstep、pH1=.8、外围log-total/logit-fraction插值、跨动物/meridian与轴对称均明确来源或project假设，不改称同眼测量。0–7.1mm之外拒绝，不clamp；按积分配额联合生成H1/H2，0/1计数可用，不插最低细胞数、不缩密度。有限温度soft-core不是作者dmin算法复现，没有混合/生物统计验证声明。

HC以物理soma密度独立生成，不从cone抽样或套cone位移。接触沿cone输入q、移位OPL pedicle p及同一个coneID查询有限HC arbor；支持显式disk/ellipse或morphology节点、有限延伸、z/区域/间距规则。H1排S为具名简化，H2可接L/M/S，不由此赋S-specific gain。divergence可配置具名source/profile及全cone分母的平衡整数目标；未配置时接触图为uncalibrated候选。反馈沿同一拓扑返回，权重与汇聚单位分别保存，无跨支持nearest填充或隐式Jacobian乘子。

arbor字段默认null；GUI新增HC空间模式、H1/H2显式mm半径与HC几何JSON入口，短状态注明未跑。CLI使用同一profile/config。GUI半径仅为project有效footprint；measurement/published_fit几何须有来源径向支持、primary材料冻结hash/locator/condition及确切geometry hash，加载器检查身份，不能只填sourceID通过。这个检查不是生理验证，也不升级历史source record。HC homotypic Delaunay/collinear候选仅在显式距离配置时导出，电导未绑定、未接动力学。

生成域保留owner、mapped OPL输入、contact active view、canonical与来源支持边界；canonical上下文至少覆盖arbor包络和全部pair相互作用支撑。固定canonical域的crop保持同一实例ID与坐标；改变自动解析canonical域会产生另一实例。预算超限或halo越界明确报告。有限输入域之外cone接触的缺失逐HC审计，循环边界误差仍未验证。

原生模型要求有输入的单位汇聚行。无cone接触HC不进入动态参与view，但仍完整保存在canonical/active组织实例、generated_active_population与接触审计，密度实例不稀释，不声明这些细胞在真实组织中无输入。当前HC gap仅候选，没有通过孤立HC传播的动态边。若cone无有限反馈接触，报HC_CONTACT_RUNTIME_UNCOVERED_CONES，不补nearest或固定背景反馈。每次未来构建导出`hc_geometry.json`，manifest/resolved profile记录完整canonical与局部ID、生成/动态/geometry-only计数、来源与假设hash、配额缺额、接触拓扑及halo状态。

用户随后明确授权“测试一下”：30项HC定向回归和3项必要source-patch回归通过；发现并修复regional_binding_audit将legacy Profile当MechanismProfile读取source_ids的接入错误，现在采用实际RuntimeComponent.source_ids。真实Tcl变量及RetinaApp._read_config→SimulationSession→source_roi→full_target_network→导出链完成一次0.2ms运行：39组件、20层、1窗口、76664个有限数值。小canonical域[4.3494977985,4.8505022015,-.2505022015,.2505022015]mm内H1/H2为659/236，对应积分质量658.5782397447/235.8374477248；局部几何173 HC，动态89，geometry-only84均保留实例身份，实际2011有限接触对。GUI显式H1 disk .04mm、H2 JSON ellipse半轴.06/.05mm及2 sweeps全部属于工程测试fixture，未写入生理默认；原生HC SOURCE_VALUES保持原值，共同反馈拓扑、q/p coneID、逐分区配额和同实例位置一致。验收见[HC定向测试入口](../reports/hc_acceptance/latest.json)，03:51旧图不作为此次验收。测试输出编码和模型字段名的检查脚本问题已修正，完成run经相同assembly hash的确定性重建检查，未增加模拟时长。2 sweeps的接受率.572067不证明混合、稳态分布或生理统计；中央真实arbor/深度/区室、独立S/H2与BC/AC/RGC/rod内层及绝对光学/release仍未标定。可见GUI窗口操作、长期适应、全域生成、研究比较和拟合未测，DeepGaze/SC继续暂停。named profile和运行审计保留实现阶段标签，当前工程执行证据以独立acceptance及版本hash为准，不将其升级为生理验证。

## 有界背景准备与中央 OPL 支持的本轮整合

背景实际入口现在有两个显式选项：`fixed_prerun` 保留旧短预运行配置；`bounded_convergence` 使用 [bounded_uniform_background_v1.json](../configurations/bounded_uniform_background_v1.json) 和 `retina_sim/background.py`，经 RunConfig、GUI、backend、CLI 的同一背景单位/来源传入原模型。CLI 使用 `--background-initialization bounded_convergence`、`--background-max-sim-seconds`、`--background-max-wall-seconds`、可选 `--background-policy-json` 和 `--background-require-converged`。所有原时间常数、噪声和动力学方程保留，没有以加速或调参取得收敛。

标准数值 policy：窗口 .1 s、连续 3 个完整窗口；最短时间为 `max(1 s, 5*最大已登记tau, 2*最大delay, 随机历史washout)`。当前 vH tau_a=.25 s，因此下限至少 1.25 s；没有把 latency/TTP 当模型 tau，未知慢模态明确列出。simulation cap=10 s 仅为配置上限，另有 wall-time 与观察点预算；本轮未运行整网 1.25/10 s。`den_i=atol_i+rtol*scale_i`，rtol=1e-4；V 的 atol=1e-5 V，归一化状态为1e-4；原生任意单位、Chen scaled concentration、uM/s 等都有单独登记的工程尺度，scale 冻结为初态绝对值与该单位数值floor的较大值，未登记单位返回 INCONCLUSIVE，不能套用电压容差。

检查包括原积分过程实际事件端点与数值监测网格上的窗内最大 excursion、`W*max|original RHS|/den`、原 causal_history/author_discrete 一步更新缺额按 native period 换算的 residual、有限值、已声明正化学状态/单位区间资源状态、对称 gap 电流守恒、delay/filter queue 与采样相位一致性。不只比较窗首尾，也不能靠缩小 dt 把 RHS residual 隐藏。结果是 sampled numerical fixed-point candidate，不是数学连续极值证明或全网络生理稳态。不足完整窗口/最短时间或残差不通过为 NOT_REACHED；状态/尺度覆盖不足为 INCONCLUSIVE。预算或取消退出保留最后有效状态及主要 component/cellID，不缓存为已收敛结果；require_converged 会在媒体记录前停止，CLI 返回非零状态。

HistoryRGC 当前 graded V 不受自身随机 spike/reset 反馈，因此 V 纳入确定性检查；随机计数历史、pending events、advanced RNG 独立推进并按 `(K+1)*dt` 报告 washout。未用“短窗口没有 spike”判稳态。stationarity statistics 未启用，状态为 STATISTICS_INSUFFICIENT；如将来启用，须明确至少1 s统计窗、至少3窗、mean/variance/rate/autocorrelation接受容差与有效样本数，不能捏造统一误差条。内部绝对 time、grid_origin 和事件 timestamps 不归零，只保存实际 stimulus_origin 来映射外部媒体相位。Checkpoint 保存所有模型状态、历史/filter/delay队列、tick indices、RNG advanced state 与输入/背景上下文；restore 保留既有assembly/model/parameter/graph/code identity校验，额外允许显式expected_input_context校验。没有新 adaptation cache。

中央 HC 新增可选 [hc_arbor_central_W2000_v1.json](../configurations/hc_arbor_central_W2000_v1.json)，GUI/RunConfig `hc_arbor_profile=central_OPL_equal_area_W2000` 和 CLI `--hc-arbor-profile` 接入已有 joint mosaic。H1/H2 的 .00804/.01222 mm 是 `sqrt(C/(pi*32000))` 的等面积 OPL 支持：H1 C=6..7 取6.5、H2 C=15。引用 Wässle2000 p599、DOI10.1017/s0952523800174097；报告算术、面积到圆盘、轴对称与跨动物/种属转移为 project_approximation，未称实测最大枝长或RF。Packer2005 16 um 与推算16.08 um是部分同源 numerical consistency，Boycott独立数值不平均拼配。该段精确source species尚未由本地原文核定，不能从整篇索引的三种macaque武断选一种。

中央有限完整 arbor、HC anchor/soma及实际同ID cone pedicle 的支持必须全部落在 OPL `.5..1 mm`；不向 `.0..5 mm` 内延，不强制逐细胞C/D整数，不扩大半径、补nearest或转为其他生理参数。默认source_patch与null显式arbor仍保留；旧模式、手填半径或JSON足迹和中央named profile冲突会拒绝。`central_OPL_support` 导航ROI使用RF .35 mm中心、.002 mm宽/高和.04 mm halo，现有McGregor adapter将中心映射到约.75 mm OPL；它是明确工程导航选择，实际完整映射仍检查，不把RF .75 mm直接当OPL .75 mm。

中央15个配置/小接触fixture已通过。首个中央整网构建曾在AII→midget零接触/非空CellTable要求处停止，保存的 [真实错误](../reports/background_acceptance/20261011T105852783731Z/run/20261011T105853128480Z_5d3d330d/error.txt) 和配置未删除。随后定位到RF/IPL生成域混用：RF中心.35 mm的BC终末已在约.75 mm IPL，而原AII仍只在RF数值矩形内生成；这是支持域未生成，不能称生物学零接线。`aii_ipl_support_domain`保留已覆盖owner的周边原域；中央需要时以既有矩形等质量后端生成原rod/RBC patch与全部实际BC IPL终末支持的物理包络，包含中间组织，仍用原5000/mm²项目转移密度、原.04 mm硬连接半径及预算，未补nearest、未降低密度、未推断新枝长。owner为两个实际patch的并集，中间组织为halo；BC terminal与cone pedicle共位、rod/RBC的原点坐标代理仍明确是现有项目近似。

有限支持域完整但候选确实为空时，既有DoubleExponentialChemicalSynapse现在复用真实target population/IDs作为无输入、无状态的精确空和算子，输出零S；不创建假细胞/接触、不给g_peak伪造零值。支持域不完整则报告IPL_SUPPORT_DOMAIN_UNGENERATED，不能走空算子。4项定向回归通过，包括实际装配/短推进的空和算子；2项必要周边原配置回归通过，周边既有可运行域保留。相同失败中央GUI配置的一次最小构建产生AII_FMB_glycine=1254、AII_DB1_glycine=315，两组全部target支持覆盖，最大实际边距≤.04 mm；整网编译仍拒绝13个必需输入缺额，集中在RBC→AII、BC→三类AC、AC/BC→部分RGC。现有RBC/AC点坐标仍是RF/局部点代理，部分RGC dendrite取mapped soma代理，与BC IPL点及固定硬支持不匹配；没有来源充分的区域arbor/终末位置合同，本轮不增长半径、重布细胞或将这些必需输入静默置零。中央结果为BLOCKED_REQUIRED_INPUT_SUPPORT、time_integration=NOT_RUN，原始partial audit和逐端口错误见 [central build acceptance](../reports/central_build_acceptance/latest.json)。最小构建后仅修正周边owner覆盖守卫及owner/all-target审计字段区分；中央几何和上述边数不变，未重复中央构建或积分。

随后按13处缺额继续修复，当前状态已更新到同一 [central build acceptance](../reports/central_build_acceptance/latest.json)。AII扩域后，上游rod/RBC仍生成在原RF小矩形是可确定的工程缺口；现在每一层以目标物理域外加原.04 mm连接支持生成上游组织，分别使用同一个rod场与`rho_RBC=(2/36)*rho_rod`，没有对rod/RBC套cone位移、改变密度值或改变连接长度。RBC生成矩形为`[.269,.875914722,-.169018813,.167118095] mm`，rod为`[.229,.915914722,-.209018813,.207118095] mm`；每个新增细胞实际消耗原预算，超限仍拒绝。原AC/parasol RF scaffold单独保留原ROI、seed及10个RF位置/IDs，不把新生成的471个物理RBC误作471个AC/RGC。三类effective AC的IPL点现在随原RF位置经既有McGregor pedicle map定位，与已选BC的co-located pedicle/terminal点近似一致，明确为BC-path有效点适配，不是AC soma/arbor测量。

最后一次小中央构建只生成/编译，不积分：471个RBC均有40条rod输入、675个AII均有3条RBC输入；local_feedforward/local_crossover各10个AC均有4条对应BC输入，widefield的10个AC均有16条BC输入；上述路径degree无缺额且实际边距≤.04 mm。AII_FMB/DB1的1254/315条边保持。原13个编译问题已减少到9个RGC输入路径，现有2µm RF owner内没有任何零行目标，下面零行都属于参与计算的halo；这不豁免halo的必需输入，也不证明owner响应收敛。受影响的4项finite-domain/empty-operator和2项原周边source回归在本次修改后全部通过（8.34 s），无长积分。原生HC源码及所有动力学参数保留。

| 当前剩余path（ON/OFF分别算一个） | 零行/目标数 | 零行最近合法类型source距离（µm） | 已定规则之间的具体问题 |
|---|---:|---:|---|
| local_crossover→midget ON、local_feedforward→midget OFF（2条） | 每条10/418 | 40.346..50.868 | 私线dendrite在parent BC IPL点；原10点AC RF scaffold的mapped halo不足，不是再一次RF坐标混用。缺少明确的独立AC生成/halo合同；不能降低degree=1或补跨40µm边。 |
| diffuse ON→parasol ON、diffuse OFF→parasol OFF（2条） | 每条3/10 | 42.248..71.721 | BC terminal=P，RGC dendrite默认=soma G；P=q+.85(G−q)。McGregor给的是soma/pedicle源衍生图，不提供这两型的单细胞IPL arbor或BC terminal形态。 |
| widefield AC→parasol ON/OFF（2条） | 每条7/10 | 43.889..71.502 | 有效AC点=P，RGC点=G；同时原10点scaffold稀疏，8条输入目标没有完全覆盖。 |
| S_ON_BC→small_bistratified（1条） | 10/25 | 46.838..71.699 | S BC的25点在P，small-bistratified默认dendrite在G；degree=8的23个目标有数量缺额，尚无typed arbor/terminal域合同。 |
| local_feedforward AC→small_bistratified（1条） | 15/25 | 46.852..108.266 | 同一P/G点代理差与稀疏AC halo，25个目标均未满degree=8。 |
| diffuse OFF→small_bistratified（1条） | 8/25 | 58.606..77.614 | BC点P与RGC点G及生成域差；17个目标未满degree=8。 |

固定40µm来自`macaque_constrained_native_v1.geometry.hard_support_mm`的project cutoff，不是这五型的实测arbor半径；P/G来自`mcgregor2018_production_map_nodes.json`及`McGregorDisplacement`的具名源衍生图。当前冲突在所选point/halo模型合同，不能反称原论文生理数值互相矛盾。例：diffuse_BC_ON:100在IPL`(.792207660,−.026881126)`，parasol_RGC_ON:6在`(.833671017,−.018778596)`，最近距离42.2476µm；local_crossover_AC:2在`(.724971221,.045533210)`，midget_RGC_ON:4在`(.696366432,.077163493)`，42.6463µm；S_ON_BC:22在`(.794746710,.006552366)`，small_bistratified:14在`(.830320253,−.023916713)`，46.8385µm。所有缺额行的真实ID、坐标、nearest距离、目标degree及归一化量保存在既有partial_build_audit，acceptance列九条实际path；完整typed IPL支持尚未确立，不将零候选标为生物学零。后续需要明确AC计算scaffold/halo与非midget的BC terminal或RGC dendritic point/arbor选法，并联动其上游输入域；仅单独扩大AC生成域会再产生无BC输入的外围AC，不能据此宣布补全。本轮没有擅自把RGC soma移到BC、增大连接半径、稀释密度、减少degree或补假边。

Patterson固定commit的两个可选资产因DNS未取得，[reference manifest](../data/source_snapshots/patterson_SConeEdgeDetection_eba20bdd5/patterson_anatomy_reference.manifest.json)保留期望hash、local tissue XYZ/um与未配准/未导入声明；没有扩展来源模型或升级环境。

有界背景的最终端到端选择已有可运行的来源受限周边2D ROI，经真实Tcl/RetinaApp配置读取→SimulationSession→原39组件20层执行：.2 ms恒背景准备+.2 ms媒体记录、1窗口、76187个数值有限，实际stimulus_origin=.0002 s、grid_origin=0。返回NOT_REACHED，无未登记状态尺度；下限1.25 s未达，主要原RHS residual是AII_AC:39，部分窗口 norm=2.261686；随机washout也未完成。图从同一已保存结果导出，没有replay。必要回归60项通过（26.42 s），覆盖背景窗内/RHS/离散检查、预算、同背景checkpoint续跑的states/RNG/delay/ticks、原HC30项、中央15项和旧source-patch3项；另有author phase守卫与实际适应时长/状态元数据对齐各一项检查，共62项通过。端到端后仅补严格监测点预算记账、author phase守卫及实际适应元数据对齐，方程未变、整网未重复模拟，run自身core hashes仍保留。原生HC源码SHA仍为cfc4ce4f9eb1015755cd790b180faaec83d2a9a7b8da083f3a471a00d36a4900。证据与图见 [background acceptance](../reports/background_acceptance/latest.json)，不把工程运行、短适应流程或中央小fixture升级为physiological_validation。


### 已批准的 P-chart typed effective IPL v1：实现与当前可执行范围（2026-10-11）

`configurations/p_chart_typed_effective_ipl_v1.json` 和 `retina_sim/ipl_geometry.py` 实现本轮已批准几何合同。中央 midget 保持 qualified parent BC 私线；small-bistratified 以冻结 S-parent P 点为锚，ON/OFF 是不同分类区域而不是捏造的 z 深度。Parasol 在原 RF seed 映射的 P 点 40 µm 球中，以全部正确 excitatory BC 候选等权质心冻结一次，再独立查询最终接触；无候选即 ANCHOR_UNRESOLVED，缺额不触发迁移或扩张。非 midget 的 legacy G soma 只用于展示，IPL dendrite 单独存储。

Local feedforward/crossover AC 在冻结 BC/cone 与 RGC sampling site 上各自生成独立 effective state，按 site ID/type 去重。Widefield AC 使用原点 (0,0)、h=40/3 µm 的全局方格整数键。这些是计算采样单位，未指定生物 AC 密度。保持 degree1/8、40 µm 硬支撑、原 activation pooling/gain、绝对每边 chemical/gap conductance、原方程、shared PCG64 和 solver；无新外部背景、无删除动态 halo、无新的 recurrent boundary operator。有限 induced graph 的 recurrent closure 仍 NOT_ESTABLISHED。

生成 guard 按潜在 seed-centroid、AC、BC-input 支撑与一格 rounding envelope 规划；通过既有 P chart 反解 RF 矩形 envelope。129/side 边界采样加 1 µm 是数值 bounding 近似，不是连续几何覆盖证明或新生理半径。DB stride 在完整 canonical cone instance 上先取 LM[::4] 再裁 output cohort；同实例裁剪保持 ID，不声称不同 ROI 独立实例无接缝。上游有限 D/source range 或预算不足仍阻断，不能将缺候选称为真实动物零连接。

GUI/RunConfig、CLI `--ipl-geometry-profile`/`--requested-owner-output`、full-target builder 和导出已经接入。新 RunConfig 选择 typed v1；没有该字段的历史 JSON 显式迁移到 `legacy_point_ipl_v1`。GUI 增加必要模式和 owner 输出选择及 owner/halo 视图，视图过滤不裁模型状态。Manifest/CLI export 记录 profile hash、P anchor、分类 stratum、owner masks；失败构建也保存 partial build audit。`configurations/source_roi_legacy_native_replay.json` 是已通过旧来源 ROI E2E 的显式 legacy 回放配置，供回退；它不是 typed v1 成功结果或严格生理配置。

当前真实验收分开记录：

- 原来源受限外围 ROI 的 GUI→backend E2E 仍是 `reports/background_acceptance/20261011T110123870904Z`：39 components/20 layers、0.2 ms 背景加 0.2 ms 媒体、76187 个有限数值。其图 `background_flow_response.png` 对应这次 legacy IPL 运行，不对应新 typed 中央全网。背景 NOT_REACHED，生理验证 NOT_RUN。
- 新 typed v1 中央完整构建 owner=[.34,.36]×[-.01,.01] mm：测试配置的 depth20/partitions8192/order12/rtol1e-4 已解除 HC_PARTITION_UNRESOLVED、partition budget 和积分一致性阻断；sweeps2 未认证混合，原 HC 默认没有改变。随后真实构建在 HC arbor source-domain 检查停止（`reports/central_build_acceptance/20261011T145038434526Z`）：该 RF rectangle guard 映射的 potential active HC radial envelope 为 .473831..1.034709 mm，超出 named central arbor .5..1 mm。矩形 guard 的过覆盖不能转称生理不可能；未裁边、延长 arbor 或用背景替代未知 HC。
- 同一新 guard 的外围 source-patch 构建在 H2 owner contact quota 停止（`20261011T145916294574Z`）。没有改配额或远距离 nearest 填充。
- 共享 `typed_layouts` 另行在真实 Packer/S/McGregor 的未缩放 6102-cone canonical instance 上完成几何诊断。该步骤不生成 HC dynamics，也不绕过完整网络的 HC 拒绝；完整 native assembly/time integration 均 NOT_RUN。RF output cohort 仍 [0.30,0.40] × [-0.05,0.05] mm，实际全部 RGC 为 midget ON/OFF 各625、parasol ON/OFF 各14、S-ON36；owner 为25/25/0/0/1。请求的 owner=midget ON 非空，parasol owner 空需要另行明确声明 owner scope，不能插入目标细胞。
- 新几何的旧九条问题路径均无零行；八条 degree 满额，S BC→small-bistratified 还有16个 halo 目标只有3..7候选，保留 strict degree8 失败。Owner S-ON 的 degree8 已满足，但不能删除16个 halo 目标来通过整个网络。这是有限候选下的 effective contact-capacity conflict，不是已测得的 S convergence。旧诊断用2 µm owner，新诊断20 µm owner且增加 generation guard，不作同实例生理统计对照。

| 旧九条路径组 | 新几何零行 | 未满 degree 的目标 |
|---|---:|---:|
| local crossover→midget ON / feedforward→midget OFF | 0 / 0 | 0 / 0 |
| diffuse ON/OFF→parasol ON/OFF | 0 / 0 | 0 / 0 |
| widefield→parasol ON/OFF | 0 / 0 | 0 / 0 |
| S BC→small-bistratified | 0 | 16（全为 halo，degree3..7） |
| feedforward→small-bistratified / diffuse OFF→small-bistratified | 0 / 0 | 0 / 0 |

最新 source-only geometry 与完整失败引用在 `reports/central_build_acceptance/latest.json`；原失败/来源/结果保留。24 个直接相关回归通过（typed geometry7、central support4、runtime/background/source identity13），XML=`reports/typed_ipl_20261011.xml`。其中原生 HistoryConductanceRGC 的1.2 ms software fixture 实际跨过 spike tick、推进 shared PCG64，使用新网格的8条归一化输入边；不是来源中央全网积分或生理验证。未运行长适应、训练、研究对照、统计稳态或全型物理校准。

本轮仍不能将新 typed v1 标为整体可运行来源生理默认：完整上游 HC finite-D/arbor/quota 与16个 S halo strict capacity 尚需可支持的几何/来源合同；不扩大未标定科学参数以凑成功。S/H2 kinetics、BC/AC/RGC/rod/RBC/AII 的项目近似及独立 Chen/Maddox source reference 保持各自原分类，未转称 measurement。


已按原失败外围配置、seed=20261010 补取静态 H2 诊断，未重新积分或换 seed：owner 共4个 cone，H2 generated127；cone index634，RF=P=(4.594193785978159,-0.005788075316585273) mm，连续配额3.0093047225432894，原 rounding 后要求3，40 µm 内实际候选2、degree2、缺1；两条边距离38.37259/38.53517 µm。H2 密度947/mm2来自既有 Wassle Table2 animal29.7.99 的4..5 mm interval，primary verification pending；配额由既有2.2→3.2插值及 largest-remainder rounding生成，3.2是2.7..3.7中值，40 µm是项目 cutoff而非 measured maximum arbor。精确记录在 `reports/central_build_acceptance/latest_blocker_details.json`。中央记录的 .47383145988653197..1.0347087609500147 mm 是 required active radial envelope；原失败实际 HC point min/max 未落盘，不能以该 envelope 代称实际点范围。未为补取这些诊断重跑80 s的joint sampler。
## 2026-10-11 最终三规则合同的实际结果

新增独立版本 [retina_contact_geometry_effective_v1](../configurations/retina_contact_geometry_effective_v1.json)，旧 `p_chart_typed_effective_ipl_v1`、旧 HC arbor descriptor、旧结果和原始来源保持可选、可追溯。新profile首次尝试于15:46因真实空反馈行拒绝，历史失败保留；随后用户自主继续授权下，明确采用具名零接触执行合同，完成 **全网络构建、背景初始化和连续20ms刺激积分**。这是一条已运行的来源受限研究原型路径，动力学仍未标定。

HC 使用 `finite_geometry_first_v1`：现有独立 H1/H2 mosaic、cone subtype/OPL 有限 disk/ellipse/节点 narrow-phase 与最终 40µm cutoff 的交集决定所有合法 cone–HC partner，不做逐 cone 的文献均值 rounding quota、不补边、不 rescale。forward/feedback 仍用现有无量纲 pooling 规则，chemical/gap gain 不变。导出完整 edges、同一合法图上旧 rounded-selector 的明确 counterfactual、保留边权重和前后 row totals；该 counterfactual 不代表旧完整模型成功。人口统计分别记录 owner cone 的 D_all（H1 包括 S 零值）、D_eligible 和完整 owner HC 的 C、直方图、均值、零率、reference residual 与边界流量，标记 `population_contact_uncalibrated`，没有人为生理通过阈值。

[W2000_equal_area_bounded_edges_047_104](../configurations/hc_arbor_W2000_equal_area_bounded_edges_047_104.json) 保留原 source anchor band [.5,1] mm，只把 **anchor 应用范围**声明为 [.47,1.04] mm 的 constant-footprint 项目外推；H1/H2 radius 仍为 .00804/.01222 mm。全 disk 可延伸至 [.45778,1.05222] mm，这是查询 guard 上界，不能自动授权更远的 anchor、增加半径或变成新测量。实际 H1/H2 owner/halo anchor、完整 footprint、active generation envelope、required pedicle query、provider 支持和 outer-halo 截断分开记录。完整 owner 及其直接接触 HC 的 query 必须有覆盖；已有 finite-induced 外层边界及未证实 recurrent closure 明示保留。

只对 S_ON_BC→small_bistratified_S_ON excitation 使用 `S_ON_effective_positive_input_legacy_selection_v1`：原 selector 最大 8、原边和 normalized weights 保留，验收是至少 1 个合法输入，零输入仍失败。实际 3–7 的 halo partner 数属于 effective_uncalibrated，既不是 measured BC convergence 也不是 ribbon count。其他旧 degree1/8 和 hard support 条件、各 AC typed state、动态 halo、shared RNG/solver 均保留。

实际新 profile 尝试来自 GUI 字段解析，owner RF=[.34,.36]×[-.01,.01] mm，seed20261010，原密度不缩放；只显式声明 HC 数值预算（2 sweeps、depth20、partitions8192、quadrature order12/rtol1e-4/8M，未认证 mixing）。[失败目录](../reports/central_build_acceptance/20261011T154651831130Z/acceptance.json) 和 [精确 blocker](../reports/central_build_acceptance/20261011T154651831130Z/blocker_details.json) 保存真实失败及静态 geometry，不修改此前严格失败。

- 新图有 6102 cones、7804 active HC、33798 合法边；counterfactual rounded-selector 保留 29321 边，新规则增加 4477 边，未改变 gain 或 gap。
- Owner H1 D_all=3.6923076923、D_eligible=3.84、C=5.2222222222；H2 D_all=D_eligible=2.0769230769、C=12。D/C 的 owner edge sums 为 H1 96/94、H2 54/60，差异有实际跨 owner 边界边解释，未强制相等。
- 实际 halo anchor 径向 H1=.4746361222..1.0304167624、H2=.4752884431..1.0318768889 mm；halo 全 footprint 分别 .4665961222..1.0384567624、.4630684431..1.0440968889 mm。169 anchors 使用 bounded extrapolation；1027 外层 HC footprint 的 cone query 截断明确导出，38 个直接 owner 依赖 HC 的完整 query 未截断。
- 11 cones 没有合法有限 HC feedback：10 S、1 M，全部是 halo。它们不是被quota删掉的边，也不是原先cone634的旧point-proxy两候选。15:46首次尝试中，原 `NativeConeH1` 的B row非空、unit-DC条件使绑定拒绝；此失败未改写。后续授权的具名适配及实际积分证据见下文。原严格机制仍拒绝空行，不暗加background、nearest、arbor reach或删除动态halo。
- 在相同未缩放 cone 实例上，共用实际 `typed_layouts` planner 的静态 IPL 检查具有 owner midgetON/OFF=25/25、S-ON=1、parasolON/OFF=0/0；请求 midgetON 非空。所有路线 mandatory-input 条件通过，16 个 S halo 的旧 degree8 mismatch 保留记录，新的 minimum1 满足；其他路线无 degree 缺额。这是静态检查，不能替代失败的完整 native assembly。

GUI/CLI默认已切换到通过验收的新研究原型：owner RF中心(.35,0)mm、20×20µm ROI、40µm halo、原来源密度、最大population预算60000；不是完整中央—周边二维视网膜。默认2ms有限prerun及6ms录制仍是数值瞬态，不声称背景稳态；下面20ms验收使用落盘配置中的有界准备模式与0.2ms预算覆盖默认时长。GUI只显示“研究原型 · 动力学未标定”等简短状态，工程预览与旧“源约束混合”单独可选。`configurations/source_roi_legacy_native_replay.json` 保留完整legacy运行配置；缺少execution字段的旧GUI文件继续迁移为旧工程预览，不被静默升级。CLI `--profile macaque_constrained_native_v1` 保留legacy source路径，`--profile engineering_preview_v1` 保留工程预览，旧严格IPL通过 `--ipl-geometry-profile p_chart_typed_effective_ipl_v1` 显式选择。

此前GUI→backend legacy source-ROI E2E保留：[实际验收](../reports/background_acceptance/20261011T155435642008Z/acceptance.json)，39 components/20 layers、0.2ms uniform background preparation + 0.2ms media、76187个有限录制值；[legacy geometry/trace图](../reports/background_acceptance/20261011T155435642008Z/background_flow_response.png)。背景状态NOT_REACHED，最低要求1.25s；该历史证据不代替下面的新有限图验收。

53 个直接相关回归已通过（新合同/typed IPL/中央 arbor/参数与冻结来源/背景 preparation）。初次运行有 37 passed、1 个更新默认值 assertion、2 个 pytest 默认 temp 权限错误；修复后另外 16 项通过，最终 7 项新合同与默认 migration 检查通过。对应 XML 在 `reports/research_contact_regression_20261011.xml`、`research_contact_repair_20261011.xml`、`research_contact_final_20261011.xml`，保留失败记录而不重写成 PASS。原 `vh2005.py` SHA256 仍是 `cfc4ce4f9eb1015755cd790b180faaec83d2a9a7b8da083f3a471a00d36a4900`。

本轮单位/输入背景统一、bounded preparation、实际参数 profile、冻结来源身份及 measurement/type-domain 检查可用。未知 S/H2 kinetics、BC/AC/RGC/rod 内网参数、Chen LM>20deg 独立 reference 和 Maddox paired activation 的科学缺口仍按原分类保留；没有环境升级、研究对照、训练、长适应运行或未授权新增物理校准声明。

### 经授权的真实零接触执行与最终验收

新机制 `VanHateren2005FiniteContactZeroFeedbackApproximation` 和路线 `native_finite_geometry_zero_contact_approximation_v1` 仅用于本研究原型。`finite_feedback_native.py` 对11条B空行独立重查全部类型合格HC：整个候选disk处于实际生成矩形内，anchor应用径向范围完整，有限disk/类型规则确实得到零partner，且对应A column也为空、A/B拓扑互易。10条S行仅查询H2；M行查询H1/H2。所有查询完整、独立合法partner数为0；最近距离也保存在 `hc_geometry.json`。缺域、应用域不足或图查询不一致仍硬拒绝，不能称真实零。

空行的反馈贡献严格为原native尺度中的空集和0；无空集均值、除零、虚构HC、补边或人口删除。非空A/B仍要求非负、有限、unit-DC，原参数值、`signals`、`derivative`、`observe`和作者时钟均沿用原VH实现。背景用实际A/B求解同一非线性方程；零行release使用实际无反馈项和原aI，未套用旧含反馈值。此次12次函数评估后最大native残差为1.4210854715202004e-13。这只验证外层该平衡方程，不代表整个BC/AC/RGC网络稳态。原 `vh2005.py` SHA256仍为上文固定值，原严格组件及其他必需通路零输入拒绝器不变。

实际实例6102个cone、7804个生成active HC、33798条合法HC边。沿用原disconnected-HC执行视图，5631个有cone输入的HC参与动力学；2173个没有任何输入、也没有连接到动力学gap的HC保留为geometry-only，不伪称已模拟反应。11个零反馈cone全部保留，所有原来耦合的动态halo、RGC output cohort与共享solver保留，适配器RNG调用为0。

最终 [GUI字段解析→完整backend验收](../reports/central_build_acceptance/20261011T163543160525Z/acceptance.json) 为PASS，实际 [run config](../reports/central_build_acceptance/20261011T163543160525Z/run/20261011T163543490912Z_d5256453/config.json) 和 [geometry/owner曲线](../reports/central_build_acceptance/20261011T163543160525Z/research_prototype_response.png) 可复核：34组件、20层，非空owner midget ON=25；0.2ms有界背景准备后，连续20ms刺激、10个2ms窗口，录制26052690个有限数值。输入为uniform sRGB128经具名display approximation产生的native rate contrast step，并非校准光刺激。20ms覆盖native tau_E=16.8ms及20个BC/RGC历史tick，属于有意义的工程瞬态，不能解释成生理潜伏期或稳态反应。背景始终NOT_REACHED，未达到1.25s最短检测条件，不以0.2ms准备声称适应完成。

总wall time341.5118s，其中积分81.6973s、227次solver call、6955次RHS评估；其余包括源几何生成、装配与完整元数据/检查点/录制导出。资源估计234231个显式机制/群体节点、466204条边、309.4MiB；该节点数含重复机制绑定与接触，不是动物细胞总数。首次完整build/准备后曾被旧64MiB录制限额拦住（需要264429736 bytes），新profile显式512MiB录制预算和CLI/GUI读取已接通，未抽稀细胞。JSON仅压缩排版，证据值未删除。另一次20ms尝试到18ms在最后输入终点因一个ULP的浮点覆盖比较失败；media输入现在与执行窗口共用author-tick终点，保留真正越界拒绝。两次实际失败目录 `20261011T161911026606Z`、`20261011T162717700107Z` 均保留，未改成成功记录。

最终checkpoint的77组状态、626473个状态值全部有限，所有初始shape保留，model hash与grid origin未变。五类RGC（625/625个midget ON/OFF、14/14个parasol ON/OFF、36个S-ON）的PCG64状态均实际推进；next history tick index=21表示沿用原时钟在0、1、…、20ms采样，未重置时间或改变RNG消费顺序。此额外核查写入同一acceptance，不新增研究对照。

[冻结profile的验收receipt](../configurations/retina_contact_geometry_effective_v1.acceptance.json) 用实际profile SHA256、checkpoint model hash和run路径绑定本次成功。模型JSON中的 `execution_acceptance` 是创建时的pending快照；为避免改变原始profile哈希及由它生成的细胞ID，不为更新验收状态修改模型文件，当前执行状态以该receipt及实际acceptance为准。背景/来源自动检查是数值与身份检查，不能充当primary verification或生理验证。

追加必要回归：零反馈方程/非空数学完全一致/缺域与错误图拒绝/非unit权重与不互易拒绝/最后media时钟覆盖共5项通过；默认与历史配置migration2项通过（`reports/zero_contact_time_boundary_20261011.xml`、`reports/research_prototype_default_20261011.xml`）。加此前53项，共58个相关测试已通过，未重复全套研究检查。HC population contact与有效S输入未标定、2-sweep生成器mixing未认证、bounded anchor外推与1027个outer-halo截断/外部recurrent closure未认证，均保留明确近似标签；本次PASS只证明具名有限实例可运行。

### 均匀背景闭环与连续刺激最终交付（2026-10-11）

本轮在同一owner RF=[.34,.36]×[-.01,.01]mm、40µm halo、seed20261010和未缩密度的有限实例上完成完整数值背景准备。入口为 `tools/run_background_preparation.py`，使用既有 `SimulationSession` 和 `bounded_uniform_background_v1`。从合法均匀背景执行全新 `runner.initialize`，native背景100td、rod背景5R*/s；physical-cone的1000R*/s仍明确声明，但中央ROI没有适用的Chen LM>20deg物理支路。本次没有使用先前刺激checkpoint作为初始化，没有更换背景输入、方程、时间常数、solver、容差或人口数。

原有标准保持0.1秒完整窗口、至少3个连续通过窗口、0.1ms监测网格、relative tolerance=1e-4及原unit-specific数值尺度。最低时长由原规则得出1.25秒（已注册最大时间常数0.25秒的5倍）；两小时wall预算仅是工程限制。实际运行13个完整窗口，每个均覆盖77个注册状态数组；0.7至1.3秒连续7个窗口通过，在1.3秒得到 **CONVERGED**。最后窗口最大归一化残差为 **0.0014698492122320929**，主导项 `AII_AC.x` 原始RHS、cell=`AII_AC:16`；全程没有domain或history consistency失败，没有未覆盖状态。背景wall时间6244.642981秒（104.08分钟）、30504次观察，没有触及预算。

原状态随即接上同一uniform sRGB128工程对比输入，连续20ms、10个2ms窗口完成，最终绝对时间1.32秒。刺激是显式display approximation，不能解释成显示器或生理光量校准。完整job wall时间6578.896736秒（109.65分钟），原solver执行14828次调用、436888次RHS求值，积分本体4890.207932秒。最终77数组、626473个状态数值全部有限，形状和model/graph/parameter/solver绑定保持一致；约2605万个记录数值全部有限。五类RGC的PCG64均在背景和刺激阶段继续推进，next tick=1321，grid origin仍为0，未在刺激开始或窗口之间重置状态。

本次最终证据集中在 [closure.json](../reports/background_preparation_runs/20261011T170043172502Z/closure.json)：[背景及连续响应图](../reports/background_preparation_runs/20261011T170043172502Z/background_and_response.png)、[77状态末窗残差CSV](../reports/background_preparation_runs/20261011T170043172502Z/final_background_state_residuals.csv)、[真实owner midget ON电位CSV](../reports/background_preparation_runs/20261011T170043172502Z/prepared_owner_midget_ON_voltage.csv)。曲线使用本次连续轨迹的21个实际样本，不使用旧短暂初始化运行代替。完整逐窗口/RHS/单位数据、initial/prepared/final checkpoint、延迟历史和随机状态保存在同一 [run目录](../reports/background_preparation_runs/20261011T170043172502Z/run/20261011T170043213842Z_8b6efb04)。冻结profile receipt追加本次closure引用与当前model hash；原profile SHA256及原VH文件SHA256均未变化，旧20ms验收绑定保留。

**34与旧39组件的差别是源适用域。** 既有导出比较仅缺少 `physical_OS`、`physical_current_SI`、`physical_membrane`、`physical_terminal`、`physical_HC` 五个独立外围LM物理参考组件。该链仅适用于LM>20deg，本次中央ROI无适用cone，manifest明确 `included=False`、count=0及原因；cone outer-segment必需层仍由 `outer_native` 实际执行，两份导出的20个必需机制层相同，当前遗漏必需层为0。没有将外围物理参考移植到中央或S cone，也没有删除其源实现。

长准备首次尝试在1ms遇到记录时间ULP覆盖错误，失败目录 `20261011T165355921727Z` 与原runtime快照保留。修复仅使密集监测点使用绝对整数tick网格，并让记录器在已有1e-12时间覆盖容差内保留请求时间戳；积分端点、作者period、event/RNG更新及科学阈值不变。两项必要时钟回归在当前修复下通过：dense mixed-clock从零开始的覆盖，以及history tick近ULP采样的时间戳和RNG等价（`reports/background_timestamp_repair_20261011.xml`）；前序14项背景回归结果保留于 `reports/background_full_clock_regression_20261011.xml`。工具短暂断线后反复确认同一现有积分进程与持久checkpoint继续推进，没有重复启动或从头计算已完成背景。终态进度文件明确写入CONVERGED并指向 `prepared_checkpoint.json`；`latest_checkpoint.json` 随后可继续更新为刺激轨迹，不能再作为背景终态引用。

GUI默认控件与配置加载通过；Start/Cancel事件接线使用明确的轻量UI worker stub检查，真实积分由上述唯一 `SimulationSession` 完成，未重复启动第二个科学worker。历史曲线导出smoke保留在 [gui_smoke.json](../reports/background_preparation_runs/20261011T165355921727Z/gui_smoke.json)，本次真实连续曲线另经 `ResultStore.export_curve` 导出。默认GUI短prerun仍明确不声称稳态；需要本次完整准备时可加载该run的 `config.json`，再运行会进行新的准备计算。

**CONVERGED仅表示此有限实例在既定监测尺度下的确定性数值准备通过。** 报告保持 `stochastic_status=STATISTICS_INSUFFICIENT`、`whole_network_steady_state_claim=False`、`physiological_validation=NOT_RUN`；已注册时间常数不是所有耦合慢模态的上界，离散监测也不是连续极值证明。S/H2共享kinetics、BC/AC/RGC与rod下游有效参数、HC population contact、bounded arbor外推、2-sweep mixing以及外部recurrent closure等未标定范围均保留。此前0.2ms短准备的NOT_REACHED是历史运行事实，不回写为本次CONVERGED；没有恢复SC/DeepGaze实验或引入新的科学校准方向。

## 中心至外围 atlas：实际来源、有限连接与剩余缺口（2026-10-12）

批准合同保存在 [中心至外围连续覆盖设计](中心至外围连续覆盖_批准设计_20261012.md)。现在已有按需全局 mosaic、五种坐标角色、有限 HC 候选及实际有效内层连接，GUI 可直接导出。新 atlas 仍有明确来源/联合几何阻塞，尚未绑定为闭合的全网络运行配置。既有研究原型、工程 preview、历史记录和完整背景成功实例保持原样；新图不能复用旧几何的背景状态。

### 实际文件和证据层级

四份来源 ZIP 及一份独立 S candidate ZIP 均实际保存在 `data/source_packages/20261012/`，字节数及冻结 SHA256 已核验；原 PDF、图、读图误差、provenance 与检索日志保留。官方下载 helper 未修改；Windows 所缺 POSIX xattr API 通过 NTFS 元数据流适配，Library ID/version 写入后读回核验，并有 `.library.json` 便携副本。哈希只证明资产身份，不证明生理有效性。

| 包 | 本机字节数 | 冻结 SHA256 |
|---|---:|---|
| HC | 3408818 | `6e6ac21073ecc189fed5e1d1736f22321e53af397e2930b24faf60cd8ceab58d` |
| Schein Henle | 84506 | `3bcbf83d595a2dd6cc08fe5e3832c6351d93f881e0e8ef6fdfaef4a1c1ce55f6` |
| photoreceptor | 2809438 | `bf7fd26555ef324240c12d8b81a63189a40d86a109b2384e23854b19d0e9ee84` |
| S public follow-up | 7368825 | `2346039c095ab7391ab441112fade80252a1edceb2fd9e35371067b225463e7b` |
| Godat S replot candidate（不启用） | 128726 | `84dc531469a5634ac2df0d4d859a599913e1a94d2af896909e1fede8cbfe30c1` |

`atlas_sources.py` 实际装载 HC 23 条 Table1/2 密度、Rohrenbeck 两个摘要锚点、Schein 343 个作者推导后数字化曲线点、Packer 116 个作者 map 重采样后数字化图点和独立 S 约束。HC 2–3、4–5 mm 保持区间、`eccentricity_mm=null`，不造中点测量。HC 密度包的直接定量 arbor 数组为空。Packer 发布代码 N/T 60 节点保留为 `published_code_tabulation`。

S follow-up 的实际新增数字仅为 Martin 2000 一个 M. nemestrina 2.2 mm field：1119 S/mm²、333 points、一个动物；方向、specimen identity 和生物误差未知。它登记为 reference-only，空间 source domain 为空，不把该点分配到整个 2.2 mm 圆或拟合主曲线。Calkins 重绘拟合、Godat 中央约5%和 Bumsted counterprofile 均不替代 Martin/Grünert 主曲线。

新增 Godat2024 Figure6A green trace candidate 来自 Martin/Grünert→Calkins→Godat转绘链，7文件已按包hash核验。11稀疏读数和305像素轨迹不是独立观测；可见r=0–.99672131mm，S占all-cones约5.27–5.60%，读图误差±.08个百分点/±.007mm，不能恢复50µm孔或1–6mm曲线。登记 `digitized_published_replot` / `inactive_candidate`、active=false、无空间/application assignment，未接主S partition或动态权重。原物种/主来源不改。

`coverage.py` / `atlas_source_domains_v1.json` 当前登记 **38 providers、44 冻结来源资产**，分开 source/application/query domains 与 source_supported/project_approximation/unsupported。source_supported 只表示具名资产和域支持；measurement、author-derived/digitized、发布代码、摘要 quotation 和项目近似没有互换分类。

### 全局身份、密度和坐标

`atlas.py` 明确 `rf`、`opl_pedicle`、`ipl_preterminal`、`ipl_post_arbor`、`soma` 五个角色。缺角色保持缺失，不自动复制另一层坐标。全局 ID 来源于 atlas/version、固定 mosaic 来源版本、population、规范来源键，不依赖 ROI、请求顺序和 owner/halo 身份。半开 owner 与 query 是同一对象的索引视图；空表合法，EMPTY_SAMPLE 和 BIOLOGICAL_ABSENCE 区分。缺来源/坐标的 owner 不生成数值零接触行。

`atlas_store.py` 独占写入，按冻结 SHA256 重载后验证 typed coordinate、table hash 和 ID。canonical_atlas 与旧 instance_archive 保持不同语义。下游坐标/实现元数据改变会使完整来源或连接资产身份变化，但不重新抽样 receptor RF、改其全局 ID。连接导出同时冻结完整 graph 文件 SHA 与各实施文件 SHA，不只记录 source ID 的存在。

`source_atlas.py` / `retina_atlas_v1.json` 用固定 32 µm 全局格、固定 source/config/seed 生成 Poisson count 和均匀点；每格中心密度的分段常数近似、无实测排斥统计。Packer 四方向径向/角向 log-linear 是项目插值，physical retinal mm→planar RF mm 是具名项目注册；中央 cone 保留发布代码桥接。Rod 四方向原曲线最内节点约 .402–.430 mm，以下缺失，不把偶发20 µm rod 观察转成所有动物的硬圆。预算不足拒绝并要求小 ROI，density scale 固定1。

`henle.py` 提供 Schein apply/coverage/inverse：输入 IS retinal mm，输出 OPL pedicle mm，RF 只在显式 `rf_equals_inner_segment_mm_v1` 注册下进入。+x 鼻侧、+y 上方；来源为 M. fascicularis M1 右眼作者累计密度推导曲线再数字化，343 点不是343细胞。.01 mm 横轴/.005 mm 纵轴实用读图误差不是生物 CI。径向线性、相邻 cardinal 角向插值属于项目规则；每个相邻方向都须支持，不补原点/尾部，不对离散点再乘 Jacobian。正文归零阈值与独立 histology 保持分别登记。

### HC 和有效内层的真实接入

`atlas_hc.py` / `atlas_hc_candidate_v1.json` 核对新包原始 density rows 后复用总密度+pH1。中央0–.5 mm 的 H1/H2 8.04/12.22 µm 等面积盘仅在连接预览中显式选择，标 `EXTRAPOLATED_UNCALIBRATED`；旧运行默认不变。外围 Wässle p599 的 temporal 5–6 mm、H1约18/H2 16–22 contacts、约6000 cones/mm² 保持区间/近似来源语义。H2中点19及 OPL density 转换是项目选择；只在正且支持的锚点用 sqrt(C/(π rho))，再用正值 log-PCHIP 在1/5/6项目节点间插值到1–6 mm。节点端点 stationarity、跨方向迁移和 filled disk 都是项目近似，不称实测枝形，不在任意查询点逆密度扩半径。

`atlas_connections.py` 默认已恢复 **generate_joint_hc→冻结canonical master→JointHCMosaic.crop**。先前 fixed32µm marked-Poisson HC 绕开了已批准的同型regularity/异型pair-energy，是实施衔接错误；保留density profile名称不能称保留分布规则。旧Poisson输出及显式 `historical_marked_Poisson_preview_v1` 可追溯，默认不会使用或fallback。原Poisson4-owner-zero结果只能描述旧preview实例，不用于否定已批准joint mosaic或生理来源。

新工程配置 [atlas_hc_joint_master_v2](../configurations/atlas_hc_joint_master_v2.json) 固定物理 x[-.4,.9]×y[-.4,.4]mm。保留原density、pair-energy（a11/a22/a12=4、T=1）、dmin派生有限支持、2 sweeps及原HC seed20261010；cone seed20261012不变。旧成功canonical x[.333329,1.111183]×y[-.497665,.498834] 不覆盖中央依赖，不能强crop复用。按需生成新master只有一次，11319 HC（9055 H1/2264 H2）、22638 proposals，耗时305.416s；未运行ODE。冻结 [master资产](../data/atlas/hc_joint_central_master_v2.json) 3227930bytes，SHA=`8a7a3498d6558cb96d09b3f274b6e37d3db05d01612f6dc86e76a14ccc663a0e`；新cell namespace=`24ab901c761ec74a85088894ecab239a9bbb89e31274c13e41f9f56a37b8d407`。恢复生成器改变HC位置/ID，不冒用旧Poisson ID；同一新master的重叠/请求顺序保持坐标、type、ID。

首阶段已支持的Schein映射与3mm独立参考需要的物理包络加arbor/pair buffer，用候选方框±3.25mm作成本估计约408496 HC、816992 proposals，超过原60000/500000预算，未物化。核心数值阵列按128B/cell粗估52MB、序列化按650B/cell粗估266MB，不含Python索引/对象；没有可靠历史单独生成计时，不能给全域ETA。当前中央master预计11318.72 cells，实际11319；它覆盖r=0/.035/.35代表性HC依赖域，**不覆盖整个RF0–3mm，更不覆盖3–6mm**。r=1/3请求明确 `HC_FIXED_MASTER_UNSUPPORTED`，不当成零HC或来源生物缺失。扩展需新的固定master资产/版本与明确工程预算。

每次HC query还检查0.140502201mm保守pair context在master内；通过后只crop，禁止ROI重采样。HC soma→OPL arbor anchor仍是原具名同位项目转换。现有 finite footprint / morphology narrow phase实际产生唯一候选边；H1未知S保留conditional，动态forward/feedback未开启。反向footprint边界采样只检查cone input envelope，不能证明完整recurrent closure。master/pair覆盖、cone输入source域/ROI覆盖、零伙伴及未物化范围分别记录。

`atlas_inner.py` 在新的 OPL 点上真正复用以下有效规则：已知 L/M 的 midget BC 父 cone 私线、中央 midget RGC 私线及 Wool temporal convergence 的具名 iso-density 迁移；S route 的原 cap8/min1；diffuse BC cap8、local AC cap4、midget inhibitory cap1、widefield AC cap16 和40 µm有限支持。Diffuse seeds 用全局 ID hash 的具名1/4工程采样；widefield AC 用旧40µm/3独立计算网格，只有 IPL 坐标。它们不是已测亚型密度。未知 L/M/S 保留结构候选和阻塞原因，不偷偷设成 L/M 或启用权重。BC IPL terminal/RGC IPL arbor 在新 OPL 上同位是项目近似；McGregor 仅在 0<RF r<=.04 mm 提供独立 midget soma 视图，不使用旧15% split。外侧 soma 缺失是坐标/显示限制，现有有效 IPL 动力学本身不要求 soma。

`atlas_rod_inner.py` 复用既有有效 rod→RBC→AII 子图：RBC rho=2*rhoRod/36=1/18；**36是密度公式基数，实际 rod→RBC selector cap40**，RBC→AII cap3；40 µm hard support，AII独立5000/mm²全局 tiles。全局身份及重叠查询坐标稳定，预算不稀释。Rod RF→OPL spherule、RBC OPL dendrite→IPL terminal、AII IPL 角色分别以旧具名有效同位规则表达，不叫 cone pedicle、不用 cone Henle。中央 rod source 缺时省略 rod/RBC，独立 AII 仍生成；不输出假生物零。上述两条路线已有真实有限归一化边，kinetics 未新增。AII 到 cone BC、parasol 等更大网络装配仍不能由这两条子图宣称完成；parasol 独立 canonical effective seeds 尚未接入，不把 AC/AII 密度借成 RGC 密度。

### HC gap：可用接口与不能猜的参数

`hc_gap.py` 实现 supplied symmetric effective-native k [ms^-1]；需明确 equal-effective-capacity/common state-scale/reference，添加 sum k(hj-hi) 到 dh/dt，换成秒仅乘1000一次，**不除 pooled_aI*tau_h**。默认关闭，无正值校准默认；connected components 保留孤立/无 cone 输入 HC。另一条 SI 路径必须显式 supplied g [S]、C [F] 与 V=V0+s*h 标定，按 I/C 转导数。等 effective 容量的 k 路线保留 sum h；异 C/增益的物理路线检查 sum C V，不能检查裸 h。均不行归一化或叠加第二套空间 kernel。

原生 vH 相对尺度不等于已标定 mV。源 tau_h generic20ms/fit20–35ms 是时间参数，不是电容或 native→V 标定。Packer/Dacey2005 Table1 打印 H1–H1 resistance=1.00E−07 Ω 的矛盾数值，不能改成100nS默认；其15µm soma/120µm field/−45mV resting是外围H1作者模型假设，不迁移为中央/H2测量。

正 gap 若扩入无直接 cone 输入 HC，现有 `vh2005.py` 的 A 行非空/unit-DC合同及 pooled_aI=((A@Vbar)/V_I)^mu 尚不能支持它们：A空行使 pooled_aI=0，z2与h的化学/适应分母未定义。需要明确这些细胞的 transmitter-drive、adaptation reference/time rule 和 coupled initializer；不得以冻结状态、dummy input或默认 aI=1 掩盖。k 正值也未标定。例如两相同 HC 的 gap-only 差模时间常数1/(2k)，k=.001/.01 ms^-1分别500/50ms；这只是工程候选的数学尺度，当前未选择或运行。

### 当前联合HC连接预览和最小修复回归

[恢复后六区图](../reports/atlas_connections/20261012T060257574819Z/connections.png)、[完整graph](../reports/atlas_connections/20261012T060257574819Z/connections.json)、[范围与阻塞](../reports/atlas_connections/20261012T060257574819Z/manifest.json) 是当前默认joint路径。graph=39801054bytes，SHA=`e25f7ca5212f527dde9a4237da646eac674a9261c8d4c38479489a23572e6848`。PNG仅限制绘图为前600稳定排序边，全部图边保留。旧 `20261012T052905745304Z` 等Poisson导出不删除，明确作为独立历史preview；其110项测试没有验证保留joint分布规则。

| RF中心/mm | owner/OPL | joint HC query数 | HC候选/conditional | owner零候选 | cone内层边 | rod内层边 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 14/0 | 144 | 212/8（halo） | 不可计数，缺坐标 | 9376（halo） | 0（rod source缺） |
| .035 N | 4/4 | 11 | 7/0（全halo） | 4 | 1198 | 0（rod source缺） |
| .35 N | 27/27 | 1044 | 3669/2333 | 0 | 0（类型阻塞） | 0（rod source缺） |
| 1 N | 14/14 | 未物化 | 不可计算 | 不可计数 | 0（类型阻塞） | 1526 |
| 1 S | 10/10 | 未物化 | 不可计算 | 不可计数 | 0（类型阻塞） | 1975 |
| 3 N | 4/0 | 未物化 | 不可计算 | 不可计数 | 0（坐标阻塞） | 2291 |

r=.035的4个cone保留原RF/OPL和globalID。最近H1距离依次27.767/22.133/21.121/23.547µm，最近H2为42.022/41.831/36.292/38.085µm，均超过原8.04/12.22µm盘；无owner边。该查询master/HC anchor/pair context完整；11个HC的cone input envelope仍0个完整：4个触及Schein中央mapped缺段，7个越过RF query。全部7条边是同一H2到halo，距离.957–10.061µm。**这是冻结的有限joint实例结果，不是生理资料/群体分布失效证据。** 0/.35案例的完整HC输入包络分别12/376个；仍有外层halo缺额。

rod几何与实例没有改变：1 N的owner RBC1/AII2、1 S的RBC1/AII1、3 N的RBC3/AII3输入包络完整，degree40/3；不由此宣称完整rod/cone网络。当前新增修复 [10项回归](../reports/atlas_joint_master_repair_20261012.xml) 通过（3.13s）：联合设置/seed保持、冻结资产读取、真实crop重叠/请求序ID、pair越界和预算拒绝、旧Poisson必须显式、新S候选inactive，以及原有限连接/内层/rod案例。只有一次master物化；未重复joint生成、背景或统计实验。

**需下一步明确的科学合同**：

- RF→OPL cone：中心示例 owner r=.001011–.005073 mm 的14 cells没有 .025 mm 内曲线；3 N示例 r=2.993348–3.003148 mm 的4 cells超过 N/T2.475 mm支持。S/I上限分别1.675/1.950 mm。原点/中央桥接、尾段/归零若要加入需具名项目规则；不拿旧内圈或文字阈值暗补。上述18个缺坐标owner不计为零接触。
- OPL cone→H1/H2：恢复joint后.035 N的4 owner仍在此冻结有限实例内零伙伴，标记仅为该配置的运行缺口，不是否定来源或joint人口规则。不得放大R、nearest补满或把零适配器当中央完成；还需具名中央footprint/位移及完整输入支持合同。
- RF/OPL L/M/S：.35 N的621 query cells范围 .300385–.402558 mm；1 N .950319–1.051027，1 S .950571–1.050258；主非零S spatial partition仍缺。2.2mm单field方向未知，不能给这些区或全域补曲线。由此H1条件边和内层typed权重未开启，不代表生理无内层连接。
- Rod/RBC：约 .4 mm以内缺支持的细密度曲线，中央/.35示例明确阻塞；AII独立网格可计算。1/3mm rod子图已有可复用有限边，仍是既有有效近似、不是实测RBC/AII亚型密度。
- HC正gap：需指定 k 或独立 C/g/V 标定，以及无cone-HC的化学/适应初始化规则。.35 N的新1044 HC中476没有直接cone候选；.035 N的11 HC中10个无直接输入；1/3mm尚未物化。它们一旦属于正gap连通分量应有动态状态，不能继续geometry-only过滤。边界外耦合未知，不能以本地守恒证明外部闭合。

**恢复前的110项必要回归通过（仅旧preview接口，不能认证joint分布保留）**：[最终XML](../reports/atlas_connected_interfaces_final_20261012.xml)。覆盖实际source类型/域/冻结身份、全局重叠/查询顺序、五坐标角色、中央/尾段source gap与零连接区分、预算拒绝、有限HC/内层/rod边、gap单位/双向守恒/异电容电荷守恒以及旧原生组件0.2ms DOP853零gap轨迹逐元素等价。测试默认临时目录在sandbox内不可写的一次fixture失败保留，指定授权可写临时目录后整组110通过；不是科学模型失败或新模拟。较早67项及106项XML和所有历史图保留。

GUI新增“来源 atlas 几何预览”和“atlas 连接预览”，沿用 owner/size/halo，不需要RGB，不改当前模拟state。实际GUI按钮链已检查，.35mm导出记录为 `reports/atlas_connections/20261012T051215495146Z`；恢复后同一backend六区导出包含rod子图、联合HC master与越域拒绝；GUI短状态明确“联合HC固定master”，不再默认Poisson。CLI为 `tools/run_atlas_geometry.py` / `tools/run_atlas_connections.py`。PIL复用现有绘图器，未安装Matplotlib/升级环境。当前连接图更新Library `libfile_97c3fc1bcf108191ad3562aee81e3d85` version1，version0旧Poisson图保留历史，ID/version和本机字节数读回一致。

原成功 `retina_contact_geometry_effective_v1.json` SHA仍为 `ab63ff04001615a6a1e48bdf8ceb9af5da01596361dc273ab735d39d3db9f893`，`vh2005.py`仍为 `cfc4ce4f9eb1015755cd790b180faaec83d2a9a7b8da083f3a471a00d36a4900`。本轮只有几何/单位/原组件极短零gap回归，没有新atlas全网积分、长背景、研究对照、SC/DeepGaze或训练。全部六区 `network_run_ready=False` 有具体阻塞；共享S/H2 kinetics、BC/AC/RGC/rod有效参数与finite-open exterior recurrent closure仍未标定/证实。

### 06:13 已批准三项研究近似的实际实施（2026-10-12）

用户于06:13:15 UTC明确“允许”（`Sentinel_88b5e01aafc48191986a5e49b70b5982`）。此前稿中的待确认措辞是历史状态；本节记录已实施合同，保持原严格来源与旧成功运行profile。

`henle_bridge.py` / `Schein1988_M1_bounded_bridge_v1.json` 采用P=r+Δ的C0分段线性桥。P0=.15mm是作者pedicle-free模型内界应用，不是原点单cone测量；0<r<.025连接四方向P(.025)=.166/.169/.165/.170mm，斜率.64/.76/.60/.80。尾部从原最后点连接作者推导归零N3.1/T2.6/S1.7/I2.0mm，斜率.8448/.672/.4/.8；之后identity至6mm，均标project_approximation。原343数字化节点保持不变，histology阈值独立保留。精确r0仅给radial limit / DIRECTION_UNDEFINED，2D点必须显式角策略；普通点方向、数量和RF ID保持，完整provider/cache身份改变。未宣称C1、实测导数或同一动物的全域解剖校准。

批准桥接下，HC输入支持使用 `inverse_support_bounds` 的保守区间包络：各cardinal逆函数分段仿射，r/P极值取节点/区间端点，角向逆处于cardinal界内，再作坐标区间乘积。P<.15的作者模型pedicle-free区域明确排除；不是未知数据填零。对原声明query取得固定HC cohort后，只扩其上游cone查询，不因support-only cones无限再扩HC；其反馈是finite-induced边界，recurrent exterior closure未证明。严格版仍保留原来源拒绝及sampled诊断。

`atlas_s_partition.py` / `deMonasterio1985_mulatta_effective_S_v1.json` 复用完整既有deMonasterio source：2–60°N/T发表拟合、原.15°方形assay区域、.75°/1.5°峰与2°fit的原PCHIP桥、N/T cos权重、上下扩展、223µm/deg面积换算，pS>1直接冲突拒绝，无clamp。原balanced conditional assignment在固定全局.5mm label domains和具名macro-keyed seeds上执行；空间点来自同一32µm receptor atlas，ROI/请求顺序不重标重叠细胞。该globalization是工程近似，不是测量。整套为mulatta-effective/cross-species，未叠Martin50µm圆孔或Godat绿线，没有强迫局部ROI出现S或每种type。主nemestrina strict仍缺完整非零曲线，Godat候选inactive保持。

`hc_passive.py` / `orphan_hc_background_referenced_passive_v1.json` 已实现optional RuntimeComponent扩展。只有positive同型gap依赖内、完整输入查询确认eligible empty的HC才可进入；caller必须提供同graph/background/context且标CONVERGED的gap-off prepared state。vRef/hRef取该分量内direct HC背景pooled Vbar/h算术均值，aRef=(vRef/19.7)^.733，uRef=hRef；orphan z2/h初值hRef，dz2=1000(uRef-z2)/(4aRef)，dh=1000(z2-h)/(20aRef)+gap。A空/B空保留，tonic reference不算cone边；direct HC原动态aI及RHS保持。未requested的无anchor分量不实例化，requested无anchor拒绝外借reference；nonuniform背景、heterotypic orphan和不完整query拒绝。default k0返回原component对象、不添状态；没有positive calibrated默认，物理C/g/V路线不冒接。ref/geometry/background变更改变model/cache hash，正k后必须重新准备背景，初值不叫coupled equilibrium。只对gap项断言守恒；reference泄漏不守恒。

[一次研究配置输出](../reports/atlas_connections/20261012T065905260400Z/connections.json) / [图](../reports/atlas_connections/20261012T065905260400Z/connections.png) / [manifest](../reports/atlas_connections/20261012T065905260400Z/manifest.json)：r0原14owner现在14映射、144HC、836边，12owner零伙伴；.035原4owner不重抽，RF查询扩至[0,.0738163183]×[-.0254980123,.0204832711]mm，共490cone（L248/M239/S3）、11HC，11/11输入包络完整且全有输入，78边、11624内层边，4owner仍无HC伙伴，因此未实例化被动HC或改footprint/density。.35为27owner、2226cone（S140）、1044HC、7510边、38509内层边；无未知type。1/3mm仍HC master越域，未新生成大资产。3mm的cone映射来自明确尾段近似，不能称新原始曲线。图标题只从同一已存graph重绘纠正，未再查询/积分；Library同一ID当前version3，旧版本保留。

本轮38项必要回归通过（43.49s），[XML](../reports/atlas_approved_approximation_final_20261012.xml)：桥节点/源段保持、单调逆、6mm边界、origin策略、保守包络、RF身份、实际上游补全、整套S密度/labels重叠、passive无零基准sink、direct动态aI保持、gap守恒、缺reference/非均匀/heterotypic拒绝，以及原组件0.2ms零gap轨迹。没有新的master生成、长背景、研究对照或训练；没有新的positive-gap真实网络适应。GUI新增一个atlas研究配置选择（严格来源 / Schein桥接·nemestrina严格 / Schein桥接·mulatta有效），作用于atlas几何/连接导出；Run旧成功网络入口不被默换。CLI选择 `tools/run_atlas_connections.py --variant bridge_mulatta_effective`，默认strict保持。新atlas全网装配/积分仍未运行，原全网络baseline可运行；中央joint owner零反馈、中央rod缺密度、部分内层sampling/source及局部master范围继续明确，不能把这些近似叫全生理覆盖。

### 同图动态装配与单一最终配置目标（2026-10-12，最新状态）

以上“尚未装配”和GUI多配置选择描述是此前状态。现在 [统一平台配置](../configurations/retina_platform_v1.json) 由GUI、[公共CLI](../tools/run_retina.py)及`SimulationSession(None, root)`共同读取。界面不再提供strict/research及空间profile切换；atlas预览读取该统一配置中的已批准桥接/整套mulatta有效S来源。历史配置不删除。完整替换仍`BLOCKED`，故唯一活动Run保持旧成功`retina_contact_geometry_effective_v1`；不会把部分atlas候选默换为完整默认。

[atlas_runtime.py](../retina_sim/atlas_runtime.py)把真实atlas global IDs、RF/OPL/IPL坐标、actual contacts及source身份绑定至原有动态组件，没有再次采样、密度缩放或改ρ/R。默认gap0，不演示被动HC。运行视图是目标内层祖先加整个cone–HC连通分量，原有限图其余组织继续归档，不称整体组织抽稀。原r=.35有限图为2226cone/1044HC；所需动态外层为1465cone（L668/M698/S99）和1041直接输入HC。其余761 support-only cone没有作为目标祖先实例化，3个eligible empty HC在gap0保持geometry-only；不把它们的未知外部反馈做零证书，也不声称全有限query/无限外部闭合。

一次 [短Run](../reports/atlas_runtime_runs/20261012T081107797528Z_6addeac9/summary.json) 实际完成1.2ms、0.1ms采样、13组件/12层、26 ON+26 OFF owner midget RGC；总wall2.386s，积分wall.213s，12次solver/204 RHS。零prerun，无新长背景/稳态声明。BC使用原`ResidualBC_ON/OFF`因果核和原bias；需要AII glycine的`HistoryBipolarTerminal`物理terminal没有实例化，没有伪造零glycine或native voltage gain。初始化是声明的outer equilibrium及inner conditional/transient状态，RGB仍未光谱校准。此前一次因interpolation声明不匹配在进入积分前失败，错误目录保留；修正为显式linear后才完成这一次短积分。

该 [新响应图](../reports/atlas_runtime_verification/20261012_final/atlas_runtime_response_readable.png) 来自上述已存13组件Run，不是原连接示意图，也不是后续19组件candidate的轨迹。CSV/PNG/bundle/GUI读回直接使用已存窗口，没有再积分。[统一核查记录](../reports/atlas_runtime_verification/20261012_final/verification.json)验证13个绝对时间点、有限值、IDs/行数/单位/坐标、checkpoint变化、12层可选、CSV逐值一致及导出checkpoint字节一致。

随后可落实的完整**owner cone typed候选**由`atlas_typed_candidate_network`装配：[19组件记录](../reports/atlas_runtime_verification/20261012_final/typed_assembly.json)。包含1465cone/1041HC、midget BC/RGC各ON26/OFF26、diffuse BC ON27/OFF18、S-ON BC6、small-bistratified owner1、local feedforward/crossover AC各27及owner widefield AC3。small-bistratified的S-ON、diffuse-OFF excitatory及local-feedforward inhibitory三条现有路由均绑定。新增两个缺失S接触适配，40µm硬support、原degree8及原Gaussian law/gain保持。

修复确定的typed选择BUG：已知`BLOCKED_DIFFUSE_SEED_IS_S`占位节点应在选择邻居前排除；此前先选邻居再阻塞整行，导致1/3 owner wide AC无有效驱动。`atlas_inner.pool_contact`及冻结图completion adapter只排除这一已知不合法类型，其他unknown/incomplete上游仍阻塞；不扩半径、degree或重抽细胞。历史graph与13组件Run不覆盖改写，新候选记录明确其补充/修正关系。wide AC的owner筛选RF是既有IPL点经已批准Schein逆及注册框架得到的项目坐标视图，没有新增测量soma/RF。19组件仅做build、初始化及一次RHS单位/形状/有限值检查，**未再次积分**，不拿13组件图作为19组件成功响应。

| 未完成路径/范围 | 已确认原因及类别 | 当前处理 |
|---|---|---|
| rod/RBC及AII driver | **数据缺口**：实际错误为`Packer rod curve unsupported at r=0.27340812mm; no extrapolation`；约.4mm以内无原始细曲线 | 拒绝外推；AII几何保留，未造静息driver |
| legacy effective rod整体转移 | **批准缺口**：条件审批尚未同意 | 未启用，不由“单一配置”偏好推导授权 |
| parasol ON/OFF | **工程耦合+seed来源未绑定**：旧脚手架借用RBC/rod生成位置，但parasol直接输入是diffuse cone BC和widefield AC，不能把工程依赖称为必需rod生理驱动 | 显式RF seed接口已解开位置耦合；新atlas仍缺获批seed来源，未默认迁入旧rod-derived人口 |
| BC physical terminal/OFF glycine | **驱动/路线缺口**：该原机制强制AII glycine输入 | 现有causal residual cone路由可用，物理terminal仍未装配 |
| r=.035四owner HC反馈 | **有限实例适用性限制**，不是source总体失败证据 | `JOINT_PROFILE_UNSUPPORTED`保持，不改圆盘/ρ/R、不nearest补满 |
| RF0–3整体HC组织 | **工程域/预算限制**：当前master只覆盖局部 | 只估算，不新生成大master |

Schein1988 p492所述极中央H1/H2树突场周向拉长、外侧更圆作为来源讨论记录；现有8.04/12.22µm是0.5–1mm峰区的面积等效推算，不是极中央直接测量。本次没有新增轴比、2:1椭圆扫描或调整圆盘。

全RF0–3第一阶段HC master保守物理square可取±3.25mm（包含桥接位移、31.75µm最大arbor及140.50µm pair context）；原density模型Gauss64积分约408,496 HC，2sweeps约816,992 proposals。至少51,062个target_mass8 partitions，超过现有8192；也超过60,000cells/500,000proposals。仅核心数组约52MB；按局部3.23MB/11,319cells比例JSON约116MB，保守复杂记录约270MB，Python索引/分区/quad缓存另计，未测实际峰值。局部11,319cells物化305.4s，36倍proposal工作量的粗线性参考约3小时，但分区/局部密度/接受率变化使其不是ETA；当前预算会先拒绝。没有启动整体生成。

[14项必要回归](../reports/atlas_runtime_regression_20261012.xml)通过（5.38s）：原实际IDs/typed权重/private-line、S/H1排除、冻结graph篡改拒绝、中央zero与预算拒绝、单一默认/无长background、新S三路及wideAC修复，并覆盖现有atlas inner适配。原`vh2005.py`、旧成功profile和HC master SHA保持此前值。完整typed全网及生理标定仍未完成；所有可落实工程装配已保留，下一步实际阻塞为来源/批准与域预算，不以midget子图称主线完成。

### Parasol RF seed接口解耦（2026-10-12）

`ipl_geometry.ParasolRFSeedInput`显式携带mm/functional_visual_field坐标、原site IDs和source_id/locator/origin/evidence_class；`typed_layouts(..., parasol_rf_seeds=...)`可在`scaffold=None`时直接接受这些位置，不调用rod/RBC provider。不改变输入顺序/ID、displacement、一次冻结centroid、40µm support、degree、权重或总gain。缺seed和scaffold时明确`PARASOL_RF_SEEDS_UNBOUND`，不自动采样。`full_target_network(..., parasol_rf_seeds=...)`透传该接口；未提供时仍走原scaffold路径，提供时只跳过用于AC/RGC位置的额外scaffold生成，实际rod/RBC/AII/terminal支路全部保留。

从旧成功运行保存的RF/site IDs提取 [14个回归seed备份](../data/atlas/legacy_parasol_RF_seeds_regression_v1.json)，SHA `31d7134e353f2c57922a3e714b8d4fe4d53901fcdd22e2565c047208194393fe`。来源manifest SHA `d33225e79acf3e5c1c06fc27f91bb8a4c0eac1c6d3f191406b4576d44653acc9`；origin明确写旧rod-derived RBC scaffold、regression-only、非独立生理density、默认新atlas不采纳。没有重生成seed或迁入legacy rod field。

一次 [最低必要等价检查](../reports/parasol_seed_interface_20261012.json)通过（9.85s）：原scaffold与显式相同seeds的五类RGC和三类AC全部坐标/IDs一致；parasol centroid记录还与旧保存结果逐值一致；ON/OFF的excitatory/inhibitory四个CSR矩阵及degree诊断完全一致。没有新ODE、背景、master生成或全套重复测试。该检查验证输入接口几何等价，不冒称全网轨迹重验。

当前统一文件仍只是**入口统一，底层图尚未统一**：日常GUI/公共CLI用旧活动Run，atlas预览/候选用新图。统一配置的`parasol_RF_seed_source.status=UNBOUND`、artifact=null；旧seed备份不作为新默认来源。最终单一模型尚未完成。legacy rod指数场、20µm硬圆边界及177000方向解释均未获来源校验；此接口改动不批准或修饰这些生理假设。

### 批准人口修订的可运行交付（2026-10-12，当前状态）

用户09:51 UTC“是”批准 `Sentinel_28318027a0ec8191a8ead57b6439c9c1` 的新人口合同，并授权根据实际局部依赖选择工程master域和seed。此轮落实该合同，没有采用旧指数rod场、20µm硬hole、rod-derived parasol或强造owner。本节取代上节“尚未统一”的当前状态。

**密度和来源。** [population_revision.py](../retina_sim/population_revision.py)读取冻结的Packer1989原数字化资料：各meridian从原点到实际首点使用 `rho1*(r/r1)^2`，原点严格0、任意正的小r可为正；首点以外采用原density节点上的非log PCHIP，禁止尾部外推。相邻轴以cos²/sin²凸组合，+x nasal/+y superior。原节点及r/ρ读图误差保留；首点仅C0连接。中央桥、方向融合及解剖密度到此RF框架的注册是已批准项目近似，不转称中央测量。

Parasol使用用户批准的固定圆整式 `D(r)=580000*exp(-4.2*r)+56000*exp(-.56*r)`，r为平面mm，ON/OFF各 `.04*D`，合计`.08*D`，函数域0–3mm。这是[Vanni2026 preprint](https://www.biorxiv.org/content/10.64898/2026.03.09.710551v1)的作者cone-derived RF sampling模型移用，非soma实测计数、非本项目RGC总数的8%。[作者commit 08aea380](https://github.com/VisualNeuroscience-UH/macaqueretina/tree/08aea38003db41c4753990038ed8ac9a75b2706a)小文件快照保存在 [source_snapshots](../data/source_snapshots/vanni2026_08aea380/frozen_source_manifest.json)，逐项核对Git blob SHA1和SHA256，没有执行作者代码、拟合或大下载。圆整系数来自获批引文合同，非YAML字面常量；作者构造器会重新fit密度，本项目没有重fit。作者YAML原polarity为.4/.6，本轮.5/.5及temporal向全角度迁移明确为项目假设，preprint并非独立验证的动物事实。

**生成器恢复。** [equal_mass_master.py](../retina_sim/equal_mass_master.py)调用已有`spatial_profiles.generate_density`：积分质量四舍五入配额、equal-mass初始化、原jitter、一次原capacity update。它是近似离散质量算法，不声称精确Laguerre或复现作者点统计。每个人口/极性拥有独立稳定namespace与seed；固定矩形master只生成一次，后续按global ID裁切。没有Poisson替换、ROI重抽或预算密度缩减。RBC=批准rod密度/18、AII=5000/mm²仍是原有效项目数值，不是新亚型测量。

| master | 实际固定域 [x0,x1,y0,y1] / mm | 数量 | seed |
|---|---|---:|---:|
| rod | [.22,.7792391412,-.1629142093,.1525663191] | 4208 | 20261011 |
| RBC | [.26,.7392391412,-.1229142093,.1125663191] | 148 | 20261012 |
| AII | [.30,.6992391412,-.0829142093,.0725663191] | 310 | 20261013 |
| parasol ON | [.2574667396,.4707715697,-.0815924649,.0864166425] | 249 | 20261014 |
| parasol OFF | 同上，独立master/namespace | 249 | 20261015 |

五份文件在 [population_masters](../data/atlas/population_masters)，完整精度域、recipe、来源、配额、ID及冻结hash均进入最终图`population_revision.masters`。域依据此次RF/实际BC IPL祖先40µm依赖推导，边界finite-open、无边界补偿；未生成RF0–3整体master。Parasol裁切至实际seed cohort保留ON73/OFF72，目标owner自然产生ON2/OFF2，不保证或强造每个ROI必有owner。

**同图装配和唯一入口。** [population_revision_graph.py](../retina_sim/population_revision_graph.py)保存 [population_revised_r035_v1.json](../data/atlas/population_revised_r035_v1.json)，SHA256 `af456b72ce087c372c29f30a0c74198d2602d1bb3db32211eccc8691c534dd6b`。HC master、圆盘、density和gap0保持原样。Parasol用实际合法diffuse BC冻结centroid，原40µm硬support、degree/gain/重叠及动态参数保留；它的位置不再借用rod/RBC。

[atlas_population_network.py](../retina_sim/atlas_population_network.py)是唯一生产装配器，先验证冻结图SHA，owner限定[.34,.36]×[-.01,.01]mm及halo.04；越域直接拒绝，未回退旧图。它经[atlas_primary_rod.py](../retina_sim/atlas_primary_rod.py)恢复实际rod→RBC→AII、原物理BC terminal、AMPA/glycine及AII双向gap。Rod OPL位置不应用锥Henle；`OPL_spherule/OPL_dendrite`明确存于typed_points和运行OPL端口，未假称cone pedicle。原Chen rod参数、BC/AC核、TRPM1/C/g/电导系数没有新拟合；条件初值仅根据实际连接的声明背景导出，不宣称耦合稳态。未启用HC正gap、被动孤儿演示或外围Chen LM向中央的未校准blend。

实际动态组织为cone1465/HC1041；midget BC及RGC ON/OFF各26；diffuse BC ON42/OFF20；S-ON BC6；local feedforward33/crossover31/wideAC10；parasol RGC ON2/OFF2；small-bistratified1；rod4208/RBC148/AII310。5920个独立rod–RBC接点、929 AMPA、78 midget-OFF glycine和60 diffuse-OFF glycine接点执行。人口/电流代理/接点总表示数18618不能当作独立生物细胞计数。原几何图2226cone/1044HC继续保存；761外侧非目标祖先cone和3个gap0空直接输入HC未运行，不声称外部递归闭合或赋零证书。

[统一配置](../configurations/retina_platform_v1.json)的默认已激活`atlas_population_revision_v1`。`run_retina_gui.cmd`、[run_retina.py](../tools/run_retina.py)、[run_atlas_runtime.py](../tools/run_atlas_runtime.py)共用该RunConfig和装配器；[GUI/CLI预览](../retina_platform/population_preview.py)读取同一冻结文件，不调用旧生成器。实际当前默认RunConfig逐字段等于已完成短Run的配置。GUI只有一个禁用的模型选择值；旧模型配置不可通过加载配置暗换，历史结果仍可加载。两个预览按钮只显示该冻结图；`run_atlas_geometry.py`和`run_atlas_connections.py`同样如此。旧CLI源码备份在`tools/history/population_revision_20261012`，旧原生profile、HC master、历史图/来源/运行结果继续保留。

**最少验收和实际输出。** [4项针对性回归](../reports/population_revision_regression_20261012.xml)全部通过：rod节点/原点/中央连接/域、角融合及误差保留、独立parasol公式/域、五master配额/裁切重叠ID/独立namespace。仅完成一次新 [实际短Run](../reports/population_revision_runs/20261012T104015129805Z_2f5499d9/summary.json)：1.2ms、0 prerun、0.1ms采样；wall8.910s、积分wall.585s、13次solver/221 RHS。无缺失机制层，无状态重置；最后checkpoint SHA `ba5b04dce01dc9daddd3a8308e0620d96889399e94c08ca465c351ecacf096e3`。构建时保存的`ASSEMBLED_PENDING_SHORT_RUN`是历史装配快照；最终完成状态以summary和下述acceptance为准，原manifest未回写。

| 对象 | 组件/层 | 实际状态块/标量 | 积分证据 |
|---|---:|---:|---|
| 旧owner cone typed候选 | 19/15 | 45/14495 | 仅构建/初始化/RHS，无新ODE |
| 本轮完整局部有效图 | 34/20 | 77/53759 | 上述一次1.2ms真实Run |

[最终读回核查](../reports/population_revision_acceptance_20261012/verification.json)复用该Run，没有再积分：13个绝对时间点、全部有限端口、ID/行数/单位/坐标、变化的checkpoint、GUI20层和五类RGC、CSV逐值相等、PNG、结果包导出及checkpoint字节相等均通过。[五类RGC最终图](../reports/population_revision_acceptance_20261012/population_runtime_response.png)及[冻结人口图](../reports/population_revision_runs/20261012T104015129805Z_2f5499d9/population_preview/population.png)均已视觉检查。Library新响应图为`libfile_8a26afc1a1a08191917c6508ab660b1c` version0（82709bytes），本机ID/version metadata读回一致，不覆盖旧13组件图片。RGC曲线在此极短窗内近乎平直：图仅证明保存输出，不证明完整刺激传播/对比敏感性；native延迟2.82ms也长于本次窗。没有据此宣称适应稳态、随机统计或响应生理验证。

**明确剩余缺额和科学边界。** 所有148个RBC均实际接40个rod、无缺额；OFF BC glycine全部3接点，ON/S BC nearest AII均在40µm内。RBC→AII有且只有一行容量不足：AII index41，ID`atlas:26ada398def795349858257ee7476b91f951719a4ab3be1c168b5707775d7301`，IPL=(.352822915549,.005719134199)mm，在40µm内仅2个候选、目标3，最远35.30794µm。其他309行均3。929条真实边原归一化law保持，并没有扩大半径、重抽、补边或另调gain；这是明确的有效网络容量缺额，不能称所有接触配额满足或真实接线已验证。共享配置状态为`LOCAL_EFFECTIVE_RUN_PASS_WITH_CAPACITY_DEFICIT`，该限制和所有row诊断纳入acceptance/后续装配audit。

r=.035四owner的HC零反馈有限实例仍未解决；局部HC master不代表全RF0–3，未调整arbor/ρ/R。S/H2共享动力学、BC/AC核、五RGC参数、RBC/AII密度和大部分物理terminal/gap/release仍为明确项目有效近似；作者引用、冻结身份和自动检查不替代生理校准。既有独立source reference不删除，measurement所需primary verification不由本次人口批准推导。没有新长背景、研究对照、训练、SC/DeepGaze、环境升级或整体master生成。

### 获批的同图背景准备与亮暗阶跃（2026-10-12，当前状态）

用户`Sentinel_6a4313c8bed48191ba4aaec294527b8a`“可以”明确批准接通新34组件图的原收敛监测、完整磁盘状态保存/恢复，以及从同一准备状态分叉的最小亮暗阶跃。上一节“没有新长背景”描述本轮批准前状态；本节仅记录这一次获批准备和两个阶跃，没有扩HC、改变生理参数、调gain/阈值或做扫参。

[background_checkpoint.py](../retina_sim/background_checkpoint.py)为GUI/CLI共用的完整保存/恢复层。[原monitor](../retina_sim/background.py)保留0.1s窗口、连续3完整窗、minimum1s、5×已登记最大τ、2×最大延迟以及原unit floor/atol/rtol，不放宽。新图静态覆盖为77/77状态块、53759标量，最大已登记τ=.25s，minimum_required=1.25s；未声明此τ列表界定全部耦合慢模态。检查原RHS、窗口最大波动、纯离散一步缺陷、有限/状态域、gap电流守恒、队列时间和历史相位。生成spike的RNG/历史独立前进，随机统计稳态仍`STATISTICS_INSUFFICIENT`。

磁盘envelope同时绑定完整实际population/坐标/ID、CSR连接拓扑及权重、gap电导、component/parameter/model/state schema、clock、solver、背景、原监测尺度及冷初值参考；不是计数hash。完整payload及独立aux hash包含绝对时间、tick、RNG、spike/history/observation/link delay队列和输入phase。原`HybridRunner.restore`再校验动态身份/形状。普通/旧图Run checkpoint不能冒充该背景envelope；身份失配拒绝，无自动fresh/旧图回退。准备状态必须同时有匹配阈值/尺度、全部状态覆盖、原连续完整窗口和最低时间证据。部分准备恢复保留绝对时间及原冷参考尺度，但重新要求3个完整窗口；未把中断半窗算成通过。

[最小恢复验证](../reports/population_background_verification_20261012/verification.json)通过：实际新图在1ms边界磁盘保存，再到3ms的完整checkpoint与不经过序列化的同边界连续运行逐字节相等，包括RNG/历史/delay/tick；同计数下实际edge weight变化、背景变化、solver变化、aux变化、普通旧checkpoint及未收敛缓存共6种情况拒绝。这是必要的极短接口验证，非重复背景实验。

GUI/公共CLI仍使用[同一默认配置](../configurations/retina_platform_v1.json)，现默认`bounded_convergence`、require_converged=True，并指向 [真实准备envelope](../reports/population_background_runs/20261012_authorized/background_checkpoint.json)。GUI读配置入口保留该路径，公共`run_retina.py --background-checkpoint ...`可显式指定恢复源；相同prepared缓存通过完整核验后可直接继续刺激。完整后台共用`prepare_with_checkpoint`，每45s及完整窗口边界原子保存可续状态。仅改变墙钟/内存预算不改变数值准则；未完成准备不进入刺激。加载亮暗结果时GUI显示实际native td/rod R*/s标量输入，清除未使用的RGB fixture预览，保留当前活动模型与准备路径。

**唯一有界准备已完成。** [运行目录](../reports/population_background_runs/20261012_authorized/request.json)墙钟上限1800s、sim上限10s、观察上限100000，Windows当前/历史峰值working set上限4GiB，系统可用物理内存保留1GiB；在有效状态边界合作检查，非对单次分配的硬OS限额。最初2GiB计划因实际装配瞬时峰2.61GiB改为明确4GiB，没有改收敛阈值。实际 [报告](../reports/population_background_runs/20261012_authorized/background_initialization.json)为CONVERGED/criteria_satisfied：elapsed1.30s、wall1026.3439s（17.11min）、30504 observations、无uncovered states。最后三个0.1s完整窗最大norm分别.0120174/.00416465/.00144326，连续通过计数5/6/7；在最低1.25s之后的完整1.3s窗口才结束。主要慢状态为`AII_AC.x`/original_rhs。RSS约1.42GiB、装配峰2.61GiB，未触发安全停止；不将旧图104min准备或短Run线性外推作为此次预算证明。

从同一完整准备state分叉的 [bright](../reports/population_background_runs/20261012_authorized/bright/summary.json)及[dark](../reports/population_background_runs/20261012_authorized/dark/summary.json)各20ms，wall45.70/46.03s，每支20个1ms窗口，0.1ms采样、201个去重绝对时间点，34组件/20层，窗口间无状态重置。两支初始checkpoint字节完全相同，SHA256 `cd1984aa9207c3547b35829ea1fe1dd3ddb27ebffeb1e254678bb134ccc0ed27`。实际输入为native cone100→110/90td及rod5→5.5/4.5R*/s；当前中央图没有`physical_cone_light`，配置中的cone R*/s参考1000/1100/900未驱动一个外围physical cone端口。两类实际单位仍是独立consumer声明，非共同光谱/显示校准场。只改变输入值，生理参数、AII max3规则、40µm支持、HC gap0和4zero局限保持。

**实际早期响应，非最终峰值。** 以下为所有被记录RGC的20ms末端ΔV范围（相对同一保存起点），单位µV；所有主ON/OFF owner都发生同向数值变化。small-bistratified只有1个owner，弱反向值不能据此认证生理极性。按当前solver rtol1e-6/atol1e-9，其约−.0473V处误差控制尺度约.0483µV，该弱变化小于此尺度；这不是实际误差上界或统计检验。

| RGC / owner数 | bright ΔV / µV | dark ΔV / µV |
|---|---:|---:|
| midget ON / 26 | +.66436..+.66618 | −.67219..−.67035 |
| midget OFF / 26 | −.67496..−.67119 | +.67825..+.68204 |
| parasol ON / 2 | +.60788..+.60800 | −.61348..−.61336 |
| parasol OFF / 2 | −.67439..−.67419 | +.68127..+.68147 |
| small-bistratified / 1 | −.03658 | +.03777 |

上游保存值同样改变：1465 cone native release末端bright约−.0039495、dark+.0040372（dimensionless）；4208 rod current分别+.000234871/−.000234873pA；rod membrane约−.1250/+.1250µV；148 RBC约+.02423/−.02423µV。310 AII末端变化范围bright约+.000405..+1.156µV、dark−1.169..−.000405µV，包含cone BC双向gap和持续的数值残差，未将每个微小变化都归因为已验证的刺激作用。没有新增无阶跃control或调参对照。

20ms末端主要曲线仍变化，短于登记的慢τ=.25s；能记录早期传播，不能认证最终峰值、稳定幅度、随机stationarity或生理adaptation完成。[最终核查](../reports/population_background_runs/20261012_authorized/postrun_verification.json)复用已存结果：fresh当前装配器严格恢复prepared cache、同一GUI/CLI模型、20层GUI、全部有限端口/IDs/units、201时间点、每支CSV、同起点字节及显式标量输入显示通过，没有再次准备/积分。实际source及参数仍未完成全生理校准，AII2/3未因此消失。

[唯一亮暗对照图](../reports/population_background_runs/20261012_authorized/bright_dark_steps.png)使用保存V的ΔV×1e6转为µV，横轴为距阶跃开始的真实秒数，展示每类recorded cell0；这是单位转换，不是模型gain修改。Library ID `libfile_0630d23dd2508191b417b76bd1fe69c7` version0，58765bytes，本机ID/version读回一致，旧短Run图保留。

并行 [atlas_chunks.py](../retina_sim/atlas_chunks.py) / [接入说明](atlas_chunks_integration_zh.md)已完成六个单人口master的冻结分块读取及等价验证。本轮运行未换reader：生产装配读取的是冻结多人口**连接artifact**，该reader明确不接受该格式，不能直接替换或声称降低整个动态装配/背景内存。转换/分块不会解除全域生成器和HC预算、也不完成RF0–3覆盖；相关后续多人口graph索引接入须保留实际图/缓存语义。当前中央HC新形态假设、全域生理人口与随机/动物验证仍是独立未完成项。

### 多人口连接artifact的生产分块接入（2026-10-12，当前状态）

背景/阶跃已结束后，用户明确要求完成同图分块生产接入，上节“未换reader”描述此前状态。本轮新增 [atlas_graph_storage.py](../retina_sim/atlas_graph_storage.py)，schema `frozen_retina_graph_blocks_v1`：多人口表按原行序每256行保存coordinate blocks，保留完整原ID/source-key索引及RF包络；连接按原边序每512边保存完整edge records，另存无pickle的pre/post端点索引。表/连接header、来源、row diagnostics及其余元数据独立压缩；descriptor和各文件SHA形成单独storage manifest hash。没有采样、改配额、密度、类型、坐标、边/权重/顺序、参数或初值。

[一次性接入工具](../tools/integrate_graph_storage.py)将47,695,349bytes的原 [冻结多人口图](../data/atlas/population_revised_r035_v1.json) 转为 [graph_blocks_v1容器](../data/atlas/graph_blocks_v1/population_revised_r035_v1/manifest.json)。346份payload共14,718,367bytes，加172,407bytes manifest合计14,890,774bytes；原JSON完整保留且SHA仍 `af456b72ce087c372c29f30a0c74198d2602d1bb3db32211eccc8691c534dd6b`。完整逻辑record hash=`26dfe85b99ec3450756d6a9e56681d068ab61d90ce70d09468fb326b51c85a24`；storage manifest hash=`9d53469cb5f0b958823fb782aa9c911593e4bba43a2bf85a7bae019031ff08de`。来源/图身份与存储路径、gzip/npz编码身份分别保存，未把layout变化伪装成新图。

[统一配置](../configurations/retina_platform_v1.json)的`atlas_target.population_revision`原binding保持不变，新增内部`population_revision_storage`绑定容器和两个冻结hash；无新增用户选项。[生产loader](../retina_sim/atlas_population_network.py)现默认打开该容器，校验受信manifest hash及其对原source SHA/logical hash的绑定，各文件读入时再核bytes/SHA；损坏/缺块直接报错，不回退原JSON或重生成。语义许可来自原图，存储编码不成为新的生理来源。

[atlas_runtime](../retina_sim/atlas_runtime.py)先从人口/端点索引读取owner及祖先依赖，闭包保持原规则；再按所需原行/边序读取坐标和连接块，构建相同CSR矩阵。HC连通分量、private parent、S/H1资格、AII2/3及finite-open规则均保持。运行所需rod/RBC/AII全表仍按原机制实例化；必要的历史几何/audit元数据会物化，未暗删其他组织。GUI、`run_retina.py`、`run_atlas_runtime.py`和现有两个preview入口共用该loader；直接Python入口仍为`atlas_population_network(root, ...)`。

[唯一最小核查](../reports/graph_storage_integration_20261012.json)PASS：完整重建record与原JSON的canonical内容hash相同（包括列表行/边顺序、坐标、权重和全部metadata）；实际编译的population、CSR数据/indices/indptr、gap、parameter、component、clock、solver及state schema绑定与已存背景envelope完全相同；从fresh当前生产装配恢复的完整checkpoint逐字节相等，SHA仍 `cd1984aa9207c3547b35829ea1fe1dd3ddb27ebffeb1e254678bb134ccc0ed27`，34组件/77状态块。**没有重新积分、背景准备或阶跃**，原数值背景身份、RNG/history/delay/time保持有效。

该生产核查实际读取240/346份payload、10,775,675压缩bytes，76/104个table blocks及84/133个edge blocks，未打开原图JSON。完整逻辑验证单独读取所有346份，这是一次性验收，不是每次生产启动。当前构建加恢复wall2.781s；该核查进程RSS约180MiB、峰值约546MiB（包含转换/完整逻辑核查，不是孤立编译峰值或严格配对性能试验）。可报告存储读取减少及本次进程实测，不能把它外推为大网内存保证。

**界限**：当前祖先闭包需要逐route读取完整端点索引，原HC–cone连通分量可能扩展到大部分外层；选中的全部机制、CSR、参数、source registry、状态及必要audit仍一次性编译/驻留内存。块JSON会在该reader实例中缓存，不能声称流式ODE或constant-memory大网运行。转换本身仍全量解析原JSON。单人口master reader继续独立保留；全RF0–3生成器/预算、中央HC假设和生理/随机验证均未因此完成。

### 输入数组的等价按需供给（2026-10-12）

[atlas_runtime.py](../retina_sim/atlas_runtime.py) 中声明的短合成 native light 原先在装配后预分配整个 `duration/0.0001 × cones` 正弦数组，即使 GUI/背景路径随后替换该通道也会产生分配。现共享入口调用 `_sampled_native_light_channel`，仅缓存当前相邻两行，保留 `round(duration/0.0001)`、整数索引乘100µs的原网格、端点钳制及同一线性加权计算；单位td、input contract、来源/evidence、background context和适用域保持相同。此输入缓存不携带动力学状态，反向/跳时读取重算对应两行；背景平移、事件时钟、RNG、延迟及状态推进代码未修改。没有增加GUI参数、换生理值、默认配置或实验输入。

[一次有界核验](../reports/native_light_input_check_20261012.json)使用7个位置、4.04ms小输入：351次数值/索引检查逐值相等（最大差0td），含采样点两侧、插值中点、反向与重复查询、末网格之后的钳制和越界拒绝；另14个背景平移及外部分块时间边界检查相等。内存单项用1,465个位置创建11s与1,100s通道，各只查询4次；缓存23,440bytes、phase11,720bytes，`tracemalloc`峰值88,017/87,929bytes。旧11s完整rates数组理论下限1,289,211,720bytes；没有为对比实际分配旧大数组。上述为通道Python/NumPy分配核验，不是整网RSS或耗时改进测量；35份受保护配置/输入状态代码/源资产哈希未变。未重新装配积分或准备背景，不增加全域0–3mm覆盖或生理验证结论。


### 连续计算内核（2026-10-15）

`HybridRunner(compiled, compute_backend='auto')` 默认对已核对的机制和无代数闭环、无连续边延迟的图使用 `numba_flat_rhs_v1`。`compute_backend='python'` 保留原参考路径，`'numba'` 要求实际可编译；auto 对不支持的图在 `result.metadata['compute_backend']` 中记录回退原因。当前34组件、77状态块、53,759标量全部保留。内核由实际绑定参数和CSR连接生成692条集中指令，float64、fastmath=False、非并行且检查边界；已核对工厂文件的固定SHA改变时必须重新审查lowering。每个内核实例的临时缓冲区受锁保护，每次重写状态、输入、运算结果和gap累加器，不跨状态缓存信号。

仅连续RHS迁入编译调用，`rhs_flat_reference`、原观察函数和组件实现继续保留。SciPy求解器、误差容限、100 µs/730 µs时间事件、历史转移、PCG64消费和记录路径未改变。新的运行身份包括内核源码、后端、Numba版本及数值计划；编译耗时不参与身份。旧checkpoint的严格身份保护未放宽。比较脚本在逐项核对参数、连接、布局和时钟后仅作测试用途的状态身份迁移，不修改或重新预热旧背景缓存。

[有限回归](../reports/compiled_kernel_20ms_20261015/summary.json)复用原B-K缓存输入与2.10 s背景状态，一次Python、一次Numba的20 ms窗口，均记录全部188组状态/端口轨迹、21个时点。首次编译2.337 s，随后进程载入编译缓存0.192 s；正常运行8.891→4.483 s（1.983倍），两路均225次求解、3,186次RHS。188组中49组逐字节一致；其余为浮点舍入差异，最大逐轨迹尺度归一化差3.345e-15；315个spike事件、RNG、历史计数和调度时刻完全一致。回归阈值是明示单位尺度下的轨迹差1e-10，不把求解器局部容限解释为全局误差保证。24项集中测试通过。此计时的全量记录比前一轮7.71 s仅记录RGC的口径更宽，不能混用来推算倍数。

本轮不作750 ms外推。编译后3.490 s仍在求解器及RHS边界内，0.992 s在该边界外，包含输入、历史、观察和全量记录；未进一步剖析各自占比。


### ISET cone与midget几何接入（2026-10-15，模型实现）

本轮停止追加性能工作，保留已完成内核。新增`isetbio_geometry.py`、`export_isetbio_geometry.m`、`generate_isetbio_geometry.py`及`build_isetbio_model.py`。实际调用固定commit d2924c2fe1d41c7dc5af1006d8081735b08f1b35的`retinalattice.generatePatch`，其全部相关m文件逐一与本地Git blob核对。传入现有Packer M.nemestrina四方向密度/中央项目桥接的明确双线性表、项目.223mm/deg、由实际最大密度计算的最小spacing；同时替换spacing与角度转换回调，不使用默认人类Watson cone/midget密度。坐标极性按上游最终翻转配对。没有改变密度scale或既有S/LM规则。

冻结master含114,949个cone点，完整可查询矩形仅为RF x/y各[-.512,.512]mm，不能称RF0–3mm全域。ID绑定固定master/source/key，不随ROI请求顺序或tile划分生成新点。上游请求50次迭代、其循环实际记录到51，因迭代上限结束，最终报告quality .6584；不宣称达到其.84质量条件。上游历史使用single后返回double的精度约定保留。此为可重现数值点阵，非逐细胞动物测量。

midget继续采用已批准中央私线和外围Wool有效汇聚，未补独立猕猴midget密度。`connected_cone_centroid_isetbio_v1`从最终通过status gate的cone→BC→RGC权重计算RF中心，矩阵轴显式区分；与固定ISET`MosaicConnector.updateDestinationCentroidsFromInputs/weightedMean`相对应。私线精确保留父cone RF；多输入使用实际权重。RF中心不写成胞体，OPL/IPL锚点和独立胞体缺口保持；无有效输入时拒绝捏造中心。Parasol/SBC密度和来源均未扩展。

当前(.34,.36,-.01,.01)mm owner及.04mm halo已重新建立受影响的HC/BC/AC/RGC和rod/RBC/AII连接。完整几何为2341cone、1054HC；owner24cone全部映射，均有HC候选。新owner自然没有small-bistratified目标，完整116个S cone/relay几何仍保存在图中；新模式明确记录EMPTY_SAMPLE，不改seed、密度或ROI强造样本。实际祖先视图装配29组件、70状态块：cone1564、HC1054、midget BC/RGC ON/OFF各24、parasol ON3/OFF2、rod4659、RBC168、AII367，保留6720个rod–RBC接点及1095个AMPA接点。此局部实例不称全部20层都有非空owner。

等质量master重建暴露原分配器的浮点尾数耗尽错误；仅在原有1e-8容量审计上限内守恒分配尾数，数量/密度/审计阈值不变，新recipe记录生成代码SHA，旧master保留。孔径支持从新完整master按实际ID映射，按原0.79×六近邻中位spacing规则重算；输入与背景签名包含新图、位置、类型和支持。GUI新增可选择的“ISET几何”模式；切换清除旧几何的capture/native-background/checkpoint绑定，运行须准备独立收敛背景。旧默认`atlas_population_revision_v1`和历史文件保留。本轮只装配模型，没有新ISET刺激捕获、背景预热、神经刺激运行或性能实验。

[统一核验](../reports/isetbio_model_20261015/verification.json)记录40项累计相关局部测试（本次共享目录9项）、装配与剩余缺口。可变cone ROI接口已实现，但完整神经图仍限定这个冻结局部实例；跨中心/周边运行尚需共享的rod/RBC/AII/parasol master与完整上游/边界支持，不能每个ROI重采样后冒称同一atlas。独立midget密度、HC全域来源/闭合及既有S/H2/内层动力学标定缺口仍未填，20°条带、停用rod/RBC及新的数值方法均未激活。


#### 组件差异、统一配置边界与共享人口目录

旧34→新29组件的五项差异是S_ON_BC、small_bistratified_S_ON，以及small-bistratified的S_ON_BC兴奋、diffuse_BC_OFF兴奋、local_feedforward_AC抑制三个conductance组件。七个状态块分别是S_ON_BC.history_0/1/2、activation、V、terminal_release和small_bistratified_S_ON.V。没有新增组件。保留组件的向量长度随新几何改变，总标量53759→59107，并非整体减少细胞计算量；逐项长度和单位见[component_diff.json](../reports/isetbio_model_20261015/component_diff.json)。同一半开owner框内旧L13/M13/S1，新L11/M13/S0。旧6个动态S-ON BC是该owner S-RGC的上游祖先，owner为空时整条对应执行路径不实例化；完整116个S cone/relay几何仍在冻结图中。模型代码支持五类目标，与这个局部实例实际只执行四类owner必须分开。运行audit新增canonical_outputs、active_owner_outputs及EMPTY_SAMPLE状态，不用假细胞或生物缺失声明补齐。

midget数目仍是一颗非S cone分别建立ON/OFF候选，BC使用qualified parent私线；外围Wool源拟合约束汇聚和尺度，并不提供独立RGC密度，当前外围候选数量不能称测量校准。RF由最终通过状态门的cone→BC→RGC权重质心给出，当前中央minimum-one路径仍落在父cone位置；IPL点和胞体不是RF的同义字段。独立macaque midget密度的来源/模型选择未擅自补齐。

点阵没有达到原生质量/位移停止条件：quality .6584相对源码条件.84；末次日志约.023µm是内部步位移的30分位而非最大误差，配置中位移tol为.000215µm。返回“iteration11”是上游打印的history索引，不可写成实际只跑11轮。对已有数据按实际双线性密度表积分，完整方框66803点对66348.309期望质量，差+0.6853%；owner24点对26.07361，差−7.9529%。这些只描述数值采样差异，不是生理误差、统计CI或收敛误差界；没有重新生成或长收敛实验。

新[population_catalog.py](../retina_sim/population_catalog.py)将已生成rod/RBC/AII/parasol ON/OFF五个人口master冻结为同一目录，全部保留原source、namespace、ID、坐标与各自有限域。register_existing_masters只登记既有资产；SharedPopulationCatalog.crop不调用生成器。require_domain及footprint_coverage区分源覆盖不足与真实空样本，拒绝隐式RF/OPL/IPL换算、域外重采样、补边和密度缩放。revise_local_graph可显式使用共享目录；rebuild_roi_from_shared_masters已在另一个更窄owner(.341,.359,-.009,.009)mm通过真实几何重建，五个master的SHA和namespace全部不变。该测试没有神经积分。

目录仅共享这五个人口的既有nominal RF/physical-chart点集；cone/HC各有原固定master，派生IPL锚点、BC/AC字段和连接完整性仍需自己的共享支持。这不是全域ROI动力学完成。未选20°条带，未停rod/RBC，未扩parasol/SBC来源。原默认继续保留，新GUI项明确是迁移候选，不能作为最终两套配置交付。最终应统一到一个canonical配置，历史作为fixture保留；统一前需明确五类目标/空owner合同、当前点阵的用途和数值接受标准、新输入/孔径/收敛背景身份，以及允许ROI的typed上游/有限边界支持。当前继续保留已批准private/Wool有效路线，无须重复批准；仅在升级为独立猕猴midget密度时才需另定来源与适用域。点阵最终数值接受/后续有界松弛标准尚未定，本轮不启动新长收敛。


本轮补充`typed_view.CompiledTypedView`：在同一个既有完整有限神经网络内选择子ROI记录对象，保留各typed坐标角色、ID、原连接和权重。当前支持owner(.34,.36,-.01,.01)mm内的子框，RF输入域为(.3,.4,-.05,.05)mm；跨域、缺少非零输入行、未知内部空间依赖或要求外部递归闭合时明确拒绝。全owner闭包解析6928个组件/细胞节点、22364条边及2315个外部通道/细胞对，包含native A/B、rod化学路径和电耦合；仍是finite-open model，不能作为全视网膜外部闭合证明。`geometry_identity.py`把实际typed人口数组、CSR权重、native A/B、gap、组件参数/代码、时钟和冻结来源绑定为同一身份；新候选输入缺失该身份即拒绝，输入重基准和背景cache读写也复核，历史fixture继续沿用其旧合同。相关49项独立局部测试通过（两轮有重叠，详见统一核验）；背景缓存仅测试t=0存取，没有新预热、刺激或性能试验。扩大完整神经ROI仍需派生BC/AC/IPL共享资产和逐层外部边界合同；该旧master的日志q.6584单列保留；新独立点阵见下文，不自动迁移其typed图，默认切换仍暂停。


当前固定域的共享派生资产已完成：`circuit_geometry.py`把原冻结图内的BC/AC/IPL表、接触和parasol锚点作字段hash绑定，并在`circuit_geometry.json`冻结实际native A/B、rod接触候选/权重、化学接点与overlap及gap配对/电导。装配直接加载，逐项核对实际人口/坐标/CSR/gap；禁用候选查询、锚点生成和gap最近邻的测试仍能装配原29组件/70状态块。参数及状态schema与冻结前一致。`owner_recording`返回原RecordingPlan，仅筛owner/记录，整网状态和接线保持相同。owner(.34,.36,-.01,.01)mm与RF输入(.3,.4,-.05,.05)mm未扩大；全owner必需输入行缺额0。完整派生几何含midget BC ON/OFF各2225、diffuse BC各597、local AC各2486、widefield AC924及S-ON BC116；这些表的存在不表示全部都是当前动态owner。16项本轮独立局部回归通过。有限开放边界已获批准，域外无边仍是其明示局限，无新增边界决定待批准。冻结坐标的新静态Delaunay诊断q第0.8百分位分别为完整查询方框.565054、RF输入.729506、owner.775375；它不复现MATLAB历史三角化/椭圆过滤或停止状态，日志.6584不能冒称该坐标集的重算质量。原算法阈值.84未降低，迭代上限停止与生理未验证分别保留；该段记录形成时300轮候选尚未获批、未运行；其后批准的独立候选及源停止修复见下文。没有新动力学、背景或性能实验，最终默认切换仍暂停。全部证据并入原统一核验。


生成器数值修复与完整初始化覆盖（2026-10-15/16）：原固定Euler步首步翻转51个面，滞后Delaunay使第3步最近点对不属于缓存弹簧，第7步达到.007µm。`lattice_relaxation.py`保留原归一化排斥弹簧及gain1.1，修复拓扑缓存、内部面/全轨迹数值碰撞保护、隐式椭圆投影及double坐标/quality同步；不设生理最小距离。当前为`pinned_retinalattice_springs_numerical_repair_v4_source_stop`，12项局部回归通过。整数边编号去重与原二维排序在2400步完整121811点、364498条边上，连接及顺序、目标长度和弹簧力逐位一致；没有改变弹簧模型。旧资产及诊断均保留。

用户2026-10-15 13:26 UTC批准补齐同一椭圆的初始化覆盖。`export_lattice_initialization_covered.m`保留源seed、spacing、旧方格相位和原Bernoulli sampler，114949个旧样本逐点逐序一致；候选由470164补至517388，独立重枚举确认原遗漏47224、补齐后0。继续原随机序列新增6862个，总121811点；未删点、改密度、改LMS或改变椭圆。原弹簧力与固定MATLAB同状态比对最大相对差3.1e-15。RF半轴保持约.799370与.978989mm，原完整查询方框仍为±.512mm，没有扩大运行canonical ROI。

用户2026-10-16 01:33 UTC批准恢复源停止合同，从有效1400步末帧及完整历史续算；此前v3把内部第30百分位运动统计收紧为全点最大位移，不是等价的公式修补。当前合同为`(q第0.8百分位>.84 AND 原width历史条件) OR (源码内部p30运动量<.000215µm)`。源码运动量是`prctile(sqrt(sum(.2*F.^2,2)),30)`，Hazen百分位；内部掩码在名义未投影位置`x+.2*F`上取`ellipse_distance<-.00215µm`，即使安全回退缩小实际步长也不缩小该源码统计。.2是几何数值步长，不是生理时间；该统计与真实位移分别记录。width使用全部已保留历史，不改为滚动窗口；每步刷新拓扑的已批准修复使quality采样周期不同于原滞后缓存，不声称复现原未修复轨迹。1400步以前缺失的源码运动统计不补造。

本轮在5677步首次由原内部p30支路停止：5676步为.000215003484µm，5677步为.000214910618µm，原容差.000215µm未改。末帧重算同一源码统计为.000214787336µm，仍通过；最后一次坐标更新逐位复现，坐标/quality精确对应。冻结的是5677步实际停止末帧，4724步最佳quality快照(.827074)另存，不冒充停止状态。

| 最终状态 | 数值/状态 | 在停止合同中的作用 |
|---|---|---|
| q第0.8百分位（Hazen） | .825473076，未过.84 | quality支路未触发 |
| 原width全历史条件 | 满足 | 与q条件共同构成quality支路 |
| 源码内部p30运动量 | .000214910618µm < .000215µm | 实际触发停止 |
| 全点最大真实位移 | 最后一步.021522959µm；末帧重算名义步.021319657µm | 仅诊断，不作为额外门槛，不表示全点静止 |

完成资产为`data/atlas/isetbio_macaque_cones_source_stop_v1_20261016`，manifest SHA256=`5ab2429aa920e9bd07f7a7d1a335edeffbbf8f263ddce731352809e7d6218a3a`。121811点全部保留；517388个初始化候选独立重枚举遗漏0，重复0、机器精度近重合0、数值域外0（1e-10µm检查界，最大浮点超量1.08e-11µm）。全域最小NN为1.869425µm；整椭圆密度积分121491.06，计数差+.2633%，不是生物置信区间。既有`.79*median6NN`孔径估计下，整域28点、查询框2点有圆盘重叠，RF输入/owner为0；孔径未测量，该诊断不等同于真实细胞碰撞或生理验收。归档包含初始MAT/RNG、固定请求/来源、各段实现与状态链、停止前后坐标、最佳快照及核验；标准master读取器加载并逐坐标匹配。完整核验见`reports/lattice_coverage_repair_20261015/source_stop_final_20261016/verification.json`，最终几何/趋势图为`reports/lattice_coverage_repair_20261015/continuation/final_geometry_and_trends.png`。历史coveragefix/v3及旧master保留；新资产不得混用旧输入/背景；上述生成任务结束时尚未迁移默认与typed图，也未运行新背景或神经刺激。后续迁移结果见文首；生理验证仍未通过。


### 2026-10-19 数值执行与初始化原型（待独立审阅）

本轮仅新增独立实验模块与入口，未切换GUI或默认backend，未修改既有方程、参数、FP64、adaptive DOP853、rtol/atol/max_step及事件/history/RNG实现。新GPU模型身份独立，旧checkpoint不迁移。审阅入口为[证据索引](../reports/runtime_prototype_review_20261019/evidence.json)和[实际代码差异](../reports/runtime_prototype_review_20261019/changes.patch)；实现自测不等于独立审阅。

`gpu_device_inner.DeviceInnerPlan`从原Numba指令/CSR生成设备程序，`gpu_device_joint.DeviceJointPartition`保持全部4,967,023个FP64状态（native4,936,006、下游31,017），复用原A/B。下游连续RHS、CSR与gap传播在CUDA执行，不按stage回传release/下游状态或上传下游导数；光输入、CPU事件与记录仍走原边界。新路径沿用原GPU后端的机制/布局限制，不能推为全部平台机制均已支持。

台式机RTX4070 Ti SUPER上，各独立候选均先warmup、再交替三次同state/input测量；下表为完整单RHS（含必须的数据传输、GPU结果生成与完成同步），不是整窗/GUI加速。诊断性CPU下载、setup、warmup和另一次profiler排除在该计时外，原始边界均在JSON说明。

| 独立路径 | 同批原路径ms | 候选ms | 结论 |
|---|---:|---:|---|
| 首版设备RHS | 6.45 | 20.96 | 大量小kernel启动，暂不采用 |
| 完整校验批处理 | 6.40 | 9.19 | 保留所有原finite/positive/nonnegative检查 |
| 固定inner RHS CUDA Graph | 6.15 | 7.14 | 只捕获固定形状/地址的纯RHS；不捕获solver/事件 |
| A两列SpMM | 5.80 | 24.31 | 未获收益，暂不采用 |
| 原生pointwise NVRTC融合 | 6.01 | 19.49 | PyTorch自带NVRTC，未安装工具，暂不采用 |
| Graph＋校验批处理（分项后组合） | 6.45 | 5.26 | 单RHS耗时约少18.4%，尚未接受数值预算 |

完整图42个下游CSR共23,427个nnz；native A/B各3,627,874个nnz。[真实trace与账目](../reports/device_joint_real_rhs_v1/result.json)记录原每RHS约4.50MB大向量D2H在候选中为0，标量域检查同步仍在；组合另有D2D输入复制，未隐去。组合与首版设备RHS逐位一致，但对原CPU-inner有217个下游电压导数不逐位，最大2.423e-16 V/s、原监测尺度归一最大1.661e-12。[逐指令定位](../reports/device_variants_real_v1/origin.json)的首个差异为CSR指令152，0.5000000000000001与0.5，随后传播；未改连接或权重。SpMM/融合另有native舍入差异，逐块单位、误差及未通过门槛均保留，未用混合单位global max作生理结论。

`background_seed.seed_background_state/prepare_seeded_background`只复用原Native/Finite实际向量平衡与Chen rod固定点，独立复制状态；100td构造参考、release/current anchors与model hash保持。真实530,929锥背景向量SHA为4760a19165ec85d86716074b485f4d07cd47f37c148e477f6e4f339a9fde1b68，范围9.4844–231.399 reference-equivalent td，rod仍5 R*/s；不是把刺激首帧当背景。[真实种子测量](../reports/background_seed_real_v1/result.json)18.16s，outer原RHS norm由2.655e6降至1.886e-9，但整网仍有AII电压norm1733及BC瞬态。seed不是prepared background。

`hc_anderson_seed.anderson_equilibrium/seed_hc_anderson_state`仅对原HC固定点作Type-II Anderson、阻尼/正则/域保护，不构造HC×HC雅可比、不训练。原source residual验收1e-8保持，另设更严格数值working target1e-13（不冒称原LS gtol的含义）；数值失败真实调用原equilibrium，取消/内存保护直接中止。[真实同条件初始化](../reports/hc_anderson_real_v1/result.json)17.89→14.54s，85迭代/277次map，HC residual9.237e-14，无回退；最大state差2.061e-13 native voltage、差/原solver尺度最大6.062e-9，仍标REVIEW_REQUIRED。原reference分母与RNG/history/clock完全保留。原最短1.25s及真实洗入不缩短，完整适应尚未执行，不能宣称整流程1.23倍加速。

小测试：首版CUDA12项通过；背景20项与Anderson20项通过；含非均匀扰动的五候选CUDA测试20通过、2项严格逐位门槛失败（SpMM/融合），保护/别名/重算/checkpoint检查保持。早期NVRTC调用出现Windows0xc0000008；[隔离探针](../reports/device_fusion_probe_v1/result.json)与后续显式项目TEMP/cache、pytest -s的完整执行正常，但根因未确证，失败记录不删除。数值预算与独立审阅未通过前，不推进20–40ms/0.5s、不启用默认；无CPU静默fallback。GPU与背景大任务串行，保留14GiB RSS/8GiB free及15GiB启动余量。

Macular来源核对：本机官方1.5.6与旧实验调用确实存在，项目自述参考其细胞/通路框架；当前BC/AC的FIR+softplus、RGC的电导膜/history随机发放，以及BC terminal，分别与官方BipolarGainControl、Amacrine、GanglionGainControl、BipolarAMPA的状态/方程不同。不能把project_profile解释为凭空原创，也不能据此称直接转录。可见Git从整项目导入开始，具体公式改写链仍无法追溯；[逐类文件/函数及hash](../reports/runtime_prototype_review_20261019/macular_provenance.json)保留该缺口，未为补来源改方程。


### 2026-10-19 审阅覆盖修正与严格门槛

按独立审阅补齐可持续运行的回归，未修改生理参数、求解器精度、稳定性准则、默认配置或正式后端注册。`gpu_ordered_joint.py` 是独立身份、未接入 GUI/正式选择器的候选；保留原始 CSR 行内边顺序和 gap 累加顺序，选用入口明确拒绝 SpMM/融合。历史候选及其失败断言保留。

- `tests/test_gpu_runner_lifecycle.py`：11 项。直接调用新 runner.advance，在普通 fixture 和64cone/96rod、3292状态、4组gap的完整通路上覆盖 rod、TRPM1、AMPA、Gly、全部状态/输出/单位、2.1ms事件边界、checkpoint续至2.6ms及13次拒步隔离。最终8通过、3个严格数值失败，0跳过。所有比较先保存完整诊断，避免早期数值断言阻止后续证据收集。
- 两图的完整辅助状态（时钟/tick、RNG/spike、history/delay、输入位置）及除后端身份外的元数据均一致；两后端各自的checkpoint直接/恢复续算逐位一致，跨后端checkpoint被拒绝。full_target跨后端连续状态/记录仍不逐位一致，其数值失败如实保留。拒步用AST定位原分支，原13拒步/40接受步/647nfev门槛不变。
- `tests/test_background_failure_lifecycle.py` 新增28项，与原种子/Anderson测试合计68项全部通过（42.890秒）。覆盖真实入口的种子构造失败、单位/证据合同错误、Anderson数值失败调用原equilibrium、内存/取消不启动额外fallback、预算/保护停止保持NOT_REACHED、完整checkpoint和绝对时钟续算。1.25秒最低适应、0.1秒窗口、连续3窗及100us监测均保留；小fixture续至4ms仍为NOT_REACHED。
- 完整GPU套件50通过/5失败（80.75秒）：原正式CudaHybridRunner10项通过、旧device12项通过、历史variants20通过/2数值失败，新增lifecycle8通过/3数值失败。补全诊断后的lifecycle单独复跑8通过/3失败（35.98秒）。没有xfail或阈值放宽。

同一非均匀full_target RHS的差异计数从原84降为19。实测独立首差是POW20（同输入三次幂）、EXP90/93及SIGMOID456；保存输入/输出FP64位型与传播链。剩余midget_BC_OFF.V 1处，最大1.3877787807814457e-17 V/s；rod_membrane.V 17处，最大1.294103713078698e-15 V/s；rod_BC.release 1处，最大1.8485213360008856e-14 dimensionless/s。原84路径的首差推导与新路径实测在证据中明确区分。不得将这些小差值当作已经接受的误差预算，候选仍严格未通过。

证据入口：`reports/runtime_coverage_fix_20261019/evidence.json`，算子证据：`reports/ordered_rhs_origin_v1/summary.json`。原独立审阅文件未覆盖；其记录的22份历史文件SHA256均保持一致。真实0.5秒和至少1.25秒背景适应仍为NOT_RUN；本轮按用户要求先完成覆盖并交回复核，未将工具准备或种子构造冒充验收完成。两个串行验收工具已准备，其中背景工具已完成小fixture完整checkpoint/resume检查；真实网络尚未执行。


### 2026-10-19 Ordered 数学原语根因与兼容性边界

独立覆盖复核确认：Ordered是本轮下游GPU常驻的活动目标，剩余19个RHS差异及2.1/2.6ms传播仍阻塞目标。旧CudaHybridRunner下游仍在CPU，其10项通过只证明旧路径未回归，不能替代Ordered验收。本轮未修改生理公式、参数、solver/背景准则或生产lowering；也未把下游移回CPU，未启动大图/背景完整适应。

原Numba实际语义已定位：fastmath=False、boundscheck=True、error_model=numpy、USING_SVML=False；原dispatcher LLVM含llvm.pow.f64/llvm.exp.f64，x64汇编通过pow/exp符号调用。安装包core/externals.py:147–152把符号绑定到_helperlib的C包装；实测机器跳转链经过_helperlib.cp311-win_amd64.pyd的相对跳转/IAT，最终精确对应System32/ucrtbase.dll的pow/exp/log导出。UCRT版本10.0.26100.9711，SHA256 dedb7611e1abc1ff3852dcac3d4c93a048d729e25b061973b9969f22698f14ca。新原语oracle直接调用原dispatcher的同一opcode/参数/编译签名，重放捕获输入与原work结果逐位一致，未另造CPU近似参考。

有界实测使用原捕获操作数、每个输入上下各4个nextafter邻值及seed=20261019的代表性有限输入域。真实NVRTC/CUDA内核固定double，在PyTorch已有context/stream执行；FMA contraction分别开启和关闭，未启用use_fast_math。结果如下（每项是相对原CPU结果的不等数量，不是允许误差）：

| 原语 | 捕获输入 | 当前Torch/直接CUDA，FMA开或关 | 捕获值±4ULP邻域 | 其他已试方法 |
|---|---:|---:|---:|---|
| POW3 | 96 | 22不等 | 231/864不等（tensor pow/直接CUDA） | Torch整数幂、逐次连乘仍22；GPU补偿cube降到1/96，但邻域仍2/864 |
| EXP90/93 | 360 | 21不等 | 54/1080不等 | 显式直接CUDA与Torch结果相同；关闭contraction未恢复CPU语义 |
| SIGMOID456 | 8 | 1不等 | 6/72不等 | 显式RN add/div、普通公式、Torch builtin sigmoid均未逐位恢复 |

补偿cube确实在GPU实现two-product误差补偿，但其余下反例不能用查表/特判抹去：输入0x1.d0fd92c8811dep+3，UCRT输出0x1.7f861b14c8f1ap+11；110位和180位Decimal独立精度给出相同舍入值0x1.7f861b14c8f19p+11，GPU补偿cube亦为后者。因此“改成更正确舍入的数学函数”与“复现这版CPU库的每一位”是不同数值合同。这没有证明跨库差异无害，也没有批准误差预算。

UCRT内部FMA3默认1。仅在一次性CPU探针内切换为0，捕获输入及邻值的POW/EXP/SIGMOID均仍与原CPU参考相同，所以不能解决现有GPU差异；代表域EXP另有1值因切换而变1ULP。finally恢复为1，另一个新进程确认仍为1且fegetround=FE_TONEAREST。Numba的fastmath=False并不会关闭CPU数学库内部算法。NVRTC的fmad=false也只禁止普通乘加合并；探针PTX仍有显式FMA，未宣称禁用了库算法中的全部FMA。

本机已安装SDK的UCRT Source/10.0.26100.0目录已核查，不含math目录或pow/exp实现。现有Torch/NVRTC未提供已验证的这版UCRT兼容设备实现。严格逐位移植在原理上可行，但需要对应版本/分派的算法与系数、运算序和边界行为，并完成专门数学库验证；本轮没有这些实现，不能在合理的局部修复中声称已完成。微软也明确说明UCRT数学函数可能不是正确舍入，结果会随CPU实现、内部FMA算法或系统版本变化。[Microsoft math支持](https://learn.microsoft.com/en-us/cpp/c-runtime-library/floating-point-support?view=msvc-170)、[FMA3分派](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/get-fma3-enable-set-fma3-enable?view=msvc-170)。CUDA文档同样区分基本IEEE运算与不保证跨库逐位相同的数学函数。[NVIDIA浮点说明](https://docs.nvidia.com/cuda/floating-point/index.html#verifying-gpu-results)。

后续合同建议仅供审议，未应用：若逐位合同不变，需取得并移植固定UCRT算法；若继续使用现有GPU数学库，应另行批准有版本的跨库数值等价合同。可审议复用旧verify_cuda_execution已有的状态差/原solver尺度<=0.01规则，并为RHS使用同一冻结background单位尺度下1%原数值准则的预算；这两个规则当前均未授权给Ordered，不更改solver容差。离散时钟/事件/RNG、单位/图/输入身份、各自checkpoint续算及拒步隔离仍严格比较，未注册输出尺度必须拒绝。另一条方案是为新CPU参考和CUDA固定同一可移植FP64数学实现；这仍会改变数值参考/检查点身份，需要显式合同批准。详见proposed_numerical_contract.json，不能把提议当已通过，也不能把测得最大1–2ULP直接升格为允许值。

证据入口为reports/gpu_math_compatibility_20261019/evidence.json。新增tests/test_gpu_math_primitive_gate.py将三个原语及邻值固化为普通严格失败（3 failed/0 skipped，4.09秒）；原生命周期/历史失败和门槛未变。原语尚未通过，故未接回lowering或无意义重跑完整生命周期；已有独立复核的3项Ordered失败仍未关闭。背景原型范围不变，未完整适应仍未验收。


### 2026-10-19 已批准数值合同、流式验收与实际短窗计时

本节更新上一节“尚未授权数值等价”的状态。用户已批准独立数值合同（Sentinel_afc3ce1b939c819198724a39e1277255）；冻结文件为 `configurations/numerical_equivalence/ordered_fp64_v1.json`，SHA256 `e85c3a8f984471a34a3caad28664945a8555a978b58d3435a424c5d6db2ffeaf`。这是工程数值验收标准，不构成生理校准或全局误差证明。每状态原尺度为 S=atol+rtol*max(|original|,|candidate|)，逐标量最大差<=.01S，参考增量<=.0025S，并检查相对细参考不退化。没有背景尺度下限；输出按原公式传播区间。事件/RNG/历史/时间戳及同身份checkpoint仍严格检查，旧bitwise失败保留为普通失败。

新路径 `NumericalOrderedCudaRunner` 采用明确的私有合同身份，未接入GUI默认。生产FP64、DOP853容差/最大步长、源参数、连接、时钟和RNG未变，原机制/积分器/默认配置9组SHA保持一致。QA的收紧参考独立运行；原始小图参考对未收敛的记录保留，进一步收紧后，小图全部3292状态及输出在2.1ms/2.6ms通过，同身份checkpoint续算逐位一致且跨身份拒绝。

`AuditSession(cpu_stream_consumers=...)` 逐接受步完成最后一组CPU参考的全部并集查询后释放F7/状态，只保留时间戳与计数；不更改原solve_ivp endpoint-only行为。`shared_cpu_reference` 复用原GPU runner已有的完整CPU RHS与同一锁，避免重复编译计划和scratch；参考仍走原CPU SciPy DOP853，不是候选GPU回退。流式CPU审计13项、CPU计划复用2项通过；小图原保留式与流式的接受步并集均452点，原非网格记录回归仍保留。

实际图4967023状态/77状态块，0.2ms短窗和随后至2.2ms的23个自然边界分块均通过四臂合同，包括1ms/2ms RGC更新、0.73ms系列AC时钟和所有接受步并集/监控边界两侧。严格失败0；最大状态差3.924e-8 S，最大参考增量2.708e-5 S。全过程峰值RSS10.34GiB，14GiB峰值上限和8GiB可用内存保留线未放松；此前保留式QA在2ms后触发保留线，峰值11.75GiB。

生产计时独立于上述收紧参考。在同一实际图、独立原厂初态及既有静态输入上，各路径一次固定顺序2.1ms测量，包含传输、事件、原host观察、记录、最终同步及正式checkpoint写入；恢复单列，四路均完整恢复状态/aux逐位一致。Graph及空发布过程与Ordered无Graph的全部状态/aux/输出/时钟逐位一致。

| 路径 | 推进秒 | 推进+输出+checkpoint写入秒 | 恢复秒 |
|---|---:|---:|---:|
| 原混合后端 | 4.85 | 18.30 | 11.99 |
| Ordered GPU | 6.12 | 19.34 | 12.14 |
| Ordered GPU+Graph | 4.37 | 17.60 | 11.96 |
| Graph+既有跳过空发布过程 | 4.06 | 17.24 | 11.98 |

最佳路径本次推进减少16.3%，含输出/checkpoint写入减少5.8%；这不是完整窗口或已适应背景的通用加速比。未实现device observe；原host观察仍在。checkpoint序列化/写入约13秒，不能用局部RHS或seed收益替代整个流程收益。

本轮按明确预算在2.2ms结束，标记为 `NUMERICAL_PREFIX_PASS_FULL_WINDOW_AND_BACKGROUND_PENDING`；未通过完整0.5s或新候选至少1.25s背景验收。实测2ms响应前缀的四臂QA耗时289.72秒，外推完整0.5s约20.1小时，含收紧CPU参考、插值审计、全状态比较和同态RHS重放，不是生产推理成本。下一步应先评估既有checkpoint流式/严格解析加载路径及保持身份的分段续验，再单独安排长验收预算。工具默认限制QA前缀至2.2ms，完整标准仍为0.5s，不把短窗当全窗通过。

统一证据入口为 [numerical_execution_20261019/evidence.json](../reports/numerical_execution_20261019/evidence.json)，实际结果及计时在 `reports/numerical_contract_real_v8/`。当前无大图进程；没有创建定时任务。两组被替代的小图工具错误目录已在核对Temp归档SHA后移除，未收敛/严格失败/内存停止及历史来源证据保留；19个执行源码/配置/测试文件有已验证回滚归档。既有用户改动和本文原有前缀完整保留。


### 2026-10-19 选择性 device observe、实际0.5s生产计时和磁盘续算

本节更新上一轮“device observe未实现”的记录。本轮新增显式私有 `ObservedCudaRunner`，有 `cpu_selected` 和 `device_selected` 两种记录方式，未接入GUI默认后端。原厂初态、FP64、DOP853参数、所有细胞/连接、时钟、因果回调、PCG64及checkpoint协议保持原值。原生机制/积分器/默认配置9组SHA未改。只在没有因果历史组件到点时跳过无消费者的纯全网观察；GPU仅计算所需记录端口，不向神经更新反馈设备观察值。原CPU因果更新和为solver/公开observer提供的host完整状态仍保留，不能把此路径称为全流程常驻GPU。

实际profile（2.1ms，含profiler开销）中，推进4.17s，`_sample`累计1.02s、全网`evaluate` .83s、`_flat` .27s；这些为包含关系，不能相加。新路径按原依赖追溯所需端口，复用既有有序表达式lowering和CUDA Graph，每次重算动态量，仅缓存静态计划。device记录按需打包状态；原输入通道传输仍在，没有宣称总H2D为零。记录、gap current、未选位置非有限状态拒绝、事件、完整状态及checkpoint续算的小图4项回归通过。

新device路径实际2.2ms四臂合同通过：77状态块/4967023标量、144接受端点并集、75监控/边界侧点，严格失败0，最大状态差3.924e-8 S、参考增量2.708e-5 S。完整0.5s加严参考验收仍未运行；下述0.5s是单独生产计时，不是扩大数值或生理验收范围。

0.1ms整个advance的launch诊断中，无Graph/Graph的CUDA事件分别13203/13259，runtime kernel launch事件12860/1492，同步事件均101。Graph主要减少调度开销，没有借此省略原检查或减少神经计算。该诊断的profiler耗时排除在生产计时之外。

同一实际图/既有静态输入、各自独立原厂初态，完整0.5s生产窗口采用相同100us记录（5001时点）、11367次监测回调。每路均5616次solver调用、90867次RHS；固定顺序单次，不含初始化/warmup，包含传输、同步、事件、观察与记录。checkpoint在整个响应结束后只保存一次，恢复单列，不摊成每个推进窗成本。

| 路径 | 完整推进秒 | 推进+输出+checkpoint写入秒 |
|---|---:|---:|
| 原混合后端 | 1332.65 | 1348.30 |
| Graph+跳过空发布 | 1173.27 | 1188.79 |
| 选择性device记录 | 917.04 | 932.56 |

本次选择性device路径比原后端推进减少31.2%，比Graph再减少21.8%；含输出及一次checkpoint写入减少30.8%。0.5s窗口内跳过4438次无消费者全网观察，仍执行1179次所需因果观察。其完整末态/aux与Graph逐位一致；所选记录的15个字段（含计数、window rate和时间）也全部逐位一致。这个记录范围不是全网每个端口，原后端与Graph的全0.5s加严CPU参考验收仍未完成。观察优化没有标定原项目的生理参数；单次工厂初态计时也不是已适应背景下的通用速度结论。

补齐了上一轮尚未覆盖的真实磁盘读取与继续推进：保留 `actual_v1/selected_device_checkpoint.json`，实际读文件后恢复并继续到0.5001s，与直接推进的全部状态、aux及记录输出逐位一致。上一轮约12s的恢复是已持有JSON文本的内存恢复，未含磁盘读取或真实大图续算，不回写其历史结论。

严格解析器仅将重复registry hash改为复用引擎既有、经过深不可变类型检查的 `_state_schema_hash`；registry替换/可变结构仍重验。原绑定、几何、嵌套hash、模型/输入身份、状态形状、可变隔离及续算19项回归通过。实际同一磁盘checkpoint，原文本读取+恢复13.81s；读取+显式JSON解析+严格解析恢复12.24s；一次性registry身份准备.93s单列。所有解码状态/aux及续算逐位一致。OS文件缓存未清空，不把单次约11.3%恢复收益当作每步推进收益。

唯一新证据入口 [runtime_segments_20261019/evidence.json](../reports/runtime_segments_20261019/evidence.json) 汇总原始profile、launch trace、2.2ms合同、完整生产计时、真实磁盘续算、4+19项测试、diff、回滚及清理记录。[实际响应图](../reports/runtime_segments_20261019/production_response.png) 来自已保存NPZ，仅用已有Anaconda基础环境绘图；模型/验证/恢复均由固定retipath_v1执行，无环境升级。图中条件率不是观测window firing rate。

本轮无活动大图进程、无自动化、未切换默认后端。新候选至少1.25s背景验收和全0.5s加严四臂合同仍待完成（此前预算约20小时），没有以本次生产测速替代。9个源码/测试/工具文件及完整diff有已验证回滚归档；过时工具错误中间件和重复Graph checkpoint经Temp归档SHA核对后移出仓库，必要独立审阅/原始launch与数值证据及新可恢复checkpoint保留。本文原有前缀和用户并行改动完整保留。


## 2026-10-19 建模范围与模块校准接口

本轮落实已批准的范围收紧和“先改代码、再校准”。默认方程版本为 `effective_dynamics_v1_existing_equations_20261019`：原有全部参数记录及共享 Run 默认值逐项保持不变；这不代表未定的新动力学草案已经全部替换，也不代表生理校准完成。

### 可执行的范围约束

- ON/OFF midget、ON/OFF parasol 是四个主要定量评价目标；S-ON small bistratified 保留为补充目标。五类输出继续装配，标签不等于逐细胞形态确认。
- `local_feedforward_AC`、`local_crossover_AC`、`widefield_AC` 是 generic 有效功能单元，没有已鉴定亚型或独立原始数据全覆盖验证的声明。`modeling_scope_v1.json` 约束其实际方向、目标和参数绑定；参数装载与中央编译器执行检查，结果 manifest 同时携带这些限制。GUI 不增加长说明。
- 19 个现有参数族的数值、单位、绑定方式冻结：三类 AC 的核/偏置/采样/历史，五类 RGC 的膜/发放/历史系数及 E/I 增益。上游基线、条件初始 V0 仍是派生量。没有新增按亚型自由调参或优化器；新校准系数须进入有来源和适用范围的新 profile/scope 版本。
- 实际 generic crossover 为 `diffuse_BC_OFF → local_crossover_AC → midget_RGC_ON`。AII 经 ON BC 双向 gap 及 OFF BC glycine 作用于另一方向/位置；ON 驱动的 local/widefield 抑制与 AII 对 OFF 通路的影响可能重叠，但尚无贡献分离数据。因此全部保留，没有凭名称删除或凭拓扑宣称已排除重复。现有 generic RGC 抑制总增益系数仍为 0.4 nS，不能当作测得的各通路贡献。
- [Strettoi 2018](https://www.nature.com/articles/s41598-018-34621-2) 支持 AII cone-driven ON→OFF crossover 的结构解释；不据此替代 OFF→ON generic route。[Crook 2014](https://pubmed.ncbi.nlm.nih.gov/24801624/) 作为限制 surround 归因的来源；代码不把 widefield 声明为全部经典 surround。

### 独立模块共用整网方程

`retina_sim/calibration_modules.py` 通过既有 `RetinaBuilder`、组件工厂与运行器建立独立模块，不另写一套细胞动力学。

| 模块 | 边界与保留内容 |
|---|---|
| `cone_hc_joint` | 保留 native Cone–HC 联合反馈和原有 H1/H2 表；原工厂使用声明的 td 背景初始化。S/H2 共用 H1 动力学的近似仍在。 |
| `bc_clamped` | 保留实际 HistoryBipolarTerminal 或历史 residual 方程；全部被切断输入改为具名钳制，任何打开的 gap 都必须逐项确认。 |
| `generic_ac_clamped` | 保留 ResidualAC；驱动基线有单位且显式给定，核与偏置不自由拟合。 |
| `rgc` | 保留 HistoryConductanceRGC。gE/gI 使用 S；可选择独立的 A 单位正向内注入电流端口。未选择该边界时原整网 RHS 完全保留。 |

默认 `inner.py` 已恢复为原编译器逐字节校验的版本，原 compiled/GPU lowering 及来源 pin 没有改动。RGC 电流注入由独立 `calibration_current_clamp.py` 边界包装既有工厂，仅增加 `I_injected/C`，使用具名 `HistoryConductanceRGCCurrentClampBoundary`；该实验路径限定 Python reference，显式 numba 会拒绝，不能静默漏掉注入电流。默认整网仍使用原 RGC 机制。膜参数、严格过去 spike-history、发放非线性与事件采样规则均未替换；没有新增未知离子通道、钙/释放资源机制，也未删除 rod、S、H2 或 AII。

所有输入必须提供：有序 cell IDs、每个 bin 的显式秒边界、逐细胞值、单位/含义、实验时间原点、基线、来源、区域与适应状态。禁止推测帧率、隐式重采样或 nS/pA 的隐式转换。模块保留初态的来源，有限运行不自动称为适应稳态；独立 BC/AC 不代表恢复了整网反馈。

电压钳 EPSC/IPSC 不能直接当注入电流或突触电导。显式转换接口要求 A 单位、正向外电流、已隔离且扣漏的受体电流，以及 V 单位 holding/reversal 声明，按 `g=I_outward/(Vhold-Erev)` 转换；不猜反转电位、不静默整流，也不推断串联电阻/空间钳校正。

### 运行、参数包和缓存失效

只读查看当前共享默认 RGC 模块的端口与细胞顺序（不推进神经时间）：

```powershell
& C:\Users\PC_User\anaconda3\envs\retipath_v1\python.exe -B -X utf8 tools/run_calibration_module.py --component midget_RGC_ON --kind rgc --mode conductance_clamp
```

实际运行必须另传 `--run --contract <实验合同.json> --output <新目录>`。合同包含 `conditions`、逐端口 `inputs`、显式 `solver` 和 `parameter_snapshot_version`；输入可使用 JSON 数组或指定 NPZ 的 edges/values 键。源码 API 是 `build_module`、`PortTrace`、`voltage_clamp_conductance`、`parameter_package`、`merge_parameter_package`。不提供猜测的真实数据映射或假校准示例。

参数包保存版本、来源、适用条件、原 profile/scope 身份、组件实现 hash、单位/证据及失效依赖。Native Cone–HC 保留 `ParameterSet` 的原始记录、profile、parameter/source-manifest 身份和原始 ms/scaled 单位；不把它假装成 `MechanismProfile.records` 或捏造 Quantity 含义。CLI 先检查可序列化性，只有 module、outputs、parameters 全部写完才原子发布成功 result；导出失败不会留下成功回执。当前可导出并合并 `unfitted_snapshot`，用于证明同参数模块能回到同一整网；初始 V0/输入基线等不作为校准系数转移。改变系数、声称 fitted 或换条件不能绕过来源约束。校准算法、可拟合参数的来源范围及拟合后 profile 尚待真实数据合同，不把接口完成写成校准完成。

失效依赖按有向化学连接与双向 gap 的反馈闭包计算。BC-ON 参数影响 AII 和其他耦合 BC 时不能冻结相关上游。旧背景只能当候选初态，不能自动当新参数稳态。光学输入仅在 optics/display/stimulus/receptor geometry 身份均不变时可复用；没有自动启动复用或长计算。编译器/参数元数据版本改变会改变运行身份，旧 checkpoint 不自动接受为本版本证据。

### 验证与剩余数据缺口

小型回归包含默认 39 组件工程 fixture、RGC 同实现 RHS、两段各 100 μs 的纯模块输入、注入电流量纲、BC/AC gap 边界及反馈失效闭包。复核后的 22 项针对性检查另覆盖 Cone–HC CLI 200 μs 完整导出/native 身份往返、晚期写入失败不发布成功、原 numba 电导驱动 RHS 对照，以及电流钳明确拒绝不支持的 lowering。测试见 `review_fix_tests.xml`；没有 GPU 神经推进。这些是软件测试，不是生理实验时长证据。共享默认源图另做纯装配检查：34 组件、五类 RGC 保留；无神经推进、背景准备或输入生成。证据在 `reports/modeling_scope_20261019/`，未重启已停止的视频/背景/旧公式长测试。

后续首轮固定参数预测，不能由 RGC 输出匹配自动标记内部层 validated，也未获授权自动训练/扫描。候选 [Wu 2024](https://www.nature.com/articles/s41467-024-52304-7)、[Latimer ON-parasol](https://figshare.com/articles/dataset/ON-Parasol_RGCs_for_the_conductance-based_encoding_model/9636854)、[Schottdorf–Lee](https://pmc.ncbi.nlm.nih.gov/articles/PMC8998785/) 的声明不等于完整数据合同。当前未接入解析完成的原始刺激/响应子集；还缺实际 recording/类型/区域，原始刺激数组与像素角度映射，刺激与 spikes 的同步/曝光边界，亮度/对比度/光谱或明确未标定映射，适应历史及训练/独立测试划分。AC 直接数据缺口保持明确。

修订前可回滚快照：`C:\Users\PC_User\AppData\Local\Temp\retina_scope_before_20261019_v1.zip`（含当时用户修改）；配合此前完整执行代码快照 `retina_dynamic_stopped_execution_20261019.zip`。没有 Git 提交、环境升级或历史结果删除。


### 已批准的 BC/AC 受体候选方程（2026-10-19）

聚合顺序复核：下面的 `f(Wu)` 目前仅保留为 synthetic 聚合示例，不能视为既定接点非线性 `Wf(u)` 的等价实现。非测试/生产参数绑定已由 `CandidatePoolingUnresolved` 阻止，填齐参数也不能绕过。待明确确认固定 W、共享 τ/驱动力及加权初态等条件下的接点先非线性聚合方案；当前没有新增逐接点独立时间状态。

用户 `Sentinel_27c9730906a08191998a122fcc59e601` 已确认具体结构迁移。本节是实际新增方程，不再只描述模块接口；只替换显式选择的 cone BC/generic AC 候选，不切换现有 GUI/CLI 默认，也不代表其余全细胞草案全部完成。

`bc_ac_receptor_components.py` 的 `CandidateConeBipolarReceptorMembrane` 复用原 `rod_pathway.membrane_component` 已有的受体/膜/释放实现，经独立 cone-BC 目标校验后调用；原 `rod_pathway.py`、`inner.py` 和编译 lowering 均未修改。五类 BC 的状态从 FIR history/activation 加终端电位/释放改为 **V、release、e**，不再用 `Vref + native_voltage_gain*(FIR-softplus - reference)`：

```text
u = frozen W @ native cone effective drive       # dimensionless, not glutamate concentration
occupied(u) = u^n / (K^n + u^n)
e_inf = 1-occupied(u) for ON; occupied(u) for OFF
de/dt = (e_inf-e)/tau_e
C*dV/dt = gL*(EL-V) + gE*e*(EE-V) + I_gap
           + g_AII*(EI-V)                       # last term only for OFF BC
 dr/dt = (sigmoid((V-release_half)/release_slope)-r)/tau_release
activation = r
```

`CandidateGenericAmacrineReceptorMembrane` 只有 **V、e** 两个连续状态，替换三类 generic AC 的 FIR/softplus history：

```text
u = frozen W @ BC effective terminal activation
de/dt = (u^n/(K^n+u^n)-e)/tau_e
C*dV/dt = gL*(EL-V) + gE*e*(EE-V)
activation = sigmoid((V-activation_half)/activation_slope)
```

AC 没有新增 Na/K、尖峰、Ca、释放资源或隐藏释放时间常数；输出是未标定的瞬时单调电压映射。三类仍是 generic 有效功能单元。所有输入权重、连接、gap 和 cell ID/顺序原样保留。输入汇聚顺序明确为 `frozen_connection_pool_then_single_effective_receptor_gate`：先保留原 W 汇聚，再作用于每细胞一个有效门控；不声明逐接点分子门控。C 使用 F、电导 S、电压 V、时间 s；u/e/release 无量纲，不隐式换算为 μM glutamate。

`configurations/bc_ac_receptor_candidate_v1.json` 是独立、未启用的候选参数合同。共 **103 项 value 为 null**：midget ON 13、midget OFF 14、diffuse ON 13、diffuse OFF 14、S-ON BC 13，每个 generic AC 12。BC 必需 C/gL/EL/V0、K/hill/tau_e/gE/EE/e0、release_half/release_slope/tau_release，OFF 另需 EI；AC 使用公共膜/受体项及 activation_half/activation_slope。来源、locator、适用条件亦须明确，不能把旧项目数值或 FIR 采样周期当生理绑定。首版仅允许 V0/e0 的逐细胞初态，其他系数按有效 family 共享，没有逐细胞自由拟合或优化器。

`CandidateParametersMissing` 在构建前列出目标和缺项；CLI 缺参正式运行返回 exit 2 的结构化 `BC_AC_CANDIDATE_UNBOUND_NO_RUN`。无旧方程回退。`test_only` 软件参数只能用于已有 synthetic 证据标签及 NUMERIC_FIXTURE；不能用该参数包运行/重标真实或 project population。当前 103 项均未获得所需同条件生理绑定，因此该候选仍不能作正式预测或生产默认。

实际入口：

- `migrate_bc_ac(parent, registry, candidate_profile, targets=...)` 保持图/细胞身份，只替换指定 BC/AC；`atlas_population_network(..., dynamics_candidate_path=...)` 提供整网候选接入。
- `build_module` 对新旧 BC/AC 均调用同一工厂，继续要求全部钳制端口和打开 gap 的显式声明。
- `tools/run_calibration_module.py --component midget_BC_ON --kind bc_clamped --mode declared_input_clamps --equation-route bc_ac_receptor_candidate_v1` 可查看缺项；参数来源齐备且接点/聚合语义明确确认后的候选版本，另需 `--candidate-profile`、实验 `--contract`、新输出目录及显式 `--run`。AC 使用 `--component local_feedforward_AC --kind generic_ac_clamped`。

新机制明确走 reference 路径，尚无候选 lowering；显式 numba 会拒绝不支持的新机制。旧默认仍是 HistoryBipolarTerminal/ResidualAC，34 组件显式 `numba_flat_rhs_v1` 构建成功，旧四个方程来源 pin 和三个 GPU 来源 guard 全匹配；默认两份配置逐字节未改。没有 GPU 神经推进、背景准备、视频或拟合。

32 项短检查通过，包括 synthetic 的 ON/OFF 门控方向、膜与释放 RHS、非线性/汇聚非交换性、BC 三状态和 AC 两状态、连接/ID/AII 不变、独立模块与整网同实现、BC/AC CLI 各 200 μs、未拟合参数包往返、缺参早停与旧后端检查。测试数值明确是 software_fixture，不作为生理默认；200 μs 仅为代码检查。证据见 `reports/modeling_scope_20261019/bc_ac_candidate_tests.xml` 及 `bc_ac_candidate_default_check.json`。

校准下一步是先为一个明确 cone BC 类型、物种、记录区域、温度及背景补齐输入标度和 source-bound 参数，再作小段固定参数电流/电压预测。释放参数需要独立的电压/释放约束；只有膜电位记录不能自动识别整条释放映射。缺直接 AC 数据时保持约束检查，不从 RGC 输出反推所有内部参数。Cone–HC、rod/RBC/AII、RGC 的原方程仍保留，其既有近似也保留。候选 BC/AC 新输出尺度下，旧下游 gain 仍未标定；旧背景/checkpoint 也不等于新结构的适应状态。

迁移前快照为 `C:\Users\PC_User\AppData\Local\Temp\retina_bc_ac_before_migration_20261019_v1.zip`；此前完整代码/交付快照与历史结果均保留。


### OFF midget/FMB 约束与实验输入准备（2026-10-19）

聚合顺序复核结论：现有候选 QA 算的是 `f(Wu)`，接点先非线性则是 `Wf(u)`，一般不等价。旧 cone BC 的原源码是 `atlas_runtime.py`/`full_target_network.py` 先经 W 到 drive，再进入 `CausalHistory`/终端；`atlas_primary_rod.py` 的 rod→RBC 则是 take 到独立 TRPM1 接点、门控后 pool。旧 cone FIR 的布局不能证明新接点门控应采用同样顺序。单一且权重为1的输入是特例，不能推广到 diffuse BC/generic AC。

`CandidatePoolingUnresolved` 现已阻止所有非 test-only 候选的来源/生产绑定；仅填齐103个参数也不能绕过。没有擅自增加接点独立时间状态。待确认的最小修正是固定 W、同一 postsynaptic cell 各接点共享 τ、同驱动力/线性电导加权且初态按同一 W 聚合时使用 `tau*de_i/dt=sum_j W_ij*f(u_j)-e_i`。这是条件性的聚合等价，不等于已证明真实接点共享动力学；当前尚未实施该改写。

`fmb_calibration_constraints_v1.json` 新建四组观测约束，全部 `runtime_parameter_binding=null`，没有改历史目录或向候选填值：

| 来源/约束 | 可使用的约束 | 禁止直接推成的参数 |
|---|---|---|
| Puthussery2013 FMB Rin=1.42±0.17 GΩ，n=19 | 总输入电阻；1/报告均值≈0.704 nS 只是逆均值，不是逐细胞电导均值 | 不是 gL，不识别 C；DB4 的 specific C/Rm/Ra 模型选择也不能借给 FMB |
| Puthussery2014 FMB puff plateau/peak=0.64±0.15，n=11 | 指定0.5mM、0.5s puff条件下的形状摘要；本次原文获取受阻，保留原目录“未完成本地原文核验”状态 | 不是 tau_e、Hill K/n 或最大兴奋电导 |
| McLaughlin2021 FMB 单事件 glycine | 0.97nS、−76mV；10–90% rise123–245μs；−100/0mV时decay2.1±.2/4±.2ms（n=20） | 不是整个 AII 通路的最大电导；rise不能直接当双指数tau_rise，也不是BC兴奋/释放tau |
| Dacey2000 Fig4 OFF midget | 已识别细胞的光响应图、约1000td/100% contrast/2.44Hz及空间刺激条件 | 图像不等于原始采样；绝对V、逐细胞物种/位置、精确同步等尚缺，不能直接校准整条链 |

2013 的 FMB 数值及实验条件已在 PMC 正文核读；2021 原始 JATS XML 和 Dacey PDF 已保存固定 hash。McLaughlin 条件为 mulatta、>8mm BC记录位置、低photopic、31–33°C，不能自动移到当前nominal RF=.35mm。Dacey PDF已渲染核对Fig4；纸面样本含多个物种，具体图中细胞的物种映射仍未解决。2014引用保留原来状态，不伪称新完成primary核验。原始论文/图与原始电流/电压数组明确分开。

`fmb_calibration.py` 提供只读约束目录、固定hash/明确秒单位/源记录区域/温度检查的单记录 NPZ intake，及 `input_resistance_from_steady_iv`、`plateau_peak_ratio`。观测摘要不能经 `runtime_parameter()` 自动变成系数。输入框架 `fmb_calibration_inputs_v1.json` 的四组真实数据入口仍为空：缺冻结原始数组或带误差的图像数字化、数组键、时钟/基线/钳制边界等。puff还缺受体实际接收波形与有效drive映射；Dacey还缺刺激响应同步与空间映射。当前四组均不能启动候选预测。另一个代码边界缺口是：BC模块当前只钳制drive/抑制输入，尚没有实验电压钳制指令与受体电流读出边界；即使取得puff/I–V数据，也不能把它们直接当现有模块输入运行。后续应包装同一门控方程补该实验边界，不凭此新增通道或系数。可用下列命令检查具体缺项，不运行模型：

```powershell
& C:\Users\PC_User\anaconda3\envs\retipath_v1\python.exe -B -X utf8 -m retina_sim.fmb_calibration
```

可辨识性检查：仅有电压时，在无独立绝对注入电流参考且全部相关电导/电流同缩放的条件下，C与全部g同比例不改变V(t)；u与K同比例不改变Hill门控。总Rin可约束绝对导纳组合，但不能分离所有漏/受体/gap分量。额外的同条件绝对电流或导纳、输入标度及独立释放观测仍需要。旧工程gL=1nS在被动单室、固定门控、其余负载非负的假设下给出Rin≤1GΩ，与报告中心值1.42GΩ存在条件性不兼容；这不是把0.704nS赋给gL的理由。

模型适用性：在固定V、实际受体drive恒定、初态平衡的OFF激活阶跃下，当前一个一阶门控只会单调趋近目标，不能产生早峰后下降。单独plateau/peak比还不足以判断，需明确peak/plateau时间窗及实际送达波形。没有为解释puff下降擅自加desensitization、释放资源或未知通道；修正聚合顺序也不会自动解决此限制。

17项针对性短测试通过，覆盖 I–V/plateau指标、单位/时间/hash/记录域、两组尺度对称性、Wf与fW差异/共享τ条件证明及生产聚合门禁。没有真实参数拟合，也未恢复旧视频/背景。具体 readiness 与来源状态见 `reports/modeling_scope_20261019/fmb_constraint_readiness.json`，测试见 `fmb_constraint_tests.xml`。下一步需明确接点先非线性的最小共享状态合同，并取得一条同条件 FMB 原始/明确数字化记录；不以输出匹配唯一识别内部机制。

### 接点先非线性候选与 FMB 图波形交付（2026-10-19，12:26 批准后）

本节取代上一节的“接点先非线性待确认”状态。用户明确批准先各输入非线性、后按原 W 汇总，以及共享条件下的单个聚合门控。新版本 `configurations/bc_ac_contact_first_candidate_v2.json` / `bc_ac_contact_first_candidate_v2` 与历史 `bc_ac_receptor_candidate_v1` 分开；后者保留原来的 QA 聚合顺序与生产绑定禁用，默认仍为原 HistoryBipolarTerminal/ResidualAC。

`contact_first_receptor.py` 给出共同的 Hill 函数和聚合检查，`bc_ac_contact_components.py` 是整网与独立模块共用的 v2 方程工厂，`bc_ac_contact_migration.py` 负责实际接入。每条原始 drive 路由前增加无状态计算节点，逐个原始突触前输入计算占据率，然后原 ConnectionTable 汇总。原表的细胞 ID、边 ID、权重、来源及归一化记录完整保留；额外 identity 传输边标记为 computational_identity_not_biological_contact，不代表新增细胞或生物接点，也不产生接点 ODE 状态。原始外部 drive 钳制对应明确的每细胞一个输入，不推断潜在接点。延迟输入的 prehistory 尚不在本版变换范围，非零延迟会明确拒绝。

令 `f(u)=u^n/(K^n+u^n)`，则 OFF BC/AC 为 `tau_i*de_i/dt=sum_j W_ij*f(u_j)-e_i`，ON BC 为 `tau_i*de_i/dt=sum_j W_ij-sum_j W_ij*f(u_j)-e_i`。只有原 W 行和确为1，ON目标才可写成 `1-Wf(u)`。本版不改变或重新归一化 W；初态 e0 的范围为 `[0,sum(W)]`，gE 的新语义为每单位加权门控的电导，整细胞最大兴奋电导是 `gE*sum(W)`。输入端口明确为 weighted_receptor_occupancy，不能将原始 u 或浓度冒充加权占据率。

从接点门控到上述聚合方程的精确条件是：W 固定非负、每个目标细胞各接点使用同一个固定 tau、相同突触后驱动力且电流对门控线性、初态满足 `e_i(0)=sum_j W_ij*m_ij(0)`。检查函数可核对 CSR 接点顺序的 tau 和初态映射；候选 profile 要求明确声明全部条件，并以显式加权 e0 初始化，不声称恢复了未知接点初态或旧稳态。共享输入变换还要求目标族的 K/n 相同；当前只允许这些系数族广播，未新增未测的逐接点自由度。异质接点 tau、变化 W 或驱动力不共享，不能凭此压缩。模型没有为此增加受体脱敏、Ca、资源池或主动 AC 通道。

独立 BC/AC 模块仍使用同一 v2 膜/释放/门控工厂；其 drive 边界钳制的是 Wf(u)，如需由原始输入建立边界，必须用 `aggregate_occupancy(W,u,K,hill)` 及原 W 做显式前处理。BC保留原一阶有效电压释放形式，AC保留瞬时有效电压输出。候选仅 Python reference 路径，显式 numba 拒绝未 lowering 的方程。旧 GPU/numba 源 pin 和默认两份配置的字节保持不变；旧 checkpoint/background 不能视为新状态布局的适应态。

可先只读检查新候选（不推进神经状态）：

```powershell
& C:\Users\PC_User\anaconda3\envs\retipath_v1\python.exe -B -X utf8 tools/run_calibration_module.py --component midget_BC_ON --kind bc_clamped --mode declared_input_clamps --equation-route bc_ac_contact_first_candidate_v2
```

v2 的103个数值项仍全部为空。结构批准不等于来源系数齐备或生理验证；提供有来源、条件和不确定性的参数后，才能经同一迁移入口绑定。现有实际配线、下游 gain、膜参数、释放和适应状态仍需同条件资料；本轮没有填入猜测数值或启动真实参数拟合。

Dacey2000 原始 PDF SHA256 `e1f9d8069fb6d9ed5c4aa91139323a37200d56a32c0e1186f05b9b5fa24dea27` 的PDF第7页/印刷页1807、Fig4B、细胞 A91199-6 已提取一个图示周期，保存于 `data/source_data/dacey2000_fig4B_digitized_v1/`。`tools/digitize_dacey2000_fmb.py` 用现有 Poppler 将该页渲染为2164×2800，再在固定ROI按连续暗线逐列中位数提取；不平滑、不插值。NPZ/CSV保留206列的页像素 x/y、相位、相对电压和误差界；provenance.json记录来源、图号、ROI、阈值、标尺、所选参考窗、方法及各文件冻结hash；digitization_check.png将所提取曲线覆盖在原图像素上核对。

垂直标尺为15mV，像素端点 y=217..390（跨度173px）。电压零点仅定义为 x=1440..1475 窗口的较低显示平台，不是静息电位。误差界合计可见线宽、半像素量化、阈值130/140/180变化、参考窗变化及标尺端点±1px，最大约1.164mV；它是保守数字化误差界，不是原实验SEM、CI或细胞间变化。横轴只按图示相同刺激边之间定义一个周期，并保留边定位误差。低阈值100–125会断线，已拒用而未插补。

需纠正前节 Dacey 摘要中2.44Hz的适用范围：图注明确将其用于C–E；本轮没有独立确认A/B。因此 Fig4B 数据只给 graphical phase_cycles，time_seconds 不存在，不能用2.44Hz补造时间。输入源形状、真实光刺激时间序列、绝对V、逐细胞物种/记录位点和光到有效drive映射未确定；1000td估计均值与100%对比度不能代替完整可回放输入。`fmb_calibration_inputs_v1.json` 已接入这份带hash/误差的数字化资料，但秒时间键/单位明确为空；`load_digitized_shape()`只返回相位域形状，而 `load_recording()` 仍拒绝将其当秒域实验运行。资料属于 figure_digitization，绝不称为原始记录。

44项针对性短回归通过，覆盖旧模块/参数快照/作用域、v1历史候选、v2实际Wf顺序、非等价反例、共享tau下接点与聚合轨迹、非单位行和ON公式、初态映射/不适用条件拒绝、细胞/生物配线/AII保留、同工厂模块及合成网络200微秒推进、原PDF数字化复算与缺失秒轴拒绝。测试只证明软件合同；没有生理校准。证据见 `reports/modeling_scope_20261019/contact_first_tests.xml`、`contact_first_delivery.json`。另对默认34组件做构造与显式 numba 选择检查，未推进神经状态、未准备背景、未打开旧视频；结果见 `contact_first_default_check.json`。


图4B来源标签补正：该图是150–1200 μm环形刺激诱发的 **annular_surround（周边）响应**，不是直接中心输入或孤立cone→BC传递。15 mV标尺校准的是ΔV，不能给出absolute V。Table 1中A91199-6的来源报告eccentricity为13.6 mm，鼻侧信息来自正文；现已记录在数字化provenance及FMB约束/输入元数据中，不擅自换算为平台functional RF或pedicle坐标。上一段“记录位点未确定”因此仅剩坐标映射缺口；逐细胞物种仍未知。A/B时间轴维持unconfirmed：方法DLP列2/4/10 Hz，Fig4E正文2 Hz与图注2.44 Hz不一，均不能替A/B补秒轴。该波形仅作为相位域周边ΔV形状资料，明确禁止孤立cone→BC直接校准；未新建刺激时间序列、未运行模拟。


整体一致性约束（用户2026-10-19补充）：局部拟合不得牺牲整网匹配。`calibration_modules.py` 的 system_compatibility 将父网各组件实现/参数、端口单位/语义/细胞顺序、绑定选项、人口几何、配线与反馈、已装配RGC目标纳入身份合同。新的参数包schema为module-parameters-1.1，回装会核对该完整上下文；接口尺度、背景绑定、反馈或周边参数不一致不能凭局部拟合结果回装。历史1.0身份快照仍可读取，但明确标记为无整网验收，不补造历史合同。当前回装仅允许原值身份快照，改动拟合系数仍需新的来源受限profile/scope修订。

局部模块与整网继续共用方程、参数和单位，实验钳制边界明确声明。更改接口尺度、输入背景、适应状态或反馈，必须先明确评估对上下游的影响。未来真实参数回装的验收需预先声明实验条件、指标和容差，检查同条件输入与重新准备的背景、上游层、改动层、下游层及全部已装配RGC；局部残差改善不是整网验收。若局部好而全局坏，不能任意调整其他层补偿。合同中的system_checks_status保持not_performed，system_acceptance=false；构造模块或匹配hash不等于完成适应或生理验证。本轮仅补合同与元数据，没有运行这些整网试验。

Dacey资料再补充：Fig4B仍为annular_surround、ΔV、150 μm内径/1200 μm外径环形刺激，相位轴未确认秒单位。来源报告100%对比度、约1000 td均值/背景与DLP CIE1931 xy=(0.304,0.349)，未据此补造完整光输入。原记录采样10 kHz、发表波形为周期平均；206个数字化列仅描述一条图线，不能当作206个独立实验样本，独立样本数保持未知，也不能用原采样率反推数字化点的秒间隔。论文含Macaca与Papio，A91199-6逐细胞物种未明，confirmed_macaque_claim=false，不标为已确认macaque。已有源文件、NPZ/CSV与图像均未重做。

### 数据缺口下的分组绑定提案（2026-10-19，只读分析，尚未实施）

建议保留当前完整有效整网作为可运行基线，另将v2按兼容参数组评估。缺乏数据允许明确标注的作者模型转移、共享假设或工程近似，但不把它们改称macaque测量，也不逐项拼接数字。本节只记录取舍；候选103项仍全空，方程、生产默认及参数均未修改，未拟合或模拟。

参数账本以 `bc_ac_contact_first_candidate_v2.json` 为准：5个BC族有67槽，3个AC族有36槽；ON midget/diffuse/S各13，OFF midget/diffuse各14，AC各12。总计103槽对应16种字段含义，其中V0/e0各重复8次，是16个初态槽，剩下87个系数槽、14种系数含义。当前每族内部已按broadcast共享各细胞系数；跨族同名槽并未建立参数别名，因此不能说系统实际上只有16个独立系数。初态是条件相关状态，不能作为16项额外稳态生理常数拟合；后续须从同条件背景得到，或明确声明初态非适应。

| 按含义分组 | 现有槽数 | 绑定与后续自由度建议（本轮均为0项拟合） |
|---|---:|---|
| 被动膜 C/gL/EL | 24 | 成组保留C、漏电导和电压锚；无绝对参考时先讨论tau_m=C/gL及电压工作点，不同时自由拟合所有量。膜面积换算必须有匹配形态，DB4模型的比电容不能直接给FMB总C。 |
| 有效受体 K/hill/tau_e | 24 | K/hill须对应同一个有效输入标度与剂量曲线；缺该曲线先不开K/hill自由度，只有真实秒轴/已知输入后才讨论每动力学组1个tau_e。ON、OFF与AC不可因同名而直接共用受体常数。 |
| 兴奋电导标度 gE/EE | 16 | 固定反转条件后可讨论每组1个rho=gE/gL；绝对gE需要匹配电流/导纳及W定义。v2整细胞最大值为gE*sum(W)，不能直接复制源模型整细胞gmax。 |
| BC有效释放 half/slope/tau_release | 15 | 需一个完整、同条件V到有效释放参数组；没有独立释放资料时不从RGC输出同时放开这3项和下游gain。Maddox锥Ca激活曲线不是BC释放。 |
| OFF抑制反转 EI | 2 | McLaughlin FMB/DB1是特定切片、氯条件下的来源约束；全区域FMB及混合diffuse OFF不能无条件套用。不是待任意拟合的补偿旋钮。 |
| AC有效输出 half/slope | 6 | 三通用AC的生物亚型未定；属于有效V到活动读出，需要同输出尺度的完整组，不能从不同细胞的sigmoid各捡一个数。 |
| 初态 V0/e0 | 16 | 不作可转移系数；e0是加权聚合且范围随W行和变化，旧单位门控不能直接当新稳态。 |

可辨识边界：隔离、无独立绝对电流参考且所有相关电导/电流同时缩放时，C与g整体缩放不改变V(t)，因此电压主要约束g/C与电导比例；整网固定绝对gap/抑制电流可打破这一对称性，但工程电流数值本身不构成生理绝对标定。若u标度未固定，u与K同比缩放不改变Hill门控；若原上游u的源标度已固定，则K可在该有效单位下估计，不能改称浓度亲和力。仅ΔV也缺绝对电位锚。受体、膜、释放的多个时间尺度，以及释放读出和下游gain，从单个RGC输出可能混淆。故没有一种不依赖实验边界的“真实可辨识参数总数”。需要物理解释的C/电导、EL/释放阈值等绝对V参数、浓度K及绝对释放率，分别需要绝对电流/面积或导纳、绝对V、浓度到输入映射、释放读出标定。

共享预算可以讨论，但必须注明假设。最先可讨论的是三通用AC共享9个系数、保留各自tau_e，共12个系数替代30个AC系数，整网87降为69；这是工程共享提案，尤其widefield旧采样间隔不同，不能声称动力学等价。若进一步把LM midget/diffuse按ON、OFF分别共享整组（各11、12），S_ON保留11，AC按上述12，则总预算为46个系数；这需要新增跨细胞类型假设，当前没有证据可直接批准，非已完成减参。不可将S_ON自动绑到LM ON，也不能把generic AC与AII绑定。若这些分组及其余成组固定项均获批准，首轮只考虑tau_m、tau_e、rho：3个BC动力学组各3项，加上generic AC共享tau_m/rho及3个独立tau_e，共最多14个有效拟合自由度；这不是声称14项足以标定全部参数，缺输入/来源时实际开放数仍为0。

| 已核查来源/模型组 | 证据级别与兼容结论 | 可执行取舍 |
|---|---|---|
| 当前effective_dynamics_v1整网；历史membrane_entries cone_BC原型 | project_approximation。原型含与新BC被动膜/Hill/一阶释放相容的完整数值组，但不是当前生产BC方程，也不是测量。其旧漏电导对应的被动Rin上界与FMB报告中心值有条件性差异。 | 首选继续冻结当前整网（新增自由度0）。如用户接受明确工程候选，下一轮只从一个完整BC原型组转移，保留来源标签，另核W/gE尺度与初态，不能称生理校准。新AC没有同等完整已核实的原型组。 |
| COREM官方SingleCompartment | 作者单室电导方程与v2膜电流形式相容；官方例子是typical values，非macaque BC测量。其滤波/静态非线性门控并不自动等同当前Hill、共享tau及BC释放组。 | 可作为整组膜接口/参数包候选；只在同一来源包给出C、Rm、反转/输入单位时使用gL=1/Rm的明确换算。不借示例值填满受体或释放空缺。 |
| Somaratna/Freeman 5406d153；Vanni 08aea380官方快照 | 本地hash核对通过。前者BC是共享一阶有符号滤波；后者含功能subunit与手动尺度设置，并非当前BC/AC物理门控包；Vanni快照为作者预印本/代码。 | 可整组保留为功能参考，不能把一个总滤波tau改称当前tau_e，或用功能subunit空间参数填C/gL。 |
| Puthussery FMB Rin、Dacey环形ΔV、McLaughlin FMB/DB1抑制记录 | 实验观测/图线约束，有明确记录条件，不能逐条冒充运行系数；Dacey只相位，物种未确认。Maddox是cone而非BC释放，DB4被动模型也不是FMB测量。 | 保留为验收/条件约束，当前不构成完整可借参数组。Rin非1/gL的直接测量，单事件峰非总gmax。 |
| retsim/其他物种整细胞模型 | 作者模型转移只有在方程、细胞类型、温度、面积、电流/释放单位兼容时才成立。父线程已核验retsim少量自由参数及Rm/Cm相关性原则；本轮官方手册访问失败，未获取可冻结的完整donor包。 | 待核实候选，不能登记成已就绪或已确认macaque参数。鼠/大鼠AII或RBC不直接代替本项目generic AC、cone BC。 |

COREM同样允许拟合/有效参数，而不要求每项直接测量：其作者稿Appendix A/B使用优化与best-fit参数（父线程全文核验；本执行器PDF抓取超时），官方单室文档则明确采用滤波与静态非线性近似门控。这支持“明确目标、限制自由度、保留模型身份”的做法，不支持跨方程照搬数字。参考：[COREM单室模型](https://github.com/pablomc88/COREM/wiki/Single-compartment-model)、[作者稿Appendix A/B](https://digibug.ugr.es/bitstream/handle/10481/110810/manuscript.pdf?isAllowed=y&sequence=1)、[retsim官方手册入口](https://retina.anatomy.upenn.edu/~rob/ncman6.html)。本地来源见 sources/author_asset_manifest.json、data/source_snapshots/vanni2026_08aea380/frozen_source_manifest.json、sources/primary_pathway_effective_profile.json；来源身份核对不是生理验证。

全系统影响必须随组记录：midget ON/OFF BC主要影响各自midget RGC；diffuse ON/OFF BC既影响对应parasol兴奋，也经generic AC改变midget与S-ON通路，diffuse OFF还进入small_bistratified_S_ON；S_ON BC直接影响该S-ON RGC。local_feedforward_AC影响midget OFF和S-ON，local_crossover_AC影响midget ON，widefield_AC同时影响两极parasol；AII与ON BC电耦合还会传播变化。局部好不能靠任意改其他层gain补偿，回装要遵守现有system_compatibility合同并分别检查全部5个RGC目标及上下游层。

后续待用户取舍只有三类：是否接受“完整工程近似参数组”供v2试运行（而非等待全测量）；是否批准哪些跨族共享（先generic AC，还是继续各族独立，不能默认启用46项方案）；若选外部作者donor，是否接受其跨物种/实验域限制及为少量有效组合预先规定拟合目标、范围与整网验收。当前可执行基线已经存在，无需为了单细胞参数缺口停止整个平台；本轮不选择新donor、不填写103空值，也不启动拟合。

### 获批的完整假设基线绑定（2026-10-19，参数证据标准获用户同意后）

用户已同意有依据的借用、明确有效假设与受约束拟合，无须每项都是猕猴直接测量；仍禁止新增机制、合并细胞类型、偷偷切换默认或用局部拟合破坏整网。本轮据此新增 `configurations/bc_ac_contact_first_hypothesis_v1.json`，沿用既有v2方程及迁移入口。5个BC和3个generic AC各有独立profile/参数记录；没有启用69/46项共享提案，也没有参数别名。相同起始先验值不构成跨族联动约束。原103空值模板与生产默认保持原字节。

103项运行时绑定齐备：71项显式数值、32项按明确背景和既有接口参考推导（每族K、V0、e0及输出half）。按证据含义为9项borrowed、69项derived、25项effective_assumption；derived包括由明确假设计算的数值，不代表测得。没有measured或fitted项，运行时保留原project_approximation证据门禁，并将细分分类、理由、适用条件、来源、敏感性写入profile conditions及源表。若声称measured，仍须通过原有primary measurement门禁；不能仅修改标签。当前own source是有固定hash的项目假设表，绝不把测试数值登记成生理来源。

膜尺度参考[COREM官方同形式单室示例](https://github.com/pablomc88/COREM/wiki/Single-compartment-model)：该示例C=0.25nF、Rm=100MOhm、EL=-70mV，是作者typical example，不是macaque BC测量。本基线显式选择0.05的C/漏电导共同尺度因子，得到C=12.5pF、gL=0.5nS及25ms漏时间尺度；2GOhm漏阻不会单独造成旧1GOhm上界与FMB 1.42GOhm报告中心值的直接冲突。这是粗粒度工程尺度选择，不是拟合该报告，也不能保证含突触/gap后的总Rin匹配。每族均独立记录这些假设。

其余选择均可追查：Hill指数1是不额外假定协同作用的最小饱和情形；K是指定背景下正的原始突触前有效输入中位数，不能叫浓度亲和力。gE由名义门控0.5、EL=-70mV、名义目标-55mV、EE=0的被动平衡设计推导，实际电流仍使用原W及真实行和。ON tau_e取漏时间的一半、OFF/AC取十分之一，tau_release取五分之一；这些仅是已声明的时间尺度分离假设，不是测得的亚型动力学。输出斜率由假设的10%–90%有效输出跨度20mV推导，half按同背景下保留接口的活动参考推导，目的是保持输出单位/工作点定义，不能当作释放实验。midget OFF的EI借用FMB切片-76mV约束并明示区域/氯条件转移不确定性；diffuse OFF独立采用-75mV有效参考，不称DB1或混合diffuse测量。

各项附带的0.5–2倍或电位±5mV范围是建议工程敏感性域，非置信区间，本轮没有据此做海量扫描。C和全部电导整体缩放的电压不变性只在相关外部电流也同比缩放等严格条件下成立；现有绝对gap/抑制输入保持不动，因此不能靠单位变换宣称整网不变。u/K标度由固定参考背景明确，不能同时自由改变u增益和K。改变背景必须显式修订该假设profile，不自动重设K来维持输出。首轮最多只预留midget_BC_OFF.tau_e和diffuse_BC_OFF.tau_e两项独立自由度；其余固定，本轮实际拟合数0。没有匹配输入、秒轴和观测时不启动优化，也不把其他层gain作为补偿旋钮。

指定背景为独立consumer参考100td native、1000R*/s cone、5R*/s rod，不声称它们经过共同物理场景标定。`bc_ac_hypothesis.py` 先用实际输入求outer source平衡，再形成带背景hash、单位和原输入/输出接口参考的条件V0/e0种子。真正运行时使用 `prepare_feedback_initial_state` 对完整上游确定性反馈闭包求初始状态：原模型方程、所有生物连接与双向电耦合均不删减；外层源平衡固定，候选膜/门控/释放、RBC/AII、化学突触及有关physical反馈共同解。它是初始状态求解，不是参数拟合；返回的状态直接交给同一runner，不能把单独compiled.initialize()的条件种子当作已解平衡。V0/e0不作可转移生理常数，RGC历史/随机spike平稳性与整网充分适应仍未验证。求解有显式未知量、迭代和墙钟预算，超限拒绝，不剪反馈或接受部分结果。

本轮系统证据分清两种范围：

- **小完整反馈工程图**：64-cone/64-rod显式预览构造，27个人口表、1070条工程cell记录、51组件、2508状态标量，保留原51条生物连接表和4组gap表；没有声称source density几何。背景闭包1427个未知状态在13次solver nfev后收敛，最大缩放残差8.735e-10/s。随后原网推进1ms，五个RGC电压记录均有限，未调其他层系数。它只证明数值/接口接回，不能凭1ms证明生理响应或适应完成。
- **实际冻结几何接回**：通过原 `atlas_population_network(..., dynamics_candidate_path=...)` 绑定，42组件（含8个无状态计算节点）、原23个人口表、20422条原cell记录及全部五个RGC目标保留。只做构造，没有大网神经推进或整网平衡求解，不把小工程图检查冒充大source图验证。

24项短回归通过，随后6项最终门禁检查通过；覆盖完整分类/背景推导绑定、不可伪称测量、原连接保留、含AII双向gap的初态求解、接点先非线性、原v1/v2与FMB资料限制。Dacey仍是annular_surround的相位域ΔV图线，未赋秒轴、未当孤立BC输入，未做真实参数校准。未恢复旧长视频。

复现入口（显式选择候选，默认不切换）：

```powershell
& C:\Users\PC_User\anaconda3\envs\retipath_v1\python.exe -B -X utf8 tools/run_bc_ac_hypothesis_baseline.py --output <新的输出目录>
# 只解析全部参数，不求背景或推进：在上命令增加 --bind-only
```

生成器 `tools/build_bc_ac_hypothesis_profile.py` 保存每项选择依据；已存完整解析值见 `reports/modeling_scope_20261019/hypothesis_bindings_v1/parameters_resolved.json`，其中V0/e0为条件种子而非可移植平衡常数。系统结果见 `hypothesis_baseline_v1/result.json`、背景残差见同目录background.json，实际几何构造见 `hypothesis_actual_geometry_binding.json`；总收据为 `hypothesis_delivery.json`。回滚前快照 `C:\Users\PC_User\AppData\Local\Temp\retina_hypothesis_before_20261019_v1.zip` 保存415个执行/配置/测试/文档文件。已逐字节核对生产两份默认、空白v2模板、既有v2方程文件及原四个核心源pin不变。现阶段可交付的是可运行、可追溯的假设基线；主要科学缺口仍是同条件输入/时间/电流或绝对V/释放资料及整网长时适应，后续须另按实际数据范围安排，不能升级为“生理校准完成”。

### 固定假设参数的短脉冲/恢复测试（2026-10-19，工程反馈网）

本轮只新增试验驱动与只读分析脚本，固定全部参数/方程/连接和生产默认，不做拟合、共享或补偿。沿用1070条工程cell记录、51组件、2508状态标量的完整工程反馈图，原51条生物连接表及4组电耦合表保留并核hash；不是实际冻结几何的大网验证，也未运行后者的长任务。

协议：先重建原确定性上游背景初态，再用原runner做100ms短背景洗入，不降低或宣称通过正式充分适应准则。将该时点的完整状态、history、待处理事件、采样时钟及RNG序列化为同一prepared_state.json，三分支restore后hash一致。分支分别为恒定背景、+10%和-10%模型输入尺度脉冲；刺激50ms，恢复150ms，绝对时间100..300ms，相对时间0..200ms。native_light按自身td作100→110/90，physical_cone_light按自身R*/s作1000→1100/900，rod_light按自身R*/s作5→5.5/4.5。它们是独立consumer单位下同日程相对扰动，不是同一物理量或已标定视觉对比度，不推断共同光谱/场景。K与全部参数固定，脉冲期间没有重新绑定。

记录每1ms一个时点，每个完整分支201点，保留全部细胞行；BC记录V、加权门控、activation与同值的有效release，AC记录V、加权门控及瞬时有效输出（不伪造AC释放机制），五个RGC记录V、gE/gI、conditional_rate、spike_count与window_rate。recorded_layout.json经同一assembly/检查点校验，保存各端口cell顺序和单位。所有完整分支记录有限，所有方向/恢复均以同一时点pulse减background计算，不把初始化漂移当刺激。

初始化检查：RGC未参与上游平衡求解，短洗入100ms内平均V从原factory初态移动7.56–20.02mV；最后25ms仍移动0.169–0.256mV。随后背景对照200ms还有0.0798–0.1156mV漂移，超过部分微小脉冲效应，故匹配对照不可省略。100ms不是充分慢适应或随机平稳性证明。AC工作点相对原输出参考0.554355：local_feedforward为0.609191，local_crossover为0.470227，widefield为0.600876；短洗入后基本不变。这些由新反馈初态引起的偏移原样保留，未通过其他层gain/half补偿。

下表为逐细胞均值、匹配背景扣除后的有符号电压峰值；峰值可出现在脉冲结束后。残留是在相对200ms（150ms恢复结束）的差值，百分比为绝对残留/该分支绝对峰值，不能称完全恢复。

| RGC通路（cell行数） | +10%峰值 μV | -10%峰值 μV | +10%末端残留 μV（占峰值） | -10%末端残留 μV（占峰值） |
|---|---:|---:|---:|---:|
| midget ON（59） | +41.91 | -45.84 | +3.53（8.4%） | -4.12（9.0%） |
| midget OFF（59） | -50.07 | +55.15 | -7.13（14.2%） | +8.04（14.6%） |
| parasol ON（6） | +32.04 | -39.49 | +4.11（12.8%） | -4.82（12.2%） |
| parasol OFF（6） | -53.92 | +78.14 | -10.86（20.1%） | +14.06（18.0%） |
| small_bistratified_S_ON（5） | -10.29 | +10.75 | -2.03（19.7%） | +1.88（17.5%） |

50ms时正脉冲使5个BC中3个ON去极化、2个OFF超极化：midget ON/OFF分别+46.21/-67.87μV，diffuse ON/OFF分别+46.09/-66.52μV，S_ON BC为+45.21μV；其门控和有效释放同向变化。三AC分别+10.79、-22.24、+11.69μV，负脉冲方向反转。全部时点的门控/输出与逐细胞轨迹保存在NPZ、mean_time_courses_SI.csv和BC_AC_matched_control图中，不用仅一个末端值判断响应。

S-ON RGC的正脉冲净超极化需要明确保留：在50ms，尽管S_ON BC释放增加，该RGC总gE下降0.000564nS、gI增加0.000284nS；它仍含原有diffuse OFF兴奋及ON驱动generic AC抑制，不能只由“S-ON”名称推断全场协调consumer刺激的净符号。这里只描述本模型已有混合输入，未做消融/调参或声称验证颜色通路。

spike输出没有被电压方向替代。对照/+10%/-10%的200ms总观察spike计数依次为：midget ON 3401/3405/3396，midget OFF 2925/2920/2925，parasol ON 360/361/360，parasol OFF 268/268/269，S-ON 391/391/391。对照每cell平均观察率分别约288.2、247.9、300.0、223.3、391.0Hz，尚未以真实数据标定，不能将这些高的工程工作点视为生理合理或归因到某一个新模块。conditional_rate包含各自strict-past spike history，脉冲后事件可分叉；例如+10%在50ms的midget ON电压上升24.69μV，条件强度均值却比对照低1.54Hz。保留该现象，不以单时点强度或单seed几次计数差异作统计/生理验证，也未修改RGC/其他层来降低基线。

实际耗时：首次构造加背景根求解约61.39s（其中根求解46.35s、16343次RHS评估），短洗入32.46s。三个成功200ms分支各77.37/78.87/79.71s。记录驱动曾在分块端点因一个ULP的请求时间越界退出，保留了75ms不完整对照；只将采样时间戳在原1e-12s容差内对齐端点，最大调整2.776e-17s，不改积分端点、状态数值、步长、参数或反馈。复用了原准备检查点，未重做根求解/洗入；该失败对照消耗约28.42s，成功恢复进程含重建开销251.54s。可核算模型计算合计约373.81s，含该次重试；只读布局校验与绘图另计。core始终用retipath_v1；图用已有Anaconda base的Matplotlib读取结果，未安装/升级任何环境。

交付目录 `reports/modeling_scope_20261019/hypothesis_pulses_v1/`：result.json含原始指标、输入合同、漂移和各层结果；verification.json含参数/核心文件字节核对、同起点及耗时；RGC_summary.csv/json含全部通路、符号/恢复/计数；mean_time_courses_SI.csv含所有层逐ms均值及对照差；各完整分支traces.npz保存逐cell原始记录。成功背景目录为background_retry_01；原不完整background目录仍保留，不能当成功对照。图为RGC_matched_control.png/svg、BC_AC_matched_control.png/svg、initialization_drift.png；RGC图右列是单seed条件强度，历史spike造成的跳变未平滑。root_state.json和prepared_state.json保留实际初态，复现入口为 `tools/run_bc_ac_hypothesis_pulses.py`；只读分析入口为 `tools/analyze_bc_ac_hypothesis_pulses.py`。

本轮结论限定为：完整工程反馈网在固定假设下有有限、方向可追踪的短刺激响应；主ON/OFF通路与补充S-ON的混合输入区别、初始化漂移、高放电基线及未完全恢复均已显露。没有证明生理校准、慢适应、spike统计准确性或实际冻结几何的运行表现，也没有因这些不足擅自调参。

### 高背景、接口与S-ON的只读分解诊断（2026-10-19）

本轮仅用既有轨迹、检查点、旧工程图工厂端口及原公式分析；只重建旧小工程图取静态参数/配线，没有任何神经advance，没有改RGC公式、参数、机制、gain或生产默认，也没有重跑三分支。完整证据为 `reports/modeling_scope_20261019/hypothesis_diagnosis_v2/diagnosis.json`、baseline_decomposition.csv、verification.json和algebraic_expectations_and_routes.npz；复现脚本 `tools/diagnose_bc_ac_hypothesis_baseline.py`。下列“旧参考”特指固定旧工厂输出的条件参考，不能冒充旧完整反馈网已适应后的运行结果。

**旧RGC工作点已带有高背景倾向。** 五个RGC族保留相同的工程强度曲线：alpha=150Hz、mu=-70mV、beta=5mV，strict-past kernel=[-12,-8,-4]mV，bin=1ms。原式为 `lambda=alpha*softplus((V+kernel@past_counts-mu)/beta)`。在无突触输入的漏平衡EL=-65mV、零历史时，入参已为1，lambda=196.989Hz；mu只是softplus偏置，不是零放电阈值，在V=mu时仍为150*ln(2)=103.972Hz。实际采样是每bin Bernoulli `p=1-exp(-lambda*dt)`、最多一个event，196.989Hz hazard对应178.801Hz无历史计数率，并无Hz/ms换算错误。

把原二值三bin历史表示为8状态概率链，可纯代数求其静态分布：V固定在EL且保留原负history时，计数率期望仍为135.382Hz。该概率链只是原采样公式的数学表示，不是新增机制或替代运行器。即使不使用新BC/AC，高背景倾向已经存在；不能将本次223–391Hz全部归咎于新候选或随机初态，也不能把这些代数值称为生理自发率。

旧工程BC工厂有效释放为0.5，三个旧AC的softplus(-0.3)=0.554355。经原有正权重/gain转换，这些正基线已产生持续兴奋/抑制输入。固定这些旧工厂输出作参照，按RGC膜平衡和同一8状态采样公式计算，可得下表；右侧新值由既有新候选背景轨迹及同一公式得到，没有拟合。

| RGC通路 | 旧工厂保持输入的V平衡 mV | 旧条件计数率期望 Hz | 新背景V平衡 mV | 新静态计数率期望 Hz | 新单seed观察率 Hz |
|---|---:|---:|---:|---:|---:|
| midget ON | -54.281 | 263.97 | -52.534 | 285.53 | 288.22 |
| midget OFF | -54.281 | 263.97 | -55.337 | 251.37 | 247.88 |
| parasol ON | -54.281 | 263.97 | -51.748 | 295.24 | 300.00 |
| parasol OFF | -54.281 | 263.97 | -57.356 | 226.97 | 223.33 |
| S-ON | -45.857 | 367.50 | -44.898 | 379.06 | 391.00 |

这证明原RGC参数与旧正输出参考足以形成高计数工作点，新背景数据也符合当前实现的数学预期；不是对真实放电率的验证。旧BC/AII/其他反馈状态尚未在本轮求同背景稳态，所以表中旧条件参考不提供旧整网最终观察Hz，也不能量化新候选的完整因果增量。

**新候选确有独立的接口工作点偏移。** 原五个BC工厂release均0.5，新闭环背景的midget ON/OFF均值为0.58875/0.46375，diffuse ON/OFF为0.58927/0.46578，S_ON为0.55106。旧AC均0.55436，新local feedforward/crossover/widefield为0.60919/0.47023/0.60088。旧AC是无上界softplus有效活动，新AC是有界sigmoid有效输出；同为dimensionless、连接/gain不变不表示基线和动态尺度已校准。背景推导half匹配的是条件种子参考，不保证经完整反馈求解后仍保持原输出参考。

按固定RGC本征参数、逐cell电导平衡做对称两种次序的代数分解，新平衡相对旧工厂参考的平均V偏移分别为+1.747、-1.056、+2.533、-3.075、+0.959mV；其中兴奋输入变化贡献约+1.649/-0.916/+2.712/-2.908/+1.083mV，抑制变化贡献约+0.098/-0.140/-0.179/-0.167/-0.125mV。两部分逐cell相加等于总差。它是条件参考下的电流账本，不是消融或新旧完整稳态的因果估计；说明不能仅修旧RGC高背景而忽视新接口变化，也不能靠下游gain抵消它们。

**大初始化漂移是单独的初态问题。** 本工程构造器把RGC V0设为-65mV，没有纳入新上游背景根求解；100ms后RGC才接近平衡。既有wash-in/control期间gE/gI几乎恒定，最大变化不足5e-22S。原被动膜闭式解 `V(t)=Vinf+(V0-Vinf)*exp(-(gL+gE+gI)*t/C)` 重构所有RGC轨迹，最大误差8.81e-15V；有效tau均值约18.12–23.06ms。故7.6–20.0mV漂移及后续小漂移完全可由未配平V0松弛解释，不是spike/RNG驱动；原模型明确没有spike到graded V的反馈。实际冻结几何在atlas_primary_rod另有按初始电导推导V0的路径，不能把本工程图漂移量直接外推到它。把RGC V0配平只消除人为瞬态，不降低其既定稳态hazard。

**history/随机实现影响单次输出符号，不是高背景的全部来源。** 在既有逐ms V上，条件于保存的prepared history及已抽取的首bin，可用原Bernoulli三bin转移算全部未来噪声的精确概率期望，无需再推进神经网络。正脉冲50ms时，midget ON的期望hazard差为+0.449Hz，而单seed记录是-1.543Hz；该分支电压差为+24.69μV。两者区别来自后续历史spike实现，不能据单次条件强度反号判电压/通路符号错误。200ms期望计数差依次为+2.402/-3.011/+0.226/-0.437/-0.036，而一次观察差为+4/-5/+1/0/0；这不是拟合或spike统计验证。静态/有限时间概率结果与高背景观察同量级，只证明实现账本自洽。

**S-ON反向首先涉及协议及旧工程配线。** 三个consumer的td、physical cone R*/s、rod R*/s被同时按自身单位乘1.1或0.9，既不是光谱标定的白光，也不是S与L/M独立、相等的cone contrast。旧工程图S_ON_BC的5个目标只接S输入，但diffuse_BC_OFF并非纯L/M：15个目标中6个有S权重，S归一化权重均值6.61%、单目标最高32.04%。该混合在候选之前存在；先前提案中的“LM组”不能当作工程配线已经排除S的证明，更不能据此自动共享或改线。

将原SynapticConductance与实际稀疏矩阵重新应用于已存逐cell活动，S-ON每一时点的分项电导与原记录总和在1e-22S容差内一致。正脉冲50ms：S_ON_BC兴奋项增加0.002292nS，diffuse OFF兴奋项减少0.002856nS，净兴奋减少0.000564nS；local feedforward抑制增加0.000284nS。权重/gain均非负，净超极化可由这些已有混合输入解释，未发现为了强制某种“S-ON亮度符号”所需的正负号修正证据。它不是已经验证的纯S-vs-LM生理颜色通路，也不能因标签为ON就按亮度响应判错。

**原指标及重试定义已复核。** 峰值是先逐时刻对cell差值取均值，再在相对0..200ms选绝对值最大的时点并保留其符号；允许峰值在50ms脉冲结束之后。恢复量是 `abs(mean_delta(200ms))/max_t(abs(mean_delta(t)))`，是端点残留/峰值，不是恢复率、时间常数或单调恢复证据。原10组有符号峰值及比值逐项重算一致。8%–20%是通路均值指标，不能说每个cell都回到该范围：逐cell末端/自身峰值最高约36.7%（S-ON正脉冲），所有详细范围已附JSON。S-ON还可过零后反弹，不能用一个端点拟合单指数恢复。

记录重试是请求时间戳超分块端点一个ULP，最大对齐2.776e-17s，模型状态/积分端点/参数未修改。原不完整对照的前三chunk（75ms）与成功重试对应chunk所有数组完全相等，包括spike计数；不是用重试掩盖数值不稳定。原脉冲数据、代码、模型和生产默认均逐字节核对未变。

**最小下一步提案，尚未实施：** 首先把RGC确定性V初态按同一已解背景的gE/gI代数配平，保留原强度曲线、history/RNG与全部gain；这是初态修正，不是降低高稳态输出的补偿。然后只补一个旧完整反馈网的匹配背景平衡参考（或其常输入代数固定点），才能严格归因新旧稳态差异，无需重跑整个脉冲套件。涉及真实工作点/放电率的参数修订，应先给出同条件绝对V/电导及spike观测约束，而非任意移动mu或压低下游gain；S问题先明确旧diffuse的S混合及分离consumer/S/LM模型域输入的协议边界，不因净符号直接改线或增机制。本轮仅完成诊断，不据此启动拟合。


### RGC背景初态配平与旧完整反馈背景参照（2026-10-19）

本节实施上节已授权的最小下一步。修正仅进入显式候选/背景参照准备函数，生产GUI/CLI默认与两份默认配置均逐字节保留；未重跑脉冲、训练、拟合或大型冻结几何。回滚为 `C:\Users\PC_User\AppData\Local\Temp\retina_rgc_initialization_before_20261019_v1.zip`。旧脉冲、诊断及来源资料均保留。

**RGC确定性初态。** `retina_sim/bc_ac_hypothesis.py::balance_rgc_initial_voltages` 在尚未advance、未抽取首bin的背景状态上，读取同一完整网络实际背景电导，设置 `V=(gL*EL+gE*EE+gI*EI)/(gL+gE+gI)`。只改变状态V，并按原intensity函数重算与该V一致的conditional-rate缓存。原参数V0、膜方程、alpha/mu/beta、strict-past history、RNG、事件、bin计数及全部时钟不变；不生成新spike，不初始化为随机平稳分布。已有抽样或推进状态拒绝此操作，避免改写既定样本。两条路径配平后的最大RGC膜导数均低于5.6e-16 V/s。

**旧完整反馈背景。** 显式 `retained_reference=True` 保留旧 `HistoryBipolarTerminal/ResidualAC` 原方程。在恒输入背景把有限FIR prehistory置于 `history[j]=drive-baseline`，activation仍由原kernel、bias、polarity和softplus计算；这是旧采样状态的固定点初态，不修改profile中的历史参数。所有连续反馈状态联合求根，包括ON BC/AII电耦合、rod/RBC/AII、OFF抑制与physical HC。8组旧BC/AC最终逐项调用原始native.transition验证固定点，误差均0；未冻结工厂BC=0.5、AC=0.554355输出。1238个未知量、14次外层求解、15668次残差调用、53.43s；最大缩放残差1.1771e-10/s。候选上游复用此前经过身份和背景hash核对的1427未知量解，原残差8.7351e-10/s，再配平RGC，无重复求解或重新拟合。

**匹配观测及适用域。** 两条路径均为相同1070工程细胞记录、27 populations、同样51张生物连接表与4张gap表。旧39组件/3003标量状态；候选51组件/2508状态，其中新增节点仅为静态计算适配。连接表权重、ID、两端细胞及gap hash一致；信号source组件经过显式contact-first适配，不能以整个PortLink完全相等来判断接触是否丢失。比较工具首版这一元数据判断已更正，`connection_identity.json/metadata_verification.json`保留依据，数值数据未改。

两条路径输入均为native 100td、physical cone 1000R*/s、rod 5R*/s，仍是独立consumer域背景，不是已校准的共同物理场景。同为100ms短洗入+200ms恒背景观测、1ms记录、原DOP853容差与步长、相同各RGC seed；在0/100/300ms RNG状态hash逐一相等，history与事件可因电压不同而不同。完整流程本次计时284.15s。旧/候选洗入所有.voltage端口最大变化分别3.50e-15/1.31e-14 V，观测最大变化1.71e-15/7.10e-15 V；RGC洗入最大2.44e-15/4.88e-15 V。确定性固定点及此短窗稳定有直接证据，100ms准备并未满足独立正式背景策略至少1.25s的工程下限，也未验证随机平稳性或真实生理适应。

各层实际背景有效接口均值如下。它们是保留接口上的模型有效量；旧AC softplus与候选sigmoid的数值不能当作已校准的生物释放概率。全部原有组件输出及单位、V/电导/候选gate等详见 `layer_background_comparison.csv`；旧AC无膜V的地方保持缺失，未补造。

| 有效接口 | 旧完整反馈背景 | 候选背景 |
|---|---:|---:|
| midget_BC_ON | 0.483140 | 0.588749 |
| midget_BC_OFF | 0.414687 | 0.463748 |
| diffuse_BC_ON | 0.483178 | 0.589273 |
| diffuse_BC_OFF | 0.423342 | 0.465785 |
| S_ON_BC | 0.488556 | 0.551065 |
| local_feedforward_AC | 0.605575 | 0.609191 |
| local_crossover_AC | 0.589732 | 0.470227 |
| widefield_AC | 0.606767 | 0.600876 |

RGC本次窗口结果如下，箭头为旧完整反馈背景→候选；事件率为总事件/(细胞数×0.2s)。hazard是各记录时点、实际history下的条件强度均值，不能等同于Bernoulli观测事件率。末列为原三bin Bernoulli sampler在最终固定V下8状态Markov方程的解析平稳期望，只是既有模型代数，不是本次实测平稳证明。

| RGC | 平均V mV | 记录hazard Hz | 单seed观测事件率 Hz/cell | 原模型平稳期望事件率 Hz/cell |
|---|---:|---:|---:|---:|
| midget ON | -54.747 → -52.534 | 306.46 → 345.79 | 260.42 → 288.22 | 258.21 → 285.53 |
| midget OFF | -56.230 → -55.337 | 285.78 → 301.83 | 233.90 → 247.97 | 239.92 → 251.37 |
| parasol ON | -54.919 → -51.748 | 299.88 → 359.74 | 266.67 → 300.00 | 256.08 → 295.24 |
| parasol OFF | -55.943 → -57.356 | 285.68 → 270.94 | 246.67 → 223.33 | 243.45 → 226.97 |
| S-ON | -47.447 → -44.898 | 433.88 → 483.71 | 361.00 → 391.00 | 348.20 → 379.06 |

这次参照纠正了旧工厂保持量比较的归因：midget OFF候选平均V比旧完整反馈背景高0.893mV，不能再沿用先前相对工厂保持参照的负差值。其他V差为mON +2.213、pON +3.171、pOFF -1.413、S-ON +2.549mV。这些是固定工程参数背景算例的确定性差异；不据此宣称实验稳态改进、真实放电率准确或单seed差异显著。初态修正消除了此前约7.6–20mV的未配平V0启动漂移，未降低也未掩盖高模型hazard；BC/AC有效参数和RGC强度工作点仍需同条件绝对V/电导/spike数据约束。未移动mu、压低gain或改S接线。实际冻结几何默认原已按参考电导构造RGC V0，本工程网结果不得外推为其启动漂移。

**验证与交付。** `tests/test_bc_ac_hypothesis.py`新增RGC闭式平衡/原状态不变/history-RNG参数保留回归，以及带真实反馈gap的旧采样固定点回归。连同candidate/contact-first既有回归共19项通过（3.20s）。核心env固定retipath_v1；仅只读绘图使用已安装Anaconda base Matplotlib，无环境更新。源文件inner/vh2005/rod_pathway/physical、network/hybrid、候选profile及生产配置与回滚逐字节一致；原文档前缀不变。

机器结果集中在 `reports/modeling_scope_20261019/background_references_v1/`：result.json、两条路径各自root/init/final状态、洗入/观测traces和事件、RGC_background_comparison.csv、layer_background_comparison.csv、interpretation.json、verification.json及background_comparison.png/svg。可重现入口 `tools/compare_bc_ac_background_references.py`；只读汇总/绘图 `tools/analyze_bc_ac_background_references.py`。本节取代前节“尚未实施”状态，不改写历史记录。剩余科学缺口为生理参数标定、同条件独立数据验证、正式适应/随机平稳证据及真实冻结几何候选运行；未为这些缺口伪造参数或切换默认。


### 本机真实观测读取、联合校准合同及冻结几何候选动力学（2026-10-20）

本轮依据用户“允许进入下一步、可以完成后继续”的授权推进；未要求每个未知系数都必须有直接macaque实测，允许显式借用/有效假设。未发现可用于当前BC/AC自由参数的匹配联合观测，故没有人为指定期望放电率或拿新旧工程输出作目标，也未伪称完成拟合。回滚保存为 `C:\Users\PC_User\AppData\Local\Temp\retina_joint_calibration_before_20261020_v1.zip`，含本轮前530个代码、配置、测试及文档文件和SHA。原来源、历史结果、候选参数及生产默认均保留。

**真实数据盘点与读取。** 检查项目data/source、source_data、literature、相关source_packages及Downloads中已知Dryad原始包；未宣称遍历整台电脑所有目录。`tools/audit_joint_calibration_data.py`通过既有冻结来源loader读取6条primate cone与6条primate rod已发表重复均值记录，共1,218,360对输入/观测样本，并与原MAT逐样本核对相等。官方MAT SHA仍为 `6724e2edd8e8708456f32900a14efae88192a5ba597365046d0031eb543d1331`。cone每条80,000点、0.1ms；rod每条123,060点、1ms；输入R*/s，观测为有符号电流除以来源估计暗电流，绝不是pA。记录为4–5次重复均值，原始trial、每条生物细胞ID、暗电流绝对分母和前史未提供。cone每条有1026个负的“总R*/s”值；rod第1条7635个，其他条各7155个，未裁剪、加偏置或改解释。高光值也未套新上限消去。

| 本机观测 | 可实际读取的内容 | 本轮联合校准边界 |
|---|---|---|
| Chen/Dryad官方MAT及README | 配对光刺激、归一化光感受器电流、README采样间隔 | 光感受器位于所选BC/AC tau_e参数上游；不识别这些系数。另有负总输入、前史、rod对齐及绝对尺度问题 |
| Dacey2000 Fig4B | 206列同一条FMB相对V曲线，phase cycles及数字化误差 | 无该波形唯一秒标定、绝对V及完整输入边界；206列不是独立观测 |
| Puthussery/McLaughlin既有来源摘录 | Rin、puff形态、glycine事件等条件性摘要 | 可约束特定assay；没有匹配当前模型端口的配对动态原始数组，不能直接替换全局gL/gmax |
| B–K 1997/1999来源转录表 | RGC功能模型参数的群体统计 | 不是配对单细胞波形；Mean列是统计均值，不是背景放电目标；无同条件BC/AC中间观测 |
| Godat/S-cone、Packer、Wassle、Schein、McGregor | S比例、密度、拓扑及几何资料 | 非神经时间序列，不用来拟合动态 |

另对本地Dacey原PDF作一次正文复核：Methods pp1802–1803中LED方案为1.22或2.44Hz，Fig4 p1807明确C/D/E为2.44Hz，但未唯一标定A/B波形的频率，不能把其他panel的秒尺度移到B。Fig4同页给出150–1200µm annulus、1000td、100% contrast；这些输入条件已入合同，仍保留13.6mm鼻侧记录及未知该细胞species的域边界。

**可执行读取/评分接口及数据分割。** 新 `retina_sim/joint_calibration.py` 提供整条source record分割及联合mean-level/centered-waveform损失。二者相加严格等于固定观测单位scale下的MSE，既不消去DC误差，也不引入可调观测gain/offset；scale不是估计出的实验噪声方差。分析窗口均值不自动叫静息/稳态基线。12条来源记录各组按ordinal 1–4训练、5–6留出；不按相关时间点/数字化像素随机划分。由于生物ID和独立来源未确认，称source-record holdout，不称独立细胞或外部验证。留出观测尚未做前向评分。

`joint_case_contract.json`固定当前候选profile SHA、当前允许的midget OFF/diffuse OFF tau_e两个自由度，项目搜索边界各1.25–5ms（来自既有0.5–2倍敏感性区间，不是生理测量范围）；保持C/gL/电流绝对尺度、有效u/K尺度、release映射和下游RGC全部gain/强度曲线固定。稳态e/V不依赖tau_e，因此这两个自由度本身不能降低高稳态率；只靠调tau把背景率做低会违反现有方程解释。未来若同条件绝对V证明工作点偏离，应依据该观测另版本化相应自由度，而非任意补偿。合同允许借用与显式假设，当前未拟合的原因是缺少与端口和自由度相匹配的观测，不是要求所有未知都有直接macaque实测。未来局部拟合仍须回装检查上游、改变层V/gate/release和所有保留RGC输出。

**真实冻结几何实际推进。** 从此前已构造的同一候选profile重建并核对冻结图：42组件、23 populations、20,422细胞记录、58,155状态量；50条PortLink中42张生物连接表、8条计算适配链接，另4张gap表。全部原生化学和电耦合反馈保留，没有裁图、合并细胞、冻结所需反馈或转换新机制。候选显式使用Python reference；没有把未lower的候选偷换到numba。本冻结图实际外部consumer仅native_light=100td与rod_light=5R*/s；LightContext内cone_background=1000R*/s字段在此图未被消费，不把它说成实际第三路输入。没有启动ISET/video长任务。

先在RGC瞬时配平状态上跑2ms背景，再从同一断点跑3ms恒背景和1ms +10%独立输入、2ms恢复分支。5ms时RGC配对差仅4.8e-12–2.2e-11V，低于1e-10V绝对容差，不能解读其符号；因此仅从两条原5ms断点继续20ms背景恢复，没有重放脉冲。继续后保持同一assembly/hash及原容差/步长。两分支到25ms的RNG抽样进度一致、所有记录输出和全部状态/导数摘要有限，事件计数与事件列表逐项一致。

构建11.18s、恢复初态0.11s、2ms背景1.62s、首两分支2.87/3.02s；续跑重建11.09s、两条20ms恢复17.13/18.13s。两次成功进程分别19.61/46.81s，峰值约344.2MiB；内存额度4GiB、可用内存至少1GiB、记录额度64MiB，推进每0.5ms保存有效断点并检查内存/时间界限。首轮统计读取字段名错误和记录预算不足均在推进前停止，后续日志相对路径错误保留了5.5ms有效断点；路径修复前后已完成首chunk所有数组完全相同。它们是验证脚本问题，没有改模型方程、数值系数或反馈；失败目录保留并注明原因。

下表是实际几何短窗工作点与25ms配对效应，均不是已适应背景率。事件计数窗口2–25ms，顺序恒背景/输入分支；小样本单seed不提供放电差异显著性。

| RGC | cells | 恒背景V均值，2→25ms（mV） | 25ms输入−背景V均值（µV） | 23ms事件数 |
|---|---:|---:|---:|---:|
| midget ON | 22 | -54.2805 → -53.9823 | +0.4333 | 137 / 137 |
| midget OFF | 22 | -54.2795 → -54.2084 | -0.9611 | 132 / 132 |
| parasol ON | 3 | -54.2668 → -53.9771 | +0.4087 | 21 / 21 |
| parasol OFF | 2 | -54.2789 → -54.2133 | -0.9604 | 11 / 11 |
| S-ON | 3 | -45.8481 → -45.5608 | -0.4408 | 27 / 27 |

背景分支本身仍有约0.067–0.311mV逐cell漂移，BC/AC和RGC慢瞬态未完成，不能拿25ms状态作正式适应cache。25ms配对RGC效应0.41–0.96µV已超过当前局部求解容差尺度，但不是步长收敛证明，更不是生理验证。响应仍在发展，未声称25ms恢复或测得完整时间传递函数。S-ON在此独立consumer协调输入下的净负差也不自动构成纯S色对抗结论。所有RGC V/gE/gI/hazard、事件及BC/AC工作点均保存；未为了改善其中一个输出调其他层或RGC参数。

**本轮交付与剩余范围。** 22项相关回归通过（2.92s），包括联合损失保留DC/动态误差、整record留出与原BC/AC背景/连接接口。核心env仍是retipath_v1，绘图仅使用已安装base Matplotlib，无环境更新。数据盘点、精确cone读取、版本化分割及合同位于 `reports/modeling_scope_20261020/joint_calibration_intake_v1/`；实际几何0–5ms位于 `frozen_candidate_dynamics_v3/`，5–25ms续跑和最终图、所有层均值时间序列、RGC工作点/事件表位于 `frozen_candidate_recovery_25ms_v2/`。入口分别是 `tools/audit_joint_calibration_data.py`、`tools/validate_frozen_candidate_dynamics.py`、`tools/extend_frozen_candidate_recovery.py`，只读分析为 `tools/analyze_frozen_candidate_dynamics.py`。

本轮没有可诚实启动的BC/AC联合生理拟合或留出预测验证；数据缺口已具体化为同条件有输入/秒时间轴的BC/AC观测及工作点/输出映射，而非一个预设期望Hz。实际大几何候选已从仅构造推进为25ms有完整反馈的有界动力学运行，正式适应与独立生理校准仍未完成。原生产默认、参数profile和来源字节不变，原文档前缀保留；无新机制、细胞合并或push。


### Latimer真实ON-parasol观测基线与被拒绝的条件性RGC拟合（2026-10-19 UTC）

本轮按用户授权优先解决下游观测缺口，沿用 `reports/modeling_scope_20261020/` 阶段目录；执行时钟为2026-10-19 UTC，不把目录标识当作观测日期。未再运行冻结几何、工程刺激或视频。回滚为 `C:\Users\PC_User\AppData\Local\Temp\retina_latimer_before_20261020_v1.zip`，含536个先前代码/配置/测试/文档文件与SHA。原生RGC方程、细胞类型、其他层和生产默认保持不变。

**公开数据与文件核查。** 本机原无Latimer真实数据。读取[Figshare官方条目9636854 v1](https://figshare.com/articles/dataset/ON-Parasol_RGCs_for_the_conductance-based_encoding_model/9636854)清单后，按体量在看响应前选最小 `Cell_6.mat`，下载17,007,338 bytes，未下载约145MB整包。官方MD5为 `83be38d612d7e31bdf38b731cd2c3ebf`，本地SHA256为 `59e1457bc27f7392a3c458902941a234c687b26a525b26d168bc8609aae40025`；两者均核对通过。来源原文件、API元数据、下载清单与论文开放XML保存在 `data/source_data/latimer2019_figshare_v1/`。eLife v1网页访问遇到浏览器挑战，Methods核查使用公开PMC6989090的出版论文XML镜像，不冒称已取得v1 XML。作者当前与论文归档代码均仅只读核查，未执行CBEM或安装其环境；README明说 `Data/example.mat` 是模拟示例，未拿它代替动物观测。

MAT源标签为 `060908Bc2`，ExcCurrents/ExcStm均30×71000，InhCurrents/InhStm均8×71000，SpikeStm为30×71000，SpikeTimes为30个独立数组。论文[Latimer et al. 2019](https://elifesciences.org/articles/47012v1) Methods说明：先做cell-attached spike记录，随后同细胞在阻断内源活性电导后，以不同保持电位分离E/I电压钳电流。不是同一模式或同一试次噪声；论文另有2例dynamic-clamp膜电位/spike实验，当前MAT没有膜电位字段，未拼接其反转电位或数据。

按71000个刺激值的完整SHA逐项配对，而非按试次序号配对。E与spike各有10个独特刺激，每个3次；I只含其中8个刺激，各1次。完整交集8组，每组3个E记录、1个I记录及3个spike记录；缺I的2组不进入钳制输入预测，排除原因与对应hash均记录。独特刺激SHA排序后前6组训练、后2组留出，分别18/6个spike试次；两个留出刺激的全部spikes未用于优化。所有数组前10000点及末1000点为恒定刺激，中间60000点为变化刺激；论文给出0.1ms采样，故源相对时间为1s前段、6s噪声、0.1s后段。前段取0.5–1s评分，动态取1–7s；未依预测效果挑窗口或调时间位移。

**尚未由来源证实的单位/同步边界。** MAT及Figshare说明未标电流单位、spike索引原点。spikes实际为1..71000范围的递增整数，符合采样索引而非秒；本轮“pA电流、MATLAB一基索引”是具名项目解释，不标为verified测量字段。原文Methods明确用约-70mV/+70mV驱动力将E/I电流换为导纳，本轮仅在该pA解释下按原文进行 `gE=mean(I_E)*1e-12/(-.070)`、`gI=mean(I_I)*1e-12/(+.070)`；未减offset、拟合输入gain、裁剪负导纳或把电压钳电流直接当电流钳输入。源E电流均负、I均正，转换后非负。该转换只构成条件性、跨记录模式的输入近似，不能声称已确认绝对导纳或同试次输入/输出。

源刺激本身是未给定物理光学标定的数值序列，含负值；没有LED命令至光谱/LMS capture、R*/s或td的充分映射，因此未接整网，更未将此数据用于BC/AC拟合。记录的0.1ms输入按具名ZOH解释保留全部样本；spike按本项目原有1ms bin聚合。30条记录11569个事件全部保留，1ms bin内无多事件，最小ISI为2.2ms；未为了适配Bernoulli模型丢弃多重spike。

**固定原生候选与推断。** `retina_sim/latimer2019.py`使用原 `HistoryRGC`、现有 `effective_parasol_RGC_ON` 参数。C=60pF、gL=2nS、EL=-65mV、EE=0、EI=-80mV、alpha=150Hz、mu=-70mV、beta=5mV、history=[-12,-8,-4]mV、spike interval=1ms。膜V仍为原不含spike reset的被动ODE，采用每100µs恒电导区间的解析解；单元回归对原native.derivative的数值积分核对通过。没有换成论文CBEM滤波器、Poisson采样或新电流。膜V是模型推导量，当前数据无实测V。

条件NLL使用真实strict-past spike history和原Bernoulli `p=1-exp(-lambda*dt)`；自由历史的预测均值则对原3-bin/8-state history过程精确边缘化，不混用hazard与事件率，也不靠单seed随机样本比较平均PSTH。前500ms不评分；此前未记录的spike前史不在评分窗口的3-bin条件history中。以下为两个留出刺激的实测/预测事件率，预测是自由历史期望，不是条件hazard；每刺激的PSTH基于3个真实cell-attached重复。

| 留出观测窗口 | 实测Hz | 固定原生候选Hz | 两参数条件拟合Hz |
|---|---:|---:|---:|
| 恒定刺激前段0.5–1s | 34.333 | 137.997 | 64.880 |
| 噪声刺激1–7s | 58.111 | 108.985 | 49.506 |
| 联合0.5–7s | 56.282 | 111.217 | 50.689 |

**一次受约束拟合及拒绝理由。** 在固定候选先得出真实数据对照后，仅对已有alpha、mu两个参数做L-BFGS-B训练NLL拟合。预先冻结C/gL、所有反转电位、beta、history核、bin及输入转换尺度；项目搜索边界alpha 15–600Hz、mu -90至-40mV，不作为动物置信范围。优化11步、15次评价，得到alpha=600Hz、mu=-54.9179mV；alpha触及上界，没有扩大边界或再换参数追逐留出结果。局部Fisher在log(alpha)、mu/beta坐标的特征值约20.235/11728.025，条件数579.6；固定锚点下局部满秩并不等于绝对物理尺度、真实膜电位偏置或全局参数均可识别。mu只能解释为给定尺度和反转电位后的有效offset。

留出联合NLL从8484.653降至7236.798（约0.820 bits/spike），但不能仅凭此采纳：恒定刺激背景仍预测64.880Hz而观测34.333Hz；20ms动态PSTH的去均值误差由2105.663增至2317.067 Hz²（约+10.04%），相关系数由0.7391至0.7355，未改善形状。总PSTH MSE由4693.861降至2391.113主要来自均值误差下降，不能称动态机制被验证。边界命中、背景偏差、形状未改善以及单位/索引来源缺口共同使 `calibration_decision.json` 判定 **REJECTED_NO_PRODUCTION_ADOPTION**。条件参数提案独立保存，不导入生产profile，不推广为ON-parasol群体参数，更不推广至其他RGC或BC/AC。

此模块提案未回装整网，现有所有RGC和中间层实际参数均未改变；原膜导数也不依赖alpha/mu或spike。若以后修改这两个输出参数，其直接作用是该RGC条件强度/history过程，不能据此声称BC/AC或其他类输出已验证。本轮已封存训练/留出划分、固定/拟合预测、分窗损失、PSTH与失败结果；不在已看过的留出集上继续自适应选择更多参数。下一步应先确认作者导出单位、索引原点及电压钳offset/驱动力来源；如继续自适应选择，须另留独立观测。

**真正配对的公开备用路径。** 已核读[CRCNS ret-1官方说明](https://crcns.org/data-sets/retina/ret-1/about-ret-1)，DOI 10.6080/K0RF5RZT：每个MAT会话文件含同会话的spike times与实际视觉刺激重建信息，61电极阵列、压缩合计约70MB，需CRCNS账户。它是Mus musculus，且该说明未给E/I钳制输入，故只作为真实光刺激/spike备用候选，不替代本轮macaque ON-parasol模块，也不能直接声称整网兼容；须先核细胞类型及光学输入映射。未创建账户或访问登录后数据。一次DANDI retina小型目录查询未确认另一套等价macaque E/I/spike记录，未下载其大包。来源/限制集中保存在alternative_candidates.json。

**验证与交付。** 本轮6项直接回归通过（0.57s）：原始配对/单位显式确认门、解析膜方程与原native导数一致、只改alpha/mu其余参数不变、原Bernoulli与strict-past history边缘化、联合评分与整record留出。主计算仍用retipath_v1；只读绘图用已有base Matplotlib，无环境升级。数据/接口入口 `tools/fetch_latimer_minimum.py`、`retina_sim/latimer2019.py`；固定/拟合入口 `tools/run_latimer_rgc_case.py`，只读汇总 `tools/analyze_latimer_rgc_case.py`。所有输出位于 `reports/modeling_scope_20261020/latimer_cell6_rgc_v1/`：contract.json、fixed_result.json、fit_result.json、原始刺激hash试次配对、精确预测NPZ、measured_vs_predicted.csv、拒绝决定及heldout_RGC_comparison.png/svg。原来源/默认配置/内核字节和旧文档前缀保留；本轮没有新机制、冻结几何重跑或push。真实配对观测基线已建立，已确认物理单位的RGC校准与BC/AC联合校准仍未完成。


### Latimer独立审阅合并与计算停止记录

已合并 `reports/independent_latimer_review_20261020_v1/review.md`。本次仅整理该审阅的实验条件、数据语义、准确缺口及停止理由，没有新增模型/优化/测试运行、网络检索或作者联系；没有发送邮件、更换物种、调整参数或生产默认。此前完整交付 `retina_latimer_delivery_20261020_v1.zip` 保留为合并前快照；原数据、预测、划分、评分和被拒绝的条件参数提案保持逐字节不变。本轮完成范围仍为真实观测基线、一次已授权条件拟合尝试及拒绝证据，不是完整生理校准。

**独立审阅结论。** 未发现本次范围内训练/留出泄漏、替换原生RGC方程、评分或已声明换算的实现错误。保存的ge/gi、spike counts及所有分窗评分已由审阅从源MAT/NPZ独立核对，评分最大差0；真实输入4ms逐100µs积分与解析推进最大误差1.0872e-13V。审阅复跑的是原有3项定向回归，与原本6项有重叠，不累计称9项独立测试。本合并没有重跑这些检查。保留“本次受约束提案未获采纳”，不能推导整个RGC机制被证伪。

**补充的实验条件不能直接成为模型参数。** 原采集论文例示E保持电位-70mV、I约+15mV；Methods说明全部报告电位校正-10mV液接电位，串联电阻8–12MΩ、补偿75%。这些是出版物的采集协议/例示，不是当前Cell_6逐试次头文件。保持电位、实际驱动力 `Vhold-Erev` 和模型反转电位是不同量；未用这些例示覆盖原EE=0/EI=-80mV，未对电流或电位补做校正。Latimer的约-70/+70mV驱动力换算仍只保留为已声明的条件解释。原采集论文包含多种灵长类，Cell_6的具体动物/种映射未闭合，不升级为具体猕猴种。官方eLife v1 API为POA，归档v1 XML无Methods正文；本轮Methods依据仍为出版论文开放全文，不假称逐段核对了v1 PDF。

**索引与留出边界。** 官方MAT导出单位/索引定义仍缺。独立审阅只把一基改成零基就改变8组中的1770个1ms计数格，固定候选留出NLL从8484.6528变为8473.8711；这是原点敏感性证据，不能按较低NLL选择索引、单位或时移。本分析原有假设与数组不变。6组训练/2组留出的动态6s刺激不同，但所有刺激共享同一1s恒定前段，不能称该前段是新背景条件。留出E/I是已知输入，检验的是条件性RGC模块，不是光刺激到spike的端到端留出。每组3E/1I输入代表波形去预测另外3个cell-attached重复，不提供同试次E/I-spike协方差；非线性系统中均值输入的预测不自动等于含真实输入噪声的平均响应。

20ms PSTH为非重叠箱平均，不是论文的2ms Gaussian平滑。审阅核对的两个留出观测20ms峰值均266.667Hz，固定预测216.442/233.661Hz，条件拟合154.417/178.332Hz；每刺激仅3个重复，不能把这些峰值当作高精度瞬时生理峰率。现有表格仍报告窗口平均事件率，hazard另列。alpha=600是softplus强度尺度的搜索上界，不是最大放电率。

**尚未闭合的准确字段与最小来源。** 完整机器记录为 `reports/modeling_scope_20261020/latimer_cell6_rgc_v1/review_closure.json`。

| 字段 | 当前已知/保留解释 | 仍需的最小来源 |
|---|---|---|
| Cell_6电流unit/scale/sign | 数值和极性已读；pA仍是项目解释 | 官方导出定义或电流通道校准头，给出物理单位、乘数及符号 |
| SpikeTimes单位、索引原点、零时刻 | 整数支持及一般0.1ms采样；一基未由导出代码确认 | SpikeTimes生成/导出表达式、原始/降采样网格及采集trigger定义 |
| 电流offset/背景处理 | 本分析未额外减offset或拟合gain；不知导出前是否已处理 | 每试次holding/leak/capacitive/background电流是否去除、数值和所用时间窗，明确零校正也可 |
| 每E/I试次holding | 出版物例示-70/约+15mV | Cell_6命令/实际/校正后保持电位、记录模式及电位参考的采集表 |
| 每E/I试次reversal与driving force | 原文近似±70mV换算；模型EE/EI保持项目值 | Cell_6测定或明确假设的Eexc/Einhib、离子条件及Vhold-Erev定义；或来源明确的校准gE/gI文件 |
| 液接/串联电阻校正映射 | 文献报告-10mV及Rs8–12MΩ/75% | 这些校正如何对应Cell_6采集头/导出电流、每试次Rs与补偿，避免重复校正 |
| 滤波/重采样/跨通道同步 | 完整刺激数组相同、长度相同 | 电流/刺激/spike的滤波、延迟、重采样和时钟对齐定义，不靠拟合选择时移 |
| Cell_6动物与实验条件映射 | 源标签060908Bc2及ON-parasol队列标签 | 细胞标签至动物/具体种、视网膜位置、温度和适应条件的对应表；不把未给条件补为测量 |
| 整网物理光输入 | 有实际full-field数值波形 | LED命令至光谱/辐射、均值/contrast定义及td或R*/s映射，才可评价整网兼容 |
| 逐试次输入噪声及后续独立验证 | 跨模式代表输入、已封存的本次留出 | 不虚构同步噪声；若根据该留出继续自适应选参数，应另取未参与选择的适当配对测试集 |

针对当前RGC模块，最小来源包是Cell_6导出README/脚本与采集/校正头：必须能绑定该文件的单位、索引/时零、offset和Vhold/Erev/校正定义；也可提供有等价单位、试次和时间来源的已校准导纳。动物/光学表只在提出更具体物种、位置或整网输入主张时进一步需要。不是要求每个模型未知量都有直接macaque实测，但不能以数值合理性替代这些数据语义来源。当前没有该最小来源包，不再盲调、换原点或更换数据物种。

独立审阅引用的Trong NCBI BioC全文、eLife v1 API响应及已有pytest XML已从指定Temp文件原样保存至 `review_sources/` 并核对审阅所列SHA；不依赖临时目录长期存在。原拒绝理由继续有效：alpha命中上界、背景仍偏高、去均值动态形状未改善，以及上述来源缺口。已保存可恢复合同、原始配对索引、固定/拟合预测及提案，并停止计算；没有新增后续任务或定时计算。

### 分阶段调参方案：先固定语义和工作点，再开放可辨识参数（2026-10-20 UTC）

本节是后续实施方案。本轮只读代码和既有证据、更新本统一文档，没有启动模拟、拟合或视频，没有修改参数和实现。允许采用来源明确的借用值、有效参数和先验，不要求103个条目都有直接猕猴测量；每个结果仍须保留其物种、细胞、输入和观测适用域。midget/diffuse、ON/OFF、S-ON及各AC家族保持独立，不因数值暂时相同就共享拟合结果。

**先处理两个实际接口问题。** 当前 `calibration_modules.merge_parameter_package` 只允许 `unfitted_snapshot`，要求系数与原装配逐项哈希相同；成功状态明确为 `unfitted_snapshot_identity_merge`、`system_acceptance=False`。它证明身份往返，尚不能回装真正拟合过的系数。另一个问题在 `bc_ac_hypothesis.resolve_background_bindings`：它从背景原始输入重算K、e0、V0，并移动release_half/activation_half以保持旧输出锚点。这样调gE时half也会变；以后开放K/half时还可能被派生绑定覆盖。这些必须在拟合前解决，不能靠优化器补偿。

后续应把K和输出half作为**模型系数**：在一个声明的参考背景下确定并冻结，或在指定实验中显式拟合；跨刺激/背景不能偷偷重新居中。V0、e0、release初态和spike history则是**初始化条件**：使用实际解析出的系数及该条件的输入重新求全反馈工作点；拟合K后e0也必须使用新K。当前候选只接受100td、cone1000R*/s、rod5R*/s参考声明，多背景实验需要新适用域和绑定合同，不能绕过检查。当前实际几何路径未消费的cone1000R*/s也不能作为已校准的输入量。

**RGC参考优先于上游压低增益。** 现有方程是 `C*dV/dt=gL*(EL-V)+gE*(EE-V)+gI*(EI-V)`，强度为 `alpha*softplus((V+h·history-mu)/beta)`；mu不是零放电阈值。当前alpha150Hz、mu−70mV、beta5mV在EL=−65mV且history为零时已给出约197Hz强度，因此先检查输出工作点，不能据高背景直接推断BC/AC驱动过强。

[Latimer论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6989090/)及本地冻结的作者代码提供EL−60mV、EE0、EI−80mV、alpha90Hz、mu−53mV、beta1.67mV等参考；非线性来自两只ON-parasol的dynamic-clamp结果，未与当前Cell_6形成同一组实测。论文省略C的归一化方程中gl=200不能直接写成200S；映射到物理形式需统一gL/C及输入g/C的时间尺度。逐细胞h等是拟合量；当前没有核实到Cell_6完整已发表h向量。作者history基底约190ms，当前只有3个1ms tap；论文使用条件Poisson，当前是每1ms最多一次事件的Bernoulli规则。故优先建立独立的、具来源版本的完整参考包，覆盖膜尺度、反转电位、非线性、history、采样规则和初态。完整条件未闭合前，只比较共同输入下的V与给定history的lambda，声明为部分方程参考；不能只替换mu/三元组就宣称重现原模型，也不能把模拟输出称为动物观测。参考包不直接覆盖生产默认值或其他RGC家族。

**COREM的用途限制在可对应的方程。** 官方[单室膜模块](https://github.com/pablomc88/COREM/wiki/Single-compartment-model)可作为RC响应、导纳符号、单位、反转电位及稳态平衡的数值/功能参考。其被动模式示例Rm100MΩ、tau10ms，与导纳模式示例Cm0.25nF、Rm100MΩ（泄漏tau25ms）是不同配置，不能混配。当前BC/AC候选C12.5pF、gL0.5nS相当于后一组C/gL同时缩放0.05，是项目借用选择；固定电流下并不保留相同电压响应。教程不能提供猕猴受体K/Hill/tau_e、ON/OFF机制、释放half/slope/tau_release、Ca释放或AC亚型动力学。即使COREM数值一致，也只通过方程参考门，不通过这些生理参数的验证门。

#### 开放次序与固定项

所有阶段先冻结ROI/halo、cell ID、原始W及行和、连线/缺额、gap、输入单位与增益、符号、时间基准和采样规则；不通过换密度、邻居、下游增益或反转电位去补当前阶段误差。BC/AC实际输入是 `q=W*Hill(u;K,hill)`，ON再取 `row_sum(W)-q`，随后tau_e滤波；未归一W时最大输入导纳是 `gE*row_sum(W)`。模块夹具须区分原始u和接触后q；现有校准模块的drive边界是后者，不能再施加一次Hill/W。

| 阶段与实验 | 本阶段具体开放字段 | 固定项、进入条件和产物 |
|---|---|---|
| 0. 语义与参考 | 不拟合 | 闭合电流/导纳单位、holding与reversal、offset、时零/索引、记录模式；建立上述RGC/COREM独立参考包，先验证同方程输入输出。Cell_6缺失导出说明时只保留已封存的条件分析。 |
| 1. RGC被动膜 | 有校准注入电流及V时，最多C、gL两个；缺绝对尺度时只估tau_m=C/gL及有证据支持的导纳比 | EL/EE/EI、输入尺度和输出非线性固定。只有spikes、没有V的Cell_6不能识别物理C/gL；使用固定来源先验时，结果标为条件有效模型。 |
| 2. RGC输出 | 首先mu一个；覆盖非线性弯曲区且有可靠V约束时才考虑mu、beta两个，alpha固定；之后单独开放现有history_kernel的一个幅度因子 | 膜/输入尺度已固定，history形状、dt及范围固定。不得一起放开alpha/mu/beta，也不逐tap盲拟合。history用条件计数似然及ISI/条件统计验收。与参考采样规则不一致的部分单列。 |
| 3. 单一BC/AC家族膜及输入动力学 | 若DC已通过，先tau_e一个；有绝对输入/V时可gE、tau_e两个 | 固定C/gL锚点、EL/EE/EI、K/hill及输出曲线；检验稳态V、上升/峰值/恢复。tau_e不会改变稳态，不能用它修背景DC。当前配置仅开放midget_BC_OFF.tau_e与diffuse_BC_OFF.tau_e，最多两个；它们独立。其他开放组需要先修订实验profile/policy。 |
| 4. 输入与输出非线性 | 多剂量、输入尺度已知时单独考虑K、hill；BC有V及释放/绝对突触输出时考虑release_half、tau_release；AC有V及输出曲线时考虑activation_half、activation_slope | 每次仅一个有观测支持的组，最多两个自由度。BC的release_slope先固定，宽V范围数据充分后再另立实验。AC当前输出是瞬时sigmoid，没有tau_release字段。接收端增益固定；只有RGC spikes时至多识别复合链的有效增益，不能拆称受体/释放实测。 |
| 5. 回装与全网 | 不继续调其他层补偿 | 新profile回装后重新求全反馈背景，验所有保留RGC及受影响BC/AC的V、gE/gI、输出、DC与动态。HC、rod/RBC/AII、glycine保留原参数，需改变时独立立项。只有通过单元、系统及真实观测门的提案才可另行采用。 |

上述是实验分组，不是自动逐组执行的优化流水线。现有Latimer CLI仅拟合alpha/mu；本节mu/beta及history幅度等新分组还需要受限适配器，当前没有实现或启动。无对应观测的组继续使用明确先验/近似，不要求等到103项全部识别才运行工程模式。电压方程对C与所有导纳共同缩放有尺度混淆；已知绝对注入电流或gap导纳可打破该对称，不能忽略它们。u与K共同缩放、释放曲线与接收增益、低激活区alpha与mu也有混淆。每组先看加权敏感度秩、参数相关和观测范围；秩亏即固定其中一项，不靠增加自由度宣布生理识别。局部满秩只说明该条件下局部可辨识。

#### 真实观测、数据隔离和验收

- 参考模拟器结果用于方程/实现一致性；真实数据用于生物适用性。Chen重复均值已参与源模型参数化，不能充作完全独立外测；FMB Fig4B尚缺AB绝对秒轴/电压且位置与中央ROI不同，不能直接拟合时间常数或绝对V；B–K边际统计不能拼成一条联合试次，Mean也不是自发放电率。
- Cell_6保持条件RGC实验：E/I来自不同记录模式和试次，代表性输入不是同步同试次输入；留出E/I作为给定条件可以，但不证明光刺激至spikes的整网预测。同一刺激的重复及时间片必须整组划分；共用的1s前缀不是新的背景条件。禁止用留出响应归一化、估offset、选索引或选延迟。
- 预先锁定训练/验证/最终测试及数据hash、预处理、自由字段/范围和评分。训练只估系数，验证选至多3个事先列明的分组/模型方案，最终测试一次；这是控制计算与自适应选择的工程预算，不是生理阈值。若根据最终测试再选范围、beta、history或模型，须新增独立测试或预先设计嵌套验证。现有Latimer末两组已影响后续判断，不能重新命名为新的独立测试。
- 同时评价绝对背景/均值、去均值动态形状、延迟/峰值/恢复；RGC再评计数似然和PSTH/历史统计。似然可使用观测的严格过去history，预测PSTH/背景统计则使用自由生成的history，不能让真实spikes替预测保持低放电。权重与容差根据测量单位、噪声/重复变异预先确定，不能去均值后丢掉背景，也不能只压放电率。无误差资料时只报告数值差异和工程容差，不发放生理通过结论。
- 保留当前拒绝：alpha600触界，背景预测64.88Hz对观测34.33Hz，留出动态去均值MSE恶化10.04%；NLL改善不覆盖这些失败。下一次不能据此扩大alpha边界或在同一留出上轮试beta。独立审阅未发现当前方程/评分实现错误，也不等于已否定整个RGC模型。

#### 真正拟合参数包回装需要补的接口（仅计划）

1. 保留原身份往返API；另设版本化fitted-proposal合同，列出允许变更的字段、单位/形状/语义、来源与适用域、基础profile/实现hash、拟合自由度/范围、数据划分hash、预处理、训练/验证分数及拒绝状态。拟合值是推断结果，不改写成直接measurement。
2. 生成新registry/profile并更新相应population profile与组件绑定；原W、IDs、gap和未授权参数保持一致。兼容检查比较明确允许的参数差异，不能要求新旧全部参数hash相同，也不能关闭整套检查。候选所需计算性occupancy连接/绑定需与新系数保持一致。
3. 分离背景派生状态与K/half系数，使用新参数重新生成初态、求反馈工作点；新参数hash使旧适应缓存/checkpoint失效。旧状态只有经过数学上明确的转换才可迁移。RGC仅改输出曲线时，若膜/输入/图不变且证明V不受spikes影响，可复用确定性V参考；仍须重建history并做统计验收。
4. 先输出可审阅参数差异，再用真实工厂重建、做直接相关单位/边界回归、全反馈背景和预先锁定观测检查。提案保持独立可回滚，失败不绑定GUI默认；“回装成功”不自动等于系统或生理通过。

#### 计算预算与停止条件

先做低维模块实验，每组最多2个自由参数、50次目标评价、5分钟墙钟，先到即停；参考比对不启动优化。已有Cell_6低维计算很便宜，但源语义未闭合不能因此多试。小工程网约1427个反馈未知量的背景根求解已耗46秒，约300ms比较各需99–116秒，后续同规模根求解可预设120秒/最多50次迭代边界。真实0.35mm候选约58,155状态：已有2ms背景耗1.62秒，20ms延续各约17–18秒；25ms仍有0.067–0.311mV背景漂移，不能当作充分适应。

整网仅在模块门通过后选一个声明背景和一个锁定刺激做首轮基线/候选比较；保持全部反馈。已有1.25s背景时间只是工程下限，按现有短段吞吐量可能需要约十余分钟，实际以收敛与实测成本为准。建议每个条件30分钟墙钟上限、RSS4GiB上限并保留至少1GiB可用内存，采用可恢复的短块计算；不得把25ms吞吐预算冒充1.25s已完成。大网超预算只保留checkpoint/失败状态，不静默降密度、删反馈或宣布稳态。适应停止同时检查状态导数/连续窗口漂移及RGC历史统计，容差须在该轮前锁定。

单位/索引/时零未闭合、敏感度秩亏、参数持续触界、DC或动态留出恶化、非有限值/非法导纳、背景未收敛、内存/墙钟超限均停止当前组；记录失败并保留最后有效结果，不自动扩边界、换层补偿或无限追加实验。先完成参考合同、绑定/回装接口和可观测的一组，再评估下一组，不启动全网103维联合搜索。

### 调参方案首阶段执行交付：工程合同、零拟合审计与视频恢复边界（2026-10-20 UTC）

本轮依据“可以，进行”的授权完成首阶段实现与运行检查；没有启动动物参数拟合或视频。改动前将567个源代码、配置、测试和文档文件保存并逐项校验于 `C:\Users\PC_User\AppData\Local\Temp\retina_calibration_contract_before_20261020T003220Z.zip`。用户原有并行改动未回退；未提交或push，未升级环境。

**两个工程合同已落实。**

- `calibration_proposals.py` 新增独立的 `module-fitted-proposal-1.0` 合同。提案和消费端均要求明确的字段白名单，检查单位/语义/形状、有限值、参数组、范围、注册来源及其冻结身份、适用域、训练/验证/测试组hash和预处理/分数/不确定性。history只允许一个正幅度因子，不开放逐tap。BC/AC不允许同时修改释放曲线和接收端gain。真实观测与工程参考明确区分，工程扰动声明 `fit_performed=False`；声明的数据划分并不自动证明实验独立性。
- 拟合提案须先保存成文件，回装时核对内容，生成新的profile与派生建模范围身份，同步群体及适配器绑定，保留原W、细胞ID和gap。新增来源会改变原生ParameterSet的全注册表来源hash，但其系数和原profile逐项保持相同。旧快照仍走严格身份往返；任何提案回装结果保持 `system_acceptance=False`、生产默认关闭，背景与checkpoint失效。新参数用冷初态重新准备全反馈工作点，不能转移旧适应状态。提案创建日期固定在包内，避免同包跨日回装改变身份；首版工程包使用其已知创建日期兼容。
- `bc_ac_hypothesis.py` 将历史配方的K和release_half/activation_half解析为带内容hash的固定参考。只允许冻结版本配方自动产生该参考；改过gE等系数的配方必须显式提供原参考，不能再次以旧输出锚点移动half。新的背景状态使用实际解析出的K/gE等系数；K变更的回归确认e0随新K重算。参考来自项目工作点约定，不是测量。当前仍保留原100td/1000coneR*/s/5rodR*/s适用域限制。

**前后运行证据。** `tools/audit_calibration_contracts.py` 对保留全部连接和反馈的1070细胞工程网做一项预先声明的QA扰动：只将midget OFF BC每加权门控的gE系数从0.272727nS增至0.327273nS（+20%，不是拟合结果）。K保持0.240810643，release_half保持−56.783099mV。回归同时确认旧算法会随gE移动half，而新路径不会。两次根求解均为1427个未知量，分别13/14次SciPy外层目标评价、45.34/46.83秒；计入数值Jacobian后实际RHS调用为16,343/17,600次，不能将外层计数当作总计算量。最终最大缩放残差为8.74e−10/1.90e−9每秒，均低于原1e−5验收值。原W、所有gap及细胞ID保持一致；新装配拒绝旧背景checkpoint。

| 全部保留RGC | 基线V / QA后V（mV） | 基线零history强度（Hz） | 基线 / QA后理论稳态计数率（Hz） |
|---|---:|---:|---:|
| midget ON | −52.534 / −52.534 | 528.774 | 285.526 / 285.526 |
| midget OFF | −55.337 / −53.431 | 450.691 | 251.366 / 274.923 |
| parasol ON | −51.748 / −51.748 | 551.827 | 295.235 / 295.235 |
| parasol OFF | −57.356 / −57.356 | 396.642 | 226.973 / 226.973 |
| S-ON | −44.898 / −44.898 | 754.040 | 379.060 / 379.060 |

这些计数率是恒定V下原3-bin Bernoulli模型的精确Markov期望，不是新动物观测，也不是一段已证明统计适应的模拟spikes。midget OFF BC平均电位上移1.923125mV，activation由0.463748增至0.564503，对应midget OFF RGC上移1.905768mV；其余BC/AC、AII及4类RGC的电位变化仅为数值误差（中间层最大绝对误差小于4e−11mV）。没有通过全网重调增益去消除此响应。

**零拟合hazard与兼容膜参考。** 全部5类RGC各以16,384个工程副本调用原生事件生成器，逐个对照同一RNG均匀数、严格过去history移位、`1-exp(-lambda*dt)`计数和首事件时间：全部一致，事件均落在正确bin内，每细胞每bin最多一次。高背景率确实来自当前工作点/非线性及短history的模型选择，未发现采样器实现把强度错当计数或重复生成事件；这不证明这些选择符合动物数据。

5类RGC与8个BC/AC真实工厂在固定接触后门控/导纳夹持下，对照独立的RC闭式轨迹，最大V误差6.39e−12V。另按COREM官方wiki固定版本 `3040efb07441c4f508612b78340cff8390969f77` 的共同导纳膜方程作SI映射对照，导数误差为0；导纳模式25ms泄漏时间常数与被动示例10ms分开处理。这是[官方膜方程](https://github.com/pablomc88/COREM/wiki/Single-compartment-model/3040efb07441c4f508612b78340cff8390969f77)的工程比较，没有运行COREM程序，没有验证BC/AC受体或释放生理。Latimer已发表alpha90Hz、mu−53mV、beta1.67mV的零history非线性也作了共同公式核对；完整逐细胞history和Poisson/Bernoulli差别仍未闭合，不宣称完整参考模型复现。

**实际几何与成本。** 来源约束的0.35mm候选图重建耗11.77秒，仍为42组件、23群体、20,422细胞、58,155状态、50条PortLink、4组gap；生物连接表和gap hash与此前冻结图完全相同，全部5类RGC保留。新增固定参考使装配/model身份更新，不能误用旧checkpoint。本轮这张实际图只重建并核对身份，没有推进或宣称适应；上面的两次根求解来自较小工程网，不能将其45秒成本套到实际58,155状态整网。工程审计总墙钟108.24秒；本轮未记录其进程峰值RSS，不将其列为已验证的内存峰值。

相关回归65项通过（46.61秒）；最后对日期身份稳定和旧half行为增加断言后，重跑其中5项通过（18.14秒），不是70项独立测试。pytest默认Temp不可写的首轮已改用工作区独立临时目录，未提升权限。模型计算始终使用retipath_v1 Python；绘图只调用既有base Matplotlib，未安装或升级包。11个受保护配置/核心方程文件的前后SHA一致，包括生产默认、候选历史配方、inner/rod/vH/physical以及network/hybrid和接触后机制实现。

**后续拟合与视频结论。** RGC按“零拟合审计后先mu、再history一个幅度因子”执行，beta留给有可靠V覆盖的独立实验；现在没有满足输入/膜约束与独立观测条件的新目标，故没有开放优化。Cell_6单位/索引及记录校正缺口、缺匹配V、旧留出已被看过，限制其绝对生理采用；FMB时间/电压语义及BC/AC独立输出观测仍不足。这些不足不要求103项全部补齐，也不阻止明确标注近似与瞬态的工程预览。

本轮没有发现需要等待全部细胞校准才能恢复工程短视频的理由，当前视频继续暂停是用户尚未恢复的范围约束。若恢复原生产默认工程预览，可保留其既有模式与来源声明；若选择本轮候选/新参数身份并要求适应后视频，真正前置项是：同图同输入的新背景准备、适应/瞬态状态明确、以及按实际Python候选吞吐量安排可恢复运行预算。此前20ms延续耗17–18秒，25ms仍有背景漂移；不能复用为已适应背景，也不能将完整背景费用省略。无需为恢复工程预览继续盲调mu、history或其他层，不自动切换候选为GUI默认，不自行启动0.5s视频。

统一证据保存在 `reports/modeling_scope_20261020/calibration_contract_execution_v1/`：参数提案与固定参考、前后完整层输出和root checkpoint、五RGC/中间层CSV、一次对比图、采样器与膜参考审计、实际几何重建记录、JUnit XML及受保护文件SHA。图为 `contract_audit_comparison.png`；可用 `tools/audit_calibration_contracts.py --output <新的工作区目录>` 重做有界工程审计，再用既有base Python执行 `tools/analyze_calibration_contract_audit.py --output <该目录>` 绘图。没有新研究对照、训练、下载或后续定时任务。

### 一次0.5秒真实视频恢复：许可服务阻塞，神经推进0秒（2026-10-20 UTC）

用户随后明确批准恢复一次0.5秒真实视频工程测试，前段的“视频暂停”范围约束已被本次授权更新。本次实际启动了原视频核查、最新候选装配及ISET导出准备，没有用gE+20%工程扰动或被拒Latimer提案，没有切生产默认。新的回滚为 `C:\Users\PC_User\AppData\Local\Temp\retina_video_halfsecond_before_20261020T005423Z.zip`，567个文件逐项SHA校验通过。

原视频是此前选用的 `C:\Users\PC_User\Videos\原视频\年番素材\已嵌完\yumepasi解散内嵌.mp4`，实际存在，SHA256仍为 `ef30b3353945b46c6e8d2310d1437526efee89973d6609ba66c561df2c25b793`；1920×1080、30fps、1350帧、45秒。本次目标仍为0–0.5秒，15个输入视频帧及终点采样，不自动延长。旧 `runtime_segments_20261019/video_input_v2` 缓存manifest的SHA核对通过，但其3mm几何、全局光学支持和当前0.35mm图不同，明确拒绝复用，也未加载任何旧神经checkpoint。

当前未扰动候选装配耗11.103秒，仍为42组件、23群体、20,422细胞、58,155状态、50条PortLink、4组gap，其中实际输入1590个cone、4586个rod，全部5类RGC保留。候选配方仍为原 `c07671...` 冻结版本。11个参数/default/核心方程保护文件的前后SHA完全一致。

**输入与执行实现。** 新增 `retina_platform/frame_indexed_frontend.py`：沿用原MATLAB光学导出器、完整光学支持、固定参考桥、严格首窗口输入读入、原独立RGB rod适配器及已有 `RowStore/held_channel`。实际1590-cone捕获本身仍按原5000个100µs窗口导出；成功后逐行确认同帧捕获位相同，才压缩成帧字典。rod保留原全局100µs时刻的取样规则，以帧索引保存值；恢复时仍先除实际窗口宽度、再除固定LMS桥系数，不用名义dt预换算。当前完整展开的输入数字数组约235.64MiB，已超旧128MiB单包限制；帧字典核心数组预计约0.742MiB，保留相同输入，未少算细胞或改变光学。

新增 `tools/run_candidate_video_halfsecond.py`，设计为新身份背景准备、原RHS/反馈/时钟/RNG、5ms推进块、1ms全注册状态及关键输出记录、逐块checkpoint和有限性/连续性检查；背景使用原收敛策略与工程时间下限，不复用旧适应身份。运行内存保护为8GiB Python峰RSS、至少2GiB可用系统内存，未修改生产配置。启动恢复入口 `--resume-startup` 只允许同一次试验在任何神经/背景推进前的恢复，不能开启第二段视频或覆盖已有有效输入。背景和视频分支因下述许可阻塞尚未真实运行验证，不把代码已具备误称为0.5秒已完成。

**确切停止点。** MATLAB在运行ISET代码前退出，stderr为“无法与所需的MathWorks服务通信（错误5201）”，退出码1，stdout为空。已生成的 `input.mat` 包含真实RGB patch，形状241×255×3×15、原5001个窗口边界和5000个帧索引；没有 `capture.mat`，没有背景或神经响应。启动到失败总墙钟125.685秒；Python进程峰RSS为668,213,248 bytes（约637.3MiB），当时可用系统内存约19GiB，未记录MATLAB子进程峰值。没有神经推进速率可供推算，未承诺10分钟，也未因推测成本终止。

[MathWorks官方说明](https://www.mathworks.com/matlabcentral/answers/1815395-why-do-i-receive-error-5201-unable-to-access-services-required-to-run-matlab)将5201归为许可问题；当前不能仅凭该错误确定是网络还是Service Host/其他环境因素。尝试在受限环境外启动同一捕获以访问许可服务，自动审批两次拒绝：第一次仍引用旧的只读/暂停范围，第二次明确表示跨任务转述的批准不算本执行任务的直接授权。受限外MATLAB实际未执行，没有换启动路径绕过拒绝、修改许可/安全设置或降级cone输入。已向用户提出仅限本次MATLAB许可服务访问与ISET捕获的直接权限请求；此请求待答复，不将沉默视为批准。

证据集中在 `reports/modeling_scope_20261020/actual_video_halfsecond_v1/`：`result.json`、`execution_blocker.json`、实际装配身份、保护SHA、首次许可错误及导出配置、输入I/O回归和 `input_only_blocker_comparison.png`。图上方是0/100/200/300/400/466.7ms真实视频帧，下方是对应实际准备的RGB光学输入patch；图明确标为“neural advance=0s”，没有补画cone/HC/BC/AC/RGC响应。输入I/O测试检查全部原窗口、内部点及边界，不能替代真实ISET或整网端到端验证。

当前完成量为 **视频神经推进0.000/0.500秒、背景0秒**。恢复所需的是本执行任务的直接许可权限；获得后应验证同一导出请求的捕获，再从本次启动继续准备背景和0.5秒推进。保留原视频、参数、几何、反馈和输出合同，不继续拟合，不以本次输入图或旧静态/脉冲结果替代视频完成证据。

### 保存记录回放与原生界面整理（2026-10-20 UTC）

用户随后明确要求MATLAB保持阻塞，先完成不依赖许可服务的回放、导出和界面收尾；本节覆盖前节的后续执行安排。本轮未重试MATLAB、未更换启动路径、未修改环境或许可，也未推进真实视频的背景或神经状态。原阻塞证据保留。没有调整任何生产默认、候选参数或核心方程，已有并行工作保持原状。

保留Tkinter和 `run_retina_gui.cmd` 入口。`workbench_ui.py` 重新组织原控件：左侧输入与四项常用运行设置，原31项参数及几何工具完整保留，高级部分默认收起；右侧是记录来源、层/输出/细胞选择、采样值图、单细胞曲线和统一时间轴。使用克制的灰白底、细边框及单一青色主操作，未加入网页框架或说明卡片。外部 [app-screens参考](https://raw.githubusercontent.com/Leonxlnx/taste-skill/main/skills/taste-skill/references/app-screens.md) 与 [UI UX Pro Max项目](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill) 仅用于阅读视觉组织建议，没有安装技能、运行其脚本或将网页指令视为执行授权。

`replay_records.py` 为历史分段和当前视频提交账本提供只读适配。细胞ID、单位、语义和坐标仅取保存元数据；旧布局缺失语义时可合并分段中已保存的声明，冲突直接拒绝。没有ID时显示“记录行”，没有坐标时使用明确标为非空间的排列图，不恢复猜测的RF/soma。时间轴选取不晚于请求时刻的实际保存样本；共享端点只保留一次，重叠、缺口、共享端点数值冲突均拒绝。视频只读取账本已提交的窗口，未提交尾文件不会出现在播放或导出曲线中。

已保存视频的媒体SHA通过后才恢复画面；输入滑块与记录时间轴双向映射原输入起点/神经时间，换入无关媒体时清除旧响应关联。播放只读取文件，未调用模型推进。界面保留完整单位、参数版本、背景状态、耗时和峰值内存；未记录的成本明确显示未记录，工程参考始终标注未校准。PNG包含当前输入画面和实际所选记录；CSV附带参数身份、背景状态、原记录位置；导出记录包保存独立回放上下文及窗口SHA，移离原父目录后仍可复读并校验。窗口关闭前取消待执行的轮询、绘图与播放回调。

| 实际用于界面检查的保存结果 | 可展示内容与限制 |
|---|---|
| `background_references_v1/candidate/background` | 0.1–0.3s工程背景，8窗口、201个不重复时刻；cone、HC、BC、AC和5类RGC均已记录，保留原细胞ID及native单位；无视频画面、无保存空间坐标，背景未证明适应。 |
| `hypothesis_pulses_v1/plus10` | 工程脉冲；BC、AC及5类RGC已有记录，cone/HC明确未记录，不关联真实视频。 |
| `frozen_candidate_recovery_25ms_v2/input_step_recovery` | 来源几何候选的5–25ms保存段，40窗口、201个时刻；参数版本 `c07671ccc8…`，耗时18.13s、原运行峰值344.2MiB。cone/HC和细胞ID/坐标未保存；未证明适应、未校准。12.35ms请求准确定位至已有12.3ms样本。 |
| `actual_video_halfsecond_v1` | 匹配原媒体SHA的真实画面，0–0.5s输入时间轴；已生成神经窗口为0，cone/HC/BC/AC/5类RGC均显示响应未生成，不借用任何脉冲或背景曲线。 |

在原retipath_v1 Python环境完成28项回归（15.73s）：包含上述适配、单位冲突、导出包独立复读与篡改拒绝、当前候选的小型checkpoint连续性、5类RGC的RNG/history/delay/tick连续性及既有checkpoint身份拒绝测试。测试仅使用毫秒级小fixture，不能视为整网生理验证。真实Tk窗口另检查1440×900、1200×780及125%字体缩放，常用操作可见，高级参数可展开/收起，单细胞选择与时间轴联动；视频滑块0.2s对应帧索引6（从0计），输入侧0.3s能回写统一时间轴。89个输入记录/元数据文件前后SHA一致，GUI回放期间未创建模拟会话。启动入口的 `--help` 与 `--results` 参数透传已检查。

统一证据位于 `reports/modeling_scope_20261020/playback_ui_v1/`：`final_regression_tests.xml`、`final_gui_checks.json`、`scope_integrity.json`、实际窗口截图和PNG/CSV导出。`before_native.png` 才是旧界面截图；早期 `before.png` 是无效的桌面矩形捕获，不用作界面证据。当前截图使用原生窗口句柄捕获并已逐张查看。3张交付截图已上传到个人Library并获得成功回执（`library_screenshot_receipts.json`）：[全层工程背景界面](https://chatgpt.com/api/library/files/libfile_92ba7ca2244c819185e3cc0547062bc8/download)、[1200×780来源几何记录](https://chatgpt.com/api/library/files/libfile_5b7002902238819182b6bbb1133ed2e0/download)、[真实视频未生成响应界面](https://chatgpt.com/api/library/files/libfile_3d8d2fb180a88191a1bfcf3b65d67a92/download)。

本轮开始前的回退包为 `C:\Users\PC_User\AppData\Local\Temp\retina_playback_before_20261020T020212Z.zip`，包含556文件及SHA清单；其中199个core/configuration/example/data文件本轮未变。交付包为同一Temp目录下的 `retina_playback_ui_delivery_20261020_v1.zip`。启动仍用 `run_retina_gui.cmd`；只打开保存结果可用 `run_retina_gui.cmd --results "reports/modeling_scope_20261020/frozen_candidate_recovery_25ms_v2/input_step_recovery"`，不会自动运行模拟。真实视频响应仍是待许可服务问题解决且另有执行授权后的未完成项。

### 用户直接授权后完成0.5秒真实视频（2026-10-20）

用户在本执行任务直接授权一次受限环境外的原MATLAB启动，用于MathWorks许可服务访问和既定ISET捕获。本次权限审查通过，实际只启动一次原R2026b程序；原 `export.json`、`input.mat`、固定ISET版本和请求 `eeb83eae…` 保持一致。未修改许可、账户、系统设置或生产配置。MATLAB用时21.754s，退出成功，日志为 `MIXED_CAPTURE_PASS 1590 cones, 5000 windows, 15 actual OI frames`；末尾的 `hiddenHandle` 清理警告原样保留。捕获SHA为 `21fc74b732c76fd8f670567a9684227c4935e3e8c717d6763a19fb41322a6a44`，5000×1590个期望R*/window、时间边界及15帧身份校验通过，同帧窗口逐位一致。仍为LCD-Apple/human-MW及原独立RGB rod近似的混合参考输入。

在同一试验目录继续原候选 `c07671ccc885…`：42组件、23群体、20,422细胞、58,155状态、50条PortLink及4组gap保持原装配。一次启动在严格首窗口绑定后因实验帧适配器漏设 `cache_reused` 元数据标志失败，耗17.202s，尚无神经推进；部分输入保存在 `startup_partial_input_02/`。随后在本次已验证缓存专用恢复进程中补齐该标志，并禁止第二次光学导出，冻结光学源码及所有数值方程保持不变。说明见 `directly_authorized_capture_v1/runtime_binding_correction.json`；两个启动失败记录均保留。

新背景推进至1.3s，通过原连续窗口及1.25s工程下限，状态为 `CONVERGED`。该状态表示确定性注册状态的数值判据通过；spike平稳性和全生理校准尚未建立。之后实际完成0.5s视频推进，绝对神经时间为1.3–1.8s，保存100个5ms窗口和501个不重复1ms采样时刻。全部56个注册状态块、58,155个状态量均记录；有限性、时间覆盖、运行时状态连续性及脉冲事件/计数核对通过，最终checkpoint已保存。没有再次运行模型或调整参数来修饰结果。

| 阶段 | 实测墙钟 / s |
|---|---:|
| 本次授权MATLAB/ISET捕获 | 21.754 |
| 捕获校验与原绑定回执 | 0.441 |
| 成功路径的装配 | 11.308 |
| 已验证缓存的解码/输入绑定 | 8.372 |
| 新背景准备（包含周期checkpoint） | 728.521 |
| 0.5s视频推进本身 | 234.029 |
| 视频推进、记录、checkpoint总计 | 281.469 |
| 其中记录写盘 / checkpoint写盘 | 19.805 / 12.733 |

成功的Python主进程累计1030.543s；加本次独立捕获及校验为1052.737s。加上17.202s元数据失败和更早125.685s许可失败后，已记录计算墙钟合计1195.624s；表中嵌套子项不能重复相加。Python峰RSS为822,833,152 bytes（784.7MiB），MATLAB计算进程峰RSS为1,336,078,336 bytes（约1.244GiB）；0.2s轮询采样的捕获进程树同时峰RSS为1,470,746,624 bytes（约1.370GiB），不同阶段的峰值不能相加。

| RGC | 已记录细胞 | 0.5s脉冲总数 | 单种子每细胞平均观测率 / Hz |
|---|---:|---:|---:|
| midget ON | 22 | 2939 | 267.182 |
| midget OFF | 22 | 3059 | 278.091 |
| parasol ON | 3 | 414 | 276.000 |
| parasol OFF | 2 | 286 | 286.000 |
| S-ON | 3 | 567 | 378.000 |

这些是未校准候选的一次随机种子输出；较高放电率仍保留。条件强度曲线与窗口观测率分别记录，不将本次计数当作重复试验PSTH。

保存值核对发现密集积分端点与下一段初值有极小浮点差异，最大按既有数值单位尺度归一化后为6.101×10⁻¹¹（native R最大绝对差1.259×10⁻⁹td），时间端点差为0。只修正 `ResultStore.history` 的读取比较：用既有单位尺度和1×10⁻¹⁰相对读取容差判断重复端点，保留第一次保存值；不改写原数据，不改模型容差或状态。SI电流使用对应单位尺度，较大不连续值继续拒绝。全段176个信号检查通过；11项回放回归通过（8.81s），包括V/A舍入容差及实际跳变拒绝。原19项checkpoint加载回归的代码未变，之前结果保留。

原界面布局、主配置和11个受保护参数/方程文件SHA一致；本轮仅额外修复结果读取器，未改变界面。实际GUI在1440×900读取完整新结果，cone/HC/BC/AC和5类RGC均显示已记录；1053个HC保留Soma/OPL坐标，单细胞曲线有501点，输入0.2s准确联动绝对神经时间1.5s。截图为 `actual_video_GUI_HC.png`，检查为 `actual_video_GUI_check.json`。

本轮统一证据仍在 `reports/modeling_scope_20261020/actual_video_halfsecond_v1/`：`execution_summary.json`、`five_RGC_actual_video.csv`、`recorded_mean_traces.csv`、100个原始窗口、全状态checkpoint、背景判据、捕获日志及回放回归XML。原 `execution_blocker.json` 是首次许可失败的历史记录，当前完成状态以 `result.json` 为准。最终图含原视频帧和实际cone/HC、BC、AC、5类RGC电位/条件强度，以及原期望cone捕获输入，已逐项查看并保存到 [Library最终结果图](https://chatgpt.com/api/library/files/libfile_43a303f92fa88191b145ceff648b719a/download)；上传回执为 `library_figure_receipt.json`。用 `run_retina_gui.cmd --results "reports/modeling_scope_20261020/actual_video_halfsecond_v1"` 可回放本次保存结果。

### 标准刺激准备与高放电率定位（2026-10-20）

后续试验入口改为显式标准刺激方案，已完成的娱乐视频试验继续作为冻结基线保留。`standard_stimulus_rate_audit_v1/baseline_frozen_identity.json` 固定了基线169个文件的SHA；本轮前后全部一致，11个参数/方程/生产配置文件和原光学源码也未变化。回退包为Temp目录下 `retina_standard_stimuli_before_20261020_v1.zip`。新增输入配置不被生产主配置引用；没有改细胞参数、动力学或研究机制，没有做新的长背景或视频计算，没有启动MATLAB。

**先核对原方法及可用刺激。** [Dacey等2000原文Methods2.2–2.3](https://www.cns.nyu.edu/~tony/vns/readings/dacey-etal-2000.pdf)使用不同大小光斑、环带及漂移正弦光栅；DLP刺激在约1000td背景上以100%对比度、2/4/10Hz调制，并作多周期平均。该文采用200µm/deg，和本项目223µm/deg借用映射不同；本地Fig4B的秒轴及绝对电位仍未闭合。[Latimer等2019的方法](https://doi.org/10.7554/eLife.47012)及本地冻结全文确认，其突触电流数据用60Hz低通的全场LED噪声、0.1ms采样、6s刺激。Cell_6原 `ExcStm/InhStm/SpikeStm` 数组均存在，跨记录模式有8组逐值相同刺激；其物理刺激尺度、电流文件单位和SpikeTimes索引/时钟契约仍需确认。本轮按原顺序导出第一行未缩放刺激与样本索引，不赋予物理单位、不转成100fps视频、不用于拟合。完整方法与来源身份见 `source_review.json`，没有下载额外数据集。

复用已有 `create_platform_example_media.py` 的媒体回读路径、`run_benardete_kaplan_comparison.py:make_stimulus` 的无损FFV1方法，以及 `correct_benardete_kaplan_analysis.py:recover_external` 中线性插值gamma、SPD及V(λ)的模型外部光度计算。旧MJPG彩色梯度及8Hz光斑均保留为原工程记录，未改称原实验刺激。

**已生成的最小输入集。** 新可读配置为 `configurations/standard_stimuli_v1.json`，生成器为 `tools/prepare_standard_stimuli.py`，输出在统一目录 `reports/modeling_scope_20261020/standard_stimulus_rate_audit_v1/stimuli/`。共6段AVI、370帧；全部FFV1逐帧解码RGB哈希一致。每段附媒体身份、时空参数、光度来源、实际量化波形CSV及独立预览配置。

| 输入 | 参数及工程用途 |
|---|---|
| 灰背景 | 0.5s；作为输入级对照，不宣称自动等同已适应神经背景。 |
| **下一次首选：全场亮暗阶跃** | 0–100ms灰，100–200ms亮，200–300ms灰，300–400ms暗，400–500ms灰；目标Weber对比度+0.10/−0.10，非周期刺激、空间频率0。先检查明暗极性、DC及输出工作点，避免同时引入空间/运动参数。 |
| 三种光斑 | 直径0.1°、0.3°、0.6°；同一亮暗时序及对比度；模型owner处定位，未做生理RF最佳位置搜索。用于后续空间募集检查，不据这三个点拟合生理RF。 |
| 移动光栅 | 空间频率1cycle/deg、时间频率4Hz、Michelson目标对比度0.20；orientation0°为竖条向+x移动；100ms灰前后段，中间1s/4周期，总1.2s。属于工程显示，未复制原文的完整条件或周期平均。 |

所有数值均为审阅方法后明确选择的工程参数。帧率100fps，1024×256像素，对应8°×2°视场、0.0078125°/像素；沿用当前owner(0.35mm,0)及223µm/deg项目映射，刺激中心为(1.5695067°,0)。RF/IS映射和有限光学支持仍是既有近似。光斑边缘按实际栅格保存，光栅时空频率均低于采样Nyquist上限。

像素为相等数字RGB码，按固定LCD-Apple各通道gamma/SPD选择灰级，属于灰度显示假设，不是等能光谱、锥隔离或Dacey的LED/DLP光谱。Weber定义为 `(模型Y−灰背景Y)/灰背景Y`，Michelson为 `(Ymax−Ymin)/(Ymax+Ymin)`。选出的灰背景码为188；相对固定浮点光学参考有+0.5442%模型Y量化偏差。实际量化亮/暗Weber为+9.8162%/−10.1936%，逐值保留，没有以像素灰度代替真实光强。既有100td仅是虚拟ISET参考，没有新增硬件亮度、td或R*/s测量。新刺激需建立自己的光学捕获及输入契约，不能复用原娱乐视频捕获；本轮一次MATLAB授权没有被再次使用。

`MediaSource` 对标准刺激sidecar核对媒体SHA、尺寸、帧率、帧数及工程证据类；GUI选择标准刺激后同步声明的视场、片长和起点，布局保持原样，主配置不写盘。实际GUI在0.15s亮段导入成功，100fps、8°视场、0.5s片长正确，未创建模拟会话。3项回归通过（约0.25s），覆盖精确帧边界、亮暗量化、光斑边界、光栅周期/Nyquist、无损回读及sidecar篡改拒绝。`next_input.json` 选定全场亮暗阶跃；其GUI配置仅负责输入预览，不自动切换候选动力学。可用 `run_retina_gui.cmd --config "reports/modeling_scope_20261020/standard_stimulus_rate_audit_v1/stimuli/fullfield_flash_pair.preview_config.json"` 打开预览，不会自动运行。

**高背景率已定位到现有工作点和输出参数组合。** 严格重建原装配及参数哈希、恢复1.3s背景checkpoint后，使用原契约各做5ms背景保持和原视频缓存重放。总新增神经时间10ms、无新背景、总墙钟17.254s，其中装配11.156s、两段推进2.188/2.126s，峰RSS约349.3MiB。视频5ms重放的返回状态、时钟、history及RNG哈希与原第一窗口精确一致。短窗口仅用于数值/因果对照，没有用5ms计数估计平稳放电率。

五类RGC仍共用项目近似 α=150Hz、μ=−70mV、β=5mV、EL=−65mV、1ms采样，以及−12/−8/−4mV三bin history。复用原三bin Bernoulli采样器的8状态精确代数：**零突触兴奋/抑制、V=EL时**，零history条件强度已为196.989Hz，包含原history后的理论平稳计数率为135.382Hz；这里没有把“零突触电导”解释为黑暗光刺激，也没有提出动物正常率目标。

| RGC | 背景平均V / mV | gE / gI均值 / nS | 固定背景V的原采样器理论计数率 / Hz |
|---|---:|---:|---:|
| midget ON | −53.489 | 0.544 / 0.230 | 273.773 |
| midget OFF | −53.766 | 0.531 / 0.232 | 270.344 |
| parasol ON | −53.534 | 0.543 / 0.232 | 273.214 |
| parasol OFF | −53.720 | 0.533 / 0.232 | 270.912 |
| S-ON | −44.887 | 1.078 / 0.232 | 379.192 |

上游持续突触驱动将V相对EL抬高约11.2–20.1mV，原被动膜电导平衡公式与保存V一致；持续背景5ms漂移极小。该高率已在视频之前存在，最初5ms视频相对背景的最大条件强度变化小于0.0001Hz；这不排除更长时段的实际视觉响应。没有发现应据此改单位、事件计数公式或强行压低μ的工程依据。RGC输出参数组、上游DC/电导尺度和实际观测之间仍缺独立约束，不能通过同时调整多个增益得到可辨识的生理校准。本轮没有参数候选回装、拟合或生产默认切换；下一步应补齐同条件V–spike、背景工作点及电流/刺激单位、时钟和独立测试数据，再按既有白名单分别校准。

统一证据为 `source_review.json`、`scope_integrity.json`、`stimulus_tests.xml`、`GUI_import_check.json` 及 `rate_audit/` 下的参数快照、原始短控制、单位/几何布局、`operating_point.csv` 和 `result.json`。原始数据和旧结果全部保留。已保存 [Library标准输入与定位总览图](https://chatgpt.com/api/library/files/libfile_266dd8e83e64819187fd503f9217e649/download)，图明确区分工程输入、保留模型代数和原视频单种子观测。

### RGC电位—条件强度来源核查（2026-10-20）

**结论：未发现可证明的公式、单位或采样实现错误；约135Hz是当前未标定参数组合的可预测结果。** 方程自洽不能证明生理适用性。本轮仅核对来源及求解原采样器的有限状态代数，未调整μ/α/β、膜参数、history、dt、候选、主配置或界面，未做拟合、神经推进、随机试验或MATLAB运行。

当前 `retina_sim/inner.py:HistoryRGC` 使用物理SI量：

```text
C*dV/dt = gL*(EL-V) + gE*(EE-V) + gI*(EI-V)
Vtotal = V + sum_j h[j]*count_in_strict_past_bin[j]
lambda = alpha*log(1+exp((Vtotal-mu)/beta))           # Hz
p = 1-exp(-lambda*dt)                              # 每bin至少一次的概率
count ~ Bernoulli(p)                               # 0或1，原机制
```

V、μ、β和h均为V，α为Hz，dt为s；history是每个过去bin的二值计数，h是每次计数贡献的电位，不是对Hz采样值积分，故不再乘dt。最新一档对应上一bin，不包含当前抽样结果。当前只有graded V连续状态，spike history仅进入输出强度，不重置或反馈V。μ是softplus偏移，**不是过线才放电的硬阈值**；在 `Vtotal=μ` 时强度仍为 `α*ln(2)`，当前约103.972Hz。

零突触电导 `gE=gI=0` 时，稳定V=EL=−65mV，而μ=−70mV、β=5mV，因此零history自变量为1。保留α=150Hz、h=−12/−8/−4mV、dt=1ms，精确计算为：

| 条件或统计量 | 结果 / Hz |
|---|---:|
| 无history的条件强度 | 196.989253 |
| 无history的二值计数率 `p/dt` | 178.800541 |
| 含原三bin历史的平稳平均条件强度 | 147.669248 |
| 含原三bin历史的平稳二值计数率 | **135.382173** |

这里用8种历史状态构造转移矩阵，解 `πP=π`，再分别算 `π·lambda` 与 `π·p/dt`；残差5.55×10⁻¹⁷，没有以随机模拟或短窗计数代替平稳解。条件强度与计数率不能混用。零突触电导不等于黑暗光刺激，也没有被当成动物正常放电率。SI到mV的全量转换、或同时使用 `U=V−EL` 和 `μrel=μ−EL=−5mV`，均保留同一自变量与率；仅移动μ属于参数变化，μ变得更负会提高固定V下的强度。

**来源边界。** [Latimer、Rieke与Pillow 2019原文](https://doi.org/10.7554/eLife.47012)式11–12/28使用膜电位加过去spike滤波，再除以β作softplus。冻结XML的MathML已确认分母，不能把丢失分数线的纯文本读成乘β。原文α=90Hz、μ=−53mV、β=1.67mV是两只ON-parasol细胞动态钳重复记录的平均电位—spike关系拟合；其膜条件固定为EL=−60mV、EE=0、EI=−80mV、去电容尺度的gl=200s⁻¹，采样为0.1ms或0.083ms。该gl不是200S，也不独立给出物理C；不能把该组值当成五类RGC共同的已测生理参数。

[作者2022年Python实现](https://github.com/pillowlab/CBEM/blob/7beb7e72b0678192f0bc98392a678cdff9601791/pyCBEM/RGC_CBEM.py)保留90/−53/1.67、历史项以mV加到V后除β，并使用 `1-exp(-rate*binSize_ms/1000)` 的二值抽样/截断Poisson似然；[utils](https://github.com/pillowlab/CBEM/blob/7beb7e72b0678192f0bc98392a678cdff9601791/pyCBEM/utils.py)的指数时间项同样将ms除1000。其拟合history使用约190ms的样条基。已缓存两文件字节与固定commit树中的git blob一致；这是后续作者实现，未称为2019发表时版本。

还核对了[2020年发表存档的MATLAB初始化](https://github.com/elifesciences-publications/CBEM/blob/cfe41c4519fd56bd5aaaa81f156357ebae101d8f/code/CBEMstructHandler/setupCBEMsimple.m)及[采样器](https://github.com/elifesciences-publications/CBEM/blob/cfe41c4519fd56bd5aaaa81f156357ebae101d8f/code/simulateCBEM.m)：最终初始化为 `c*softplus(a*V+b+sph)`，c=90、a=0.45mV⁻¹、b=53a，对应μ=−53mV而β=1/a≈2.222mV，history的sph在无量纲自变量内；换成电位history须除a。这个初始化与原文β及后续Python值不同，且拟合入口还组合/调整leak项，不能混取数值。[示例](https://github.com/elifesciences-publications/CBEM/blob/cfe41c4519fd56bd5aaaa81f156357ebae101d8f/exampleScript.m)明确dt为秒，虽setup头注释写成ms；原采样器仍是二值 `1-exp(-dt*rate)`。因此不能只根据原文“Poisson”用词，把当前Bernoulli改成多事件Poisson当成纠错。此处仅在线读取源代码，没有启动MATLAB；受限shell无法直接下载的字节未假称本地验证。

当前五类的C=60pF、gL=2nS、EL=−65mV、α=150Hz、μ=−70mV、β=5mV及三bin history，在实际参数快照和可读registry中全部标为 `PRIMARY_PATHWAY_project_profile / project_approximation / Uncalibrated/retractable`。来源资产的 `identity_verified` 仅验证文件身份。即使EE/EI数值碰巧同原文，也不自动取得测量身份。C/gL=30ms，和原文归一化gl的5ms不同。V0是上游背景电导条件平衡推导值，不是静息电位测量。局部Git只追溯到 `1e4dec7ce8ba9cec801f122ba91ae33a9c141a46` 导入，导入fixture已含同一150/−70/5和原history；没有找到更早的作者参数依据。可确认的是借用了同类机制结构，当前数值仍属于项目假设。

**可执行修正选择与确认边界。** 当前保留原fixture并记录限制。若下一步要校准输出，应先选明确RGC类及同条件V—spike记录，闭合绝对电位原点、时间/索引、重复平均约定、bin/history及背景电导工作点，再使用既有白名单的μ或μ+β拟合组，按独立train/validation/test评估；单一电位或任意“正常Hz”目标不能识别完整输出函数，也不与上游多个增益一起自由拟合。已有Cell_6数据尚有刺激/电流单位及spike时钟缺口，不能据此伪造匹配条件。

若要做原文/作者参考候选，需先明确采用原文、2020 MATLAB还是2022 Python，以及配套膜归一化、history单位/支持、dt和细胞条件；修改α、反转电位、history形状/支持或dt均超出现有受限拟合白名单，需要明确科学范围授权。该候选应独立保存、使用新身份，不能自动替换五类生产默认。改变dt但保留三样本history还会改变物理支持长度，不能作为纯数值修复。当前无足够证据选择任何新的系数。

统一证据在 `reports/modeling_scope_20261020/rgc_voltage_hazard_source_audit_v1/`：`result.json`、`eight_state_stationary.csv`、`parameter_provenance.csv`、`primary_source_review.json`。新增 `tools/audit_rgc_voltage_hazard_sources.py` 可离线复核生产强度函数与独立代数、单位/原点不变性、五类快照和registry、原文分数结构、作者blob身份及冻结基线。一次运行约1s，7组检查通过；原视频169文件及11个受保护方程/参数/配置文件SHA一致，未重复先前随机采样/工程审计测试。

### 原始V—spike数据续查与标定准备（2026-10-20）

在明确授权继续μ或μ+β标定后，先复用上述 `primary_source_review.json`，再核对原文Methods/图3附件、公开数据描述、作者仓库及原始采集论文。**已完成数据适用性检查，但未取得匹配的原始膜电位，因此没有执行新的动物数据拟合，也没有生成标定候选。** 这次没有使用目标Hz压低输出，没有恢复旧α+μ尝试。

本次补齐了之前容易混淆的两组条件：

| 记录/模型 | 来源和条件 | 本次可用数据 |
|---|---|---|
| 图3输出非线性记录 | [Latimer2019](https://doi.org/10.7554/eLife.47012)引用Cafaro–Rieke2013的两只ON-parasol动态钳细胞；19s，18/24次重复；current clamp，注入电流反馈延迟100µs，EE=0、EI=−90mV | 原文方法和汇总曲线；在已检查公开资产中没有找到原始逐次V—spike文件 |
| 原文CBEM归一化模型 | EL=−60、EE=0、EI=−80mV、gl=200s⁻¹ | 方程和作者实现；这些固定模型设置不是上述动态钳注入参数 |
| Figshare六细胞记录 | Trong–Rieke2008的ON-parasol；cell-attached spike与之后分开做的E/I voltage-clamp记录 | 有电流、spike和刺激；不能冒充图3两细胞的动态钳V—spike数据 |

[Figshare9636854公开页](https://figshare.com/articles/dataset/ON-Parasol_RGCs_for_the_conductance-based_encoding_model/9636854)说明六个文件提供刺激、电流与spike。冻结元数据列出六文件；实际再次检查的Cell_6通过SHA/MD5及精确schema核验，字段只有 `ExcCurrents/ExcStm/InhCurrents/InhStm/SpikeStm/SpikeTimes/fn`，没有膜电位。其他五文件没有下载，未声称逐字节排除其未公开描述的字段。由电流和假设C/gL计算的V是模型预测，不能作为独立实测V来标定同一模型。

图3XML仅有图像，没有原始数据附件；全文附属media清单为透明报告表。作者CBEM的固定树/Data目录只有两个example文件，当前README明确称MATLAB示例为模拟数据。[Rieke-Lab的15个公开仓库索引](https://github.com/orgs/Rieke-Lab/repositories?type=all)主要为采集协议与仪器校准包，没有从名称/说明定位到该两细胞数据；没有将元数据检查称为遍历了全部源码。eLife网页遇到JavaScript挑战，部分API及旧实验室站点不可访问；使用可读PMC原文、公开Figshare页面和GitHub完成限定检查，没有绕过访问限制。此结果只说明已检查资产未提供匹配记录，不证明作者未保存或其他位置绝无原始数据。

[Cafaro–Rieke2013原始论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC3687519/)进一步说明动态钳使用共同E/I电导缩放，必要时注入恒定偏置电流；原文报告电位做液接电位修正。两只图3细胞的具体缩放、偏置值、细胞编号和电位修正映射没有取得，不能把一般采集条件或其辅助维持的自发率变成项目目标Hz。该采集论文的队列包括多个macaque物种，尚未闭合两细胞的个体映射。已给 `primary_source_review.json` 增补明确的 `recording_dynamic_clamp_conditions`，并将原 `reported_membrane_conditions` 标明为CBEM设置；当前参数没有因此增加或扣除10mV，也没有将EI切到−90mV。

2020 MATLAB与2022 Python边界沿用前次精确核查：前者初始化的斜率0.45mV⁻¹对应β≈2.222mV、history位于无量纲自变量，后者与原文采用β=1.67mV、history为电位。它们不是同一可直接替换的数值快照；两者的数据目录都未给出所需原始V—spike证据。完整作者参数、history形状或膜参数迁移依然超出本轮仅μ/μ+β白名单。

旧 `latimer_cell6_rgc_v1/calibration_decision.json` 和独立审阅已经拒绝过一次α+μ条件拟合，且其验证结果已被查看、单位/时间索引及电流修正仍有缺口。本次没有重新选择这份holdout、扩大边界或把旧结果转称新独立验证。没有匹配输入时，标准亮暗阶跃不能代替动物V—spike数据，故候选短刺激对照也尚未执行。

已落实的最小交付在 `reports/modeling_scope_20261020/rgc_v_spike_intake_v1/`：

- `source_inventory.json`：逐源检查范围、结果、未检查范围及访问状态。
- `public_file_inventory.csv`：六个公开文件的名称、字节、MD5及实际检查状态。
- `minimum_primary_data_handoff.json`：可直接交给数据提供方的原始数组/元数据清单和后续分析约束，不是参数候选。至少需同钟逐次物理V及spike、时间/滤波/检测定义、两细胞及条件标识、液接修正是否已施加、实际注入电导/偏置信息。
- `result.json`：明确已执行与未执行阶段；状态为 `NO_MATCHED_PRIMARY_V_SPIKE_DATA_CALIBRATION_NOT_PERFORMED`。

新增 `tools/inspect_latimer_v_spike_intake.py` 可离线重复本轮检查。7组数据/来源/身份检查执行通过，冻结视频169文件、11个模型/配置文件及此前输入/GUI源码SHA一致。没有生产参数、默认值、界面或机制修改；没有额外数据集下载、MATLAB、新神经推进、拟合或作者联系。无新权限请求、拒绝、待批或不确定执行，也未重试被拒绝的权限操作。

拿到原始数据后的最小范围已写入交付契约：仅ON-parasol，固定现有α、膜条件、history和dt；先声明电位采样/二值计数转换和分割，分别在各分割内构造重复平均V，避免把holdout试次混入训练预处理。若只有单个19s重复波形，则预先按时间块跨所有重复分组并保留history隔离，限定为该条件内的V—spike标定；要作跨细胞/新刺激声明仍需独立记录。随后才在来源支持或事先明确的边界内拟合μ或μ+β、保存独立候选及测试指标，再考虑既定短标准刺激对照。缺原始V时这些阶段保持未执行，不能以现有工程审计通过替代。

### 独立ON-parasol作者参考实验（2026-10-20）

用户另行授权独立原模型复现实验后，实际完成已核验部分的参考复算，未恢复Cell_6拟合。核心环境无JAX且不升级环境，因此采用NumPy/SciPy按固定来源逐式重算，**没有直接执行作者JAX notebook或MATLAB程序，也没有称为完整作者拟合流程复现**。计算入口为 `tools/run_on_parasol_author_reference.py`，输出在 `reports/modeling_scope_20261020/on_parasol_author_reference_v1/`；科学计算仍用retipath_v1，base Python仅读取保存数组并绘图。

固定来源为[作者2022 Python commit](https://github.com/pillowlab/CBEM/tree/7beb7e72b0678192f0bc98392a678cdff9601791)和[2020发表存档](https://github.com/elifesciences-publications/CBEM/tree/cfe41c4519fd56bd5aaaa81f156357ebae101d8f)。缓存Python源码字节与固定树git blob一致。分别保留三条命名参考：①原文归一化被动膜方程的精确ZOH解；②2020 `simulateCBEM.m` 的电压矩阵算法，在原文膜参数下复算，未声称是整个setup/fit入口的输出；③2022 `utils.getVoltage` 的首项与下对角索引字面复算。静态输出图另显示2020初始化β≈2.222mV，不与原文/2022的1.67mV混合。

原文/2022参考保持EL=−60mV、EE=0、EI=−80mV、gl/C=200s⁻¹、α=90Hz、μ=−53mV、β=1.67mV和0.1ms二值观察步长。这里的gl已省略电容尺度，不是物理200S。项目对照从未改动的ON-parasol参数记录读取：C=60pF、gL=2nS、EL=−65mV、EE=0、EI=−80mV、α=150Hz、μ=−70mV、β=5mV及1ms采样/原三bin history。**孤立测试的初值显式置于各自EL、history初始为空，不使用整网背景V0或声称复制其已适应状态。** 生产V0推导规则没有修改。

运行了0.5s显式合成电导钳，每段100ms，归一化 `(gE/C,gI/C)` 依次为 `(0,0)、(40,0)、(40,40)、(0,0)、(10,10)`，单位s⁻¹。它是声明在脚本中的工程输入，用于同时检查兴奋、抑制、恢复和时间索引，不是视觉刺激、原作者滤波输出或动物记录，也没有调参寻求特定Hz。项目以自身C将同一g/C换成物理S输入；没有为原文捏造物理C。共5000个0.1ms区间；项目原history期望值在原1ms采样上精确边缘化，不做随机Monte Carlo。作者拟合history和刺激滤波器未取得，作者动态/静态参考明确为h=0；图中与项目原history的曲线分开标示，不把它们当作完整同条件生理性能比较。

同一电位、零history时的实际输出如下。两种步长下 `p/dt` 不等于λ，不能直接混用：

| V / mV | 原文/2022 λ / Hz | 项目λ / Hz | 原文0.1ms二值率 / Hz | 项目1ms二值率 / Hz |
|---|---:|---:|---:|---:|
| −65 | 0.068140 | 196.989253 | 0.068139 | 178.800541 |
| −60 | 1.350776 | 319.039202 | 1.350685 | 273.152945 |
| −53 | 62.383246 | 514.924271 | 62.189067 | 402.454155 |
| −40 | 700.636249 | 900.371353 | 676.655017 | 593.581293 |

这确认了当前参数与作者输出映射有显著差异，同时展示了较大λ下采样步长本身可改变计数率的相对大小，不能据此将作者参数宣布为任意条件下“正确的低率”。合成电导钳中，前100ms各自EL条件下作者h=0的期望二值率为1.350685Hz，项目原history从空历史起的100ms均值为135.953579Hz；后者不是之前无限时平稳135.382173Hz的另一个估计。兴奋段作者h=0均值161.090123Hz，项目原history期望均值496.234728Hz；这些是既定工程输入的模型输出，不是动物验证指标。

**作者版本的电压实现差异也被保留。** 2020电压矩阵与原文ZOH方程在相同条件下全段最大差1.421×10⁻¹³mV。固定2022源码使用 `C[0]+=V0`，并将 `-exp(-G*dt)[:-1]` 放到下对角；按[JAX官方下对角定义](https://docs.jax.dev/en/latest/_autosummary/jax.lax.linalg.tridiagonal_solve.html)，第j行使用前一区间的衰减。原方程则用当前区间衰减，首项需衰减初始V。字面复算的零输入首bin为−61.188080mV，原方程/2020算法为−60mV；本合成输入转换时仍有差异，首50ms后最大约0.424mV。该结论是固定源码数学形式的比较，尚不是直接JAX运行验证，不扩大成对原文全部拟合结果的否定。没有静默修正作者源码或生产求解器。

新增4项针对性测试通过（约0.145s）：恒定输入闭式解、2020矩阵等价性、2022首项与前一区间索引的明确保留、2020初始化参数化独立性。参考科学计算墙钟0.7417s；输出均有限，运行前后187个保护文件身份一致，覆盖原视频、模型参数/方程、配置及此前输入/GUI源码。本轮没有整网推进、背景准备、训练、拟合、环境安装、权限请求/拒绝或MATLAB运行。

可复核交付为 `result.json`、`reference_traces.npz`、`zero_history_mapping.csv`、`voltage_versions.csv`、`common_bin_start_comparison.csv`、`reference_checks.xml`、PNG/SVG比较图及 `scope_and_checks.json`。图包含相同V的映射、合成输入、各自膜电位、λ、二值期望率及版本差异，已人工查看并保存到 [作者参考比较图](../reports/modeling_scope_20261020/on_parasol_author_reference_v1/on_parasol_author_reference.png)。原JAX notebook、作者模拟MAT例子的重新拟合、作者拟合history/滤波器，以及实测V—spike拟合均未执行：前两项缺本机运行依赖/缓存示例，后两项缺对应原始输入或记录。独立部分参考实验完成，不等于整网或ON-parasol实测标定完成，也不改变五类默认。

### ON-parasol文献迁移候选与短整网对照（2026-10-20）

用户另行明确授权参数候选修改后，实际接入并运行了 **仅ON-parasol的文献输出参数迁移候选**。新增 `configurations/on_parasol_literature_candidate_v1.json` 将该类α/μ/β从150Hz/−70mV/5mV改为原文与2022作者实现一致的90Hz/−53mV/1.67mV；默认关闭、未拟合、未获生产采用。它引用原 `retina_platform_v1.json` 主配置及既有 `bc_ac_contact_first_hypothesis_v1.json`，不建立另一套生产主配置。原文身份、参数和条件沿用前述已冻结审计，没有再次检索或重复独立映射复算。

接入使用原 `merge_parameter_package` 重建机制，新增独立的 `rgc-source-reference-candidate-1.0` 分派和 `retina_sim/source_reference_candidate.py`。该路线只接受指定ON-parasol组件、三项确定文献值及其单位/来源身份；没有扩展μ/μ+β等原拟合白名单，也不把迁移伪装成拟合。仍使用原 `HistoryConductanceRGC` 工厂、状态和采样器；关联电导适配器仅更新owner profile引用。其他41组件的完整参数记录/参数profile保持一致，连接与gap不变。新profile和装配身份使旧参数检查点失效，必要回归已验证跨参数恢复被拒绝。

**膜条件及适用域差异明确保留。** 候选仍为C=60pF、gL=2nS、EL=−65mV、EE=0、EI=−80mV、原−12/−8/−4mV历史和1ms采样，V0仍按原上游背景电导推导/平衡，没有平移电位原点、改膜时间常数、改history或以目标Hz调参。原文CBEM条件为EL=−60mV、归一化gl=200s⁻¹及0.1ms采样，原动态钳记录EI=−90mV且有细胞级注入条件；拟合history未取得。原记录来自外围视网膜，本模型ROI在0.35mm附近，不能据细胞类型相同就声称区域/膜条件匹配。三项迁移值在项目中仍标 `project_approximation`，不是本项目实测参数。

**实际整网执行。** 使用主配置的冻结atlas图，42组件、23群、20,422细胞、58,155动态量、50条link和4类gap，全部耦合机制经 `HybridRunner` 推进，没有减少密度、剪断反馈或用离线V映射替代推进。两组各自冷启动，使用实际声明背景的原native/rod平衡seed及原RGC电导平衡初始化，再各推进100ms短洗入。未调用正式最短1s的背景收敛验收，不宣称完全适应或随机平稳。

刺激为500ms全场根输入阶跃：每段100ms，灰/+10%/灰/−10%/灰；在每个既有td或R*/s消费者自身域内按比例调制。该路线复用已有显式根输入扰动接口，保持两组输入契约逐字节一致。**它是工程标准输入，不是新标准视频的真实ISET捕获，也不声称这些独立消费者对应同一个校准物理场景。** 标准视频没有兼容的ISET捕获包；本轮未重试此前被拒的MATLAB捕获动作，未将旧视频捕获重标为新输入。

因首个灰段仍有约0.010mV漂移，又为两组分别执行500ms配对灰保持控制。每个控制严格恢复该组自己的100ms准备检查点，装配身份匹配且checkpoint回写逐字节一致；没有跨参数混用状态。四个完整分支加两次短洗入共推进2.2s模型时间。亮暗组运行墙钟561.56s，配对灰控制459.39s，成功科学执行合计1020.96s。各段输出有限，控制与刺激都保留全状态检查点及分段原始输出。最初一次记录器在推进前因16MiB小于22.7MB估计量而拒绝，改为64MiB后复算冷初态并确认原root checkpoint逐字节一致；启动失败记录保留，没有重复已经推进的试验。

ROI内有3个ON-parasol细胞。下表背景为完整500ms灰控制的每细胞均值；期望率在实际推进得到的V上精确边缘化未来三bin历史，条件于各自准备历史，首个已抽样bin计数固定。亮/暗幅度为同一绝对时刻的刺激减配对灰控制，再作相应100ms阶段平均。

| ON-parasol输出 | 旧参数 | 文献迁移候选 |
|---|---:|---:|
| 灰背景期望二值计数率 / Hz | 273.222839 | 42.007734 |
| 灰背景实际单种子计数率 / Hz | 273.333333 | 44.666667 |
| 亮段期望率增量 / Hz | +0.254904 | +0.343447 |
| 暗段期望率增量 / Hz | −0.098443 | −0.131560 |
| 全序列期望率响应绝对峰 / Hz | 0.480812 | 0.651464 |

两组平均灰背景V均为−53.539042mV；配对扣除后的亮段V增量+0.020404mV、暗段−0.007844mV，人口平均V响应绝对峰0.038848mV，两组完全相同。刺激分支与灰控制分支的连续终态分别在参数组之间逐位一致；其余四类RGC的V/gE/gI/条件强度/实际计数/窗口率在两组间逐位一致，灰控制另核对24个信号。变化来自ON-parasol输出映射，没有变动上游动力学或电位。

**基线下降不代表视觉响应已校准。** 本工程输入下，亮暗期望增量仍小于1Hz；当前单种子中，ON-parasol各阶段实际计数与各自配对灰控制的计数差均为0。它没有分辨出这一级响应，不能把解析期望曲线称为已观测动物响应，也不能把42Hz称为指定“正常率”。未为放大响应调整其他增益/膜条件，没有使用Cell_6记录、旧holdout、拟合或训练。该迁移实验提供了可运行候选和实际整网效果，实测校准与独立生理验证仍未完成。

11项必要回归通过（32.71s），覆盖指定三项变更、其他组件保持、旧检查点拒绝、源参数/单位/目标/默认篡改拒绝及原拟合提案回归。最终187个保护文件SHA一致，包括原视频169文件、11个参数/方程/主配置文件及此前输入/GUI源码。生产默认、其他四类参数、GUI和已有视频结果不变。没有新的审批拒绝、待批或不确定执行，没有启动MATLAB或升级环境。最初pytest自动临时目录不可写的问题通过指定授权Temp目录解决，没有扩大权限。

统一交付在 `reports/modeling_scope_20261020/on_parasol_candidate_network_v1/`：`result.json`、`matched_control_result.json`、`ON_parasol_summary.json/.csv`、`five_RGC_phase_comparison.csv`、实际参数快照、装配/输入身份、11项回归XML、每组root/prepared/final检查点、刺激和配对灰控制原始分段/完整NPZ及响应差数组、`scope_and_checks.json`。可复现入口为 `tools/run_on_parasol_candidate_network.py` 和 `tools/run_on_parasol_matched_gray_controls.py`；均只在实验目录写结果，主配置文件不写盘。最终PNG/SVG比较图已查看并保存到 [整网候选比较图](../reports/modeling_scope_20261020/on_parasol_candidate_network_v1/on_parasol_candidate_network.png)，图明确区分实际单种子计数、模型条件强度及基于实际V的计数期望。
