# 阶段 C 实际交付报告

已批准范围：独立 Chen2024 外周 >20° L/M cone 非线性外段电流模块。阶段 A/B 保留；全视网膜主线仍未完成。实现与数值源码对照完成，真实来源数据复现因官方文件下载阻塞未完成，独立生理验证未运行。

## 实现与范围

入口 `retina_sim/chen2024.py`；显式选择 named reference / numeric fixture，禁止 full_physiology。输入为每 cone 总光异构化速率 R*/s（含背景），0≤u<50,000；输出绝对负外段 pA，基线差分需显式同单位同证据基线。固定作者 dt=0.0001 s、float64、原 nonlinear 离散更新，不进入 linear/inverse 分支。

R/P/G/C 四个主要动态状态与 S 代数缓存跨绝对时间、chunk、checkpoint 保存；Snew 用 Cnew，Gnew 用 Sold。流式 N 区间消费 N 输入并输出 N+1 节点；作者 N 样本包装只消费前 N−1 项，明确保留源码最后输入不消费语义。输入 rate×dt 只转换一次。

参数来自历史 peripheralPrimateCone 分支；作者三 Macaca 物种混合来源必须显式声明。中心、过渡区、S cone、单一物种默认参数、电压转换、terminal Ca、HC/BC、bleaching/regeneration 与全部慢适应均未覆盖。没有训练或服务。

## 来源、参数与单位证据

仓库 https://github.com/chrischen2/photoreceptorLinearization ，固定 commit `1bcf7c9a6a6814ef39d2c39d82cf8dd1b5b39db2`，MIT。四份实际下载文件逐一核 Git blob 与 SHA256，demoMain 仅身份审计、不执行。每个固定参数都有注册表证据条目、原位置/注释/字面量、单位、shape、适用域与用途。作者参数归类 author_assumption，源式派生量标 derivation；不冒称逐项生理实测。

R/P 的有效活动量采用方程推导的形式单位 s^-2 / s^-1；eta=s^-2、gamma=s^-2/R* 是归一化推导，源 eta 注释 1/sec 的量纲差异保留说明，未伪造严格分子实测单位。G/C 为 uM，S 为 uM/s，输出 pA。timeStep 为数值设置。

| 实际源码文件 | Git blob SHA1 | SHA256 |
|---|---|---|
| src/model/BiophysModel.m | `8117c9f157fc8768993e41c521efe23aff8fc7dc` | `fb5ff0a6880d5c78421990981dcf31932b0d415d9d216deb73a64753e5f37937` |
| src/model/initPhotoreceptorParams.m | `086e324f5f5a38f04249c63b1650f68fea4d3586` | `dbff286bc0bda0fbc0aa87775f83285f5dc93d5028559d19efb64651b9ef4de0` |
| src/model/demoMain.m | `322935d64a4237153923ef26117e41a6d72f5919` | `f448fe4f9c0c1d3aa1a443240ff19fb6ff43e31912f57abb7e3ccee54bf2e88d` |
| LICENSE | `8ca78f6d2010e1fdbc38127789f8e6bb96afb9b8` | `2a0fe3991f0323bd12ab145d0328acbf9dca63545ce6f569df69cdb319c848dc` |

当前来源账本：16 来源、21 参数、2 profiles。`sources/registry.json` SHA256 `68dc55b8a55671cee62da6935d99fc4293af881acd1abd9673d29c3199215bb9`。

## 真实执行的验收

最终聚合：`reports/20261010T115050664306Z`；A54、B8、C32 全通过，四条检查命令退出码均 0，AST 解析无错误。未配置 lint/typechecker，不声称执行。首轮 C31 和修复后 C32 的真实日志均保留，容差未放宽。

原 MATLAB R2026b 实际执行历史 BiophysModel 和 initPhotoreceptorParams，biophysFlag=1。版本与完整命令见 `reports/chen2024_matlab_execution.json`，退出码 0；导出 MAT `reports/chen2024_matlab_oracle.mat` 仅为 synthetic_fixture 数值 oracle，绝不是 Dryad 动物数据。

原 MATLAB oracle SHA256 `cd26255840512068b32dd563a9e71a867d767b9c81183428c9d96c78eceb831f`。独立 scalar 源式 oracle 不调用生产 `_step`，用于状态顺序对照；它不构成独立生理验证。

| synthetic fixture | 原 MATLAB 电流最大绝对误差 pA | 最大相对误差 | scalar 五状态最大误差 | 样本数 |
|---|---:|---:|---:|---:|
| dark | 0 | 0 | 0 | 5000 |
| background | 7.3896444519050419e-13 | 2.0915727461515425e-15 | 0 | 40000 |
| weak_flash | 1.1368683772161603e-13 | 2.6515880465246114e-16 | 0 | 18000 |
| step_flash | 7.3896444519050419e-13 | 2.0915727461515425e-15 | 0 | 30000 |
| high_rate | 5.6843418860808015e-14 | 2.8070261030716738e-16 | 0 | 6000 |

预先固定 MATLAB 容差：绝对 ≤1e-8 pA、相对 ≤1e-10；scalar 状态 ≤1e-8（各声明单位）；chunk/checkpoint 要求逐点相同。
暗态相对默认 -428.75 pA 的绝对残差 2.8421709430404007e-13 pA；持续背景 3.9999 s 后最后 1000 样本 scaled 单步状态残差 0，预定阈值 1e-8；chunk/checkpoint 最大误差 0。
synthetic 闪光非线性残差 0.23894008765773833 pA，只说明该固定 kernel 的非线性，未对动物反应做符合度判断。

错误单位/shape/来源/参数值、中心/S/未知物种、负值/非有限/≥50k、错 dt/solver/身份/off-grid、linear/inverse/pA→mV 等拒绝门禁均在实际测试中通过。独立限定审查发现 rate 查询 round 违背 ZOH，已改半开区间边界查找并补非网格/边界回归，审查复核唯一发现修复；详见 `docs/chen2024_review_report_zh.md`。

实际检查命令：

```text
C:\Users\PC_User\anaconda3\envs\retipath_v1\python.exe -B D:\macaque_retina_simulator\tools\extract_packer_cone_density.py --verify
```
退出码 0；日志 `reports/20261010T115050664306Z/check_0.log`。

```text
C:\Users\PC_User\anaconda3\envs\retipath_v1\python.exe -B -m pytest -p no:cacheprovider D:\macaque_retina_simulator\tests\test_stage_a.py -q
```
退出码 0；日志 `reports/20261010T115050664306Z/check_1.log`。

```text
C:\Users\PC_User\anaconda3\envs\retipath_v1\python.exe -B -m pytest -p no:cacheprovider D:\macaque_retina_simulator\tests\test_stage_b.py -q
```
退出码 0；日志 `reports/20261010T115050664306Z/check_2.log`。

```text
C:\Users\PC_User\anaconda3\envs\retipath_v1\python.exe -B -m pytest -p no:cacheprovider D:\macaque_retina_simulator\tests\test_stage_c.py -q
```
退出码 0；日志 `reports/20261010T115050664306Z/check_3.log`。

MATLAB 原执行命令：

```text
C:\Program Files\MATLAB\R2026b\bin\matlab.exe -batch run('D:/macaque_retina_simulator/tools/chen2024_matlab_oracle.m') -logfile D:\macaque_retina_simulator\reports\chen2024_matlab_oracle.log
```

## Dryad 真实数据取得情况

官方 DOI https://datadryad.org/dataset/doi:10.5061/dryad.q2bvq83vg ，版本 3 / 325663。实际电脑保存官方网页与 file 元数据；网页和 API 元数据均 HTTP200，官方实际页面标 CC0。元数据 SHA256 `ad747e0169b651f0e5b1d731518336dba2bc72dd6d6f141ccde1fdb5bb04a177`。

**MAT 未取得、未核本地 MAT 哈希、未读取 MAT 变量或细胞分组。** 以下只有发布值，不能当本地实得值：

| 文件 | 官方发布字节数 | 官方 SHA256 | 本地文件/实际 SHA |
|---|---:|---|---|
| Figure2-SourceData1.mat | 41490462 | `6724e2edd8e8708456f32900a14efae88192a5ba597365046d0031eb543d1331` | 未取得 / null |
| README.md | 1357 | `f25cb72716327f5fc08cfe26cc423bc3364f8f331900dd1972eeb664c1911d03` | 未取得 / null |

实际普通公开请求：file_stream 返回 403，API download 返回 401；Chrome CUA 启动器因 Windows 平台目录缺失在导航前失败。未登录、付费、取得凭据或绕过 WAF；未重复同一拒绝路径。具体 URL、时间与状态保存在 `reports/chen2024_download_attempt.json`。

网页 README 给出的 PrimateConeResponses 等变量名仅为预期；`observed_mat_variables` 与 `observed_cells_and_groups` 都为 null。网页描述 cone 反应为 4–5 次重复平均、暗电流归一化 voltage-clamp、dt=0.0001 s、输入 R*/s；逐 cell 物种、精确偏心度、绝对 pA 与 holdout 未核均保留 null。不能把这些网页说明变成实际 MAT inventory。

来源数据复现：NOT_RUN；独立生理验证：NOT_RUN。即使后续取得真实数据，该数据也是作者定参数来源，只能称来源数据复现，不得充当独立验证。下一步依赖是通过正常官方网页取得真实 MAT/README 并提供可读路径；先核 published hash，再逐项读取真实变量、shape、cell/分组及记录条件，元数据不足则停止相应生理宣称。

## 主线完整度

20 项完整需求保留，full_physiology 仍拒绝，15 项阻塞保留。A02 仅新增外周 L/M 子项 SOURCE_PARITY_VALIDATED，父 A02 仍 BLOCKED。中心—过渡—周边连续模型、标定光路、完整细胞连接、位移、区域动力学、眼动光路和独立验证等主线未完成。真实生理符合度、数值测试通过与实现完整度分别记录。

源码/审计/合同/测试/日志/覆盖表与报告均在新项目内；旧实验、环境及 A/B 历史验收目录没有覆盖。`reports/chen2024_delivery_manifest.json` 保存本次文件 SHA256 清单（排除清单自身与旧历史清单）。
