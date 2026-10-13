# 来源审计档案与50k勘误

2026-10-10 已批准门禁勘误：作者复现模式不再用瞬时50k作拒绝条件；默认工程封套保留，正值高峰记录不含漂白的生理适用性警告。Fig.3/Fig.4及作者kernel不支持普遍瞬时hard cap，也不支持机械套用全段均值上限。真实负值及预历史/相位仍未解，本轮真实模型对照未运行。见 docs/chen2024_gate_correction_report_zh.md。

以下保留原审计正文作为历史档案；其中旧范围与下载状态描述不代表当前实施状态。原文件副本已存 reports/chen2024_gate_correction_before/。

---

# Cone nonlinear phototransduction 候选来源审计

审计日期：2026-10-10（UTC）。范围：只读文献和作者源码核验；未安装、运行、移植模型，未改 RetiPath 架构或训练流程。本文件是待选择的来源审计，不是实施完成声明。

## 结论与待确认选项

推荐候选 A：**Chen et al. 2024 / Angueyra–Rieke 生化模型的作者 MATLAB 非线性 forward kernel**。它能补齐“机制 ODE、完整数值参数、作者代码与可复现版本”的来源缺口。推荐只把它提名为**外周猕猴 L/M cone 外段光电流候选模块**，待用户确认后才接入。

不能由此宣称：中心至周边参数已齐、S-cone 动力学已验证、cone 电压已得到、HC→cone feedback 已闭环或全视网膜仿真已完成。**I_OS(pA)→cone V(mV)/terminal Ca/递质释放/HC feedback 的来源与接口仍 BLOCKED。**

可向用户提出的具体选择：
> 是否批准将 Chen2024 历史版本的 nonlinear forward ODE 作为外周（来源记录 >20°）L/M cone 光转导候选，先做独立单细胞作者代码一致性核验；中心/中周边/S-cone、电压及 HC-feedback 保持未解决，不借用默认参数覆盖？

原因：其机制与参数有一对一公开来源；影响：新增的是非线性外段电流来源，尚不改变整条研究主线或替代原计划实验。这里不授权实施。

## A. 推荐：Chen et al. 2024 作者 photoreceptorLinearization

### A1. 文献、生理范围、验证边界

- [论文版本记录及官方 PDF](https://cdn.elifesciences.org/articles/93795/elife-93795-v1.pdf)，DOI [10.7554/eLife.93795](https://doi.org/10.7554/eLife.93795)，2024-11-05。方程：Fig.1A / Methods “Forward model” Eqs.1–6，PDF pp.18–19；consensus：Table 1，p.4。
- Methods pp.17–18：Macaca fascicularis、M. nemestrina、M. mulatta；primate 记录均 >20°；cone 为 L/M，whole-mount whole-cell patch clamp。没有猕猴 S-cone dynamics 验证。
- 变量均值噪声用于拟合；独立刺激/细胞用于泛化；不是固定背景线性核。Discussion p.15：cone 适用限制 <50,000 R*/s，不含 pigment bleaching/regeneration；不能外推明亮户外或全部慢适应。
- 原始拟合数据由论文指向 [Dryad DOI](https://doi.org/10.5061/dryad.q2bvq83vg)。本次 DOI 页面读取失败，**未核验数据文件列表、格式、许可证或下载完整性**；不把“论文说公开”写成“数据已落地”。

### A2. 可追溯版本与真实文件

仓库：[chrischen2/photoreceptorLinearization](https://github.com/chrischen2/photoreceptorLinearization)。
论文 References 中 Chen 2024 的 Software Heritage revision 为：
**1bcf7c9a6a6814ef39d2c39d82cf8dd1b5b39db2**。
GitHub API 已核验其日期 2024-08-09，commit message “Update IsomerizationConverting.m”。

以此历史版本作论文复现基准，避免将 2026 新增 Python 包误标成论文原始实现。

| 文件 | 审计内容 | Git blob SHA |
|---|---|---|
| [src/model/BiophysModel.m](https://github.com/chrischen2/photoreceptorLinearization/blob/1bcf7c9a6a6814ef39d2c39d82cf8dd1b5b39db2/src/model/BiophysModel.m) | L33 biophysFlag=1 分支；L39–51 暗态/派生参数；L57–69 时间步进；L72 负向电流输出 | 8117c9f157fc8768993e41c521efe23aff8fc7dc |
| [src/model/initPhotoreceptorParams.m](https://github.com/chrischen2/photoreceptorLinearization/blob/1bcf7c9a6a6814ef39d2c39d82cf8dd1b5b39db2/src/model/initPhotoreceptorParams.m) | L16 dt；L19–31 peripheralPrimateCone 全参数；L69 darkCurrent | 086e324f5f5a38f04249c63b1650f68fea4d3586 |
| [src/model/demoMain.m](https://github.com/chrischen2/photoreceptorLinearization/blob/1bcf7c9a6a6814ef39d2c39d82cf8dd1b5b39db2/src/model/demoMain.m) | 5000 R*/s 示例、rate×timeStep、前置背景；只作调用约定证据，不能直接当验收实验 | 322935d64a4237153923ef26117e41a6d72f5919 |
| [LICENSE](https://github.com/chrischen2/photoreceptorLinearization/blob/1bcf7c9a6a6814ef39d2c39d82cf8dd1b5b39db2/LICENSE) | MIT，Copyright 2024 chrischen2；派生/分发保留许可和版权 | 8ca78f6d2010e1fdbc38127789f8e6bb96afb9b8 |

本次读取的 current main commit 为 **c1429e9d1850181b44a3fe9178fd4dbed8f0f7bc**（2026-03-27）。
其上述 BiophysModel.m 与 initPhotoreceptorParams.m blob SHA 与论文历史版本完全相同。
当前 [Python kernel](https://github.com/chrischen2/photoreceptorLinearization/blob/c1429e9d1850181b44a3fe9178fd4dbed8f0f7bc/photoreceptor_linearization/biophys_model.py) 自称 MATLAB port，并明确 stimulus 单位为 R*/s × time_step、返回 pA；本次只审阅，未执行 Python/MATLAB parity test。

### A3. 明确状态与连续时间方程

以下由上述 MIT 作者 kernel 整理；符号 J 表示电流正幅值，以消除作者方程与记录电流符号混用。

动态状态四个：
- R：有效 opsin activity；不是输入光子计数，也不是独立 transducin 状态
- P：有效 PDE activity
- G：外段 cGMP concentration（µM）
- C：外段 Ca concentration（µM）

代数变量：S 为 cGMP synthesis rate，J 为外段电流正幅值。记录形式输出 I_OS = −J（pA）。

```text
u(t) = total photoisomerization rate per cone, R*/s, including background
dR/dt = gamma*u(t) - sigma*R
dP/dt = R + eta - phi*P
dG/dt = S(C) - P*G
dC/dt = q*J(G) - beta*C
S(C)  = Smax / [1 + (C/kGC)^m]
J(G)  = k*G^n
I_OS  = -J(G)

I_dark_magnitude = k*Gdark^n
q    = beta*Cdark / I_dark_magnitude
Smax = (eta/phi)*Gdark*[1+(Cdark/kGC)^m]

dark initialization:
R=0; P=eta/phi; G=Gdark; C=Cdark; S=Gdark*eta/phi
```

这是有内源基线 PDE、乘性消耗 P×G、Ca→cyclase Hill feedback 的非线性动力学，不是 slanted Gaussian / fixed-tau 线性滤波。作者保留 linear 分支供其 light-adaptation-clamp 工作；RetiPath 若选择本候选，不应误调用 biophysFlag=0 或 inverse/target-linearization 例程。

R/P 是模型的有效活动量，源代码的 eta/gamma 注释、论文惯例与严格量纲推导不宜混作“已逐项实测的分子速率”。作者公式应按其归一化约定复现；不能把 gamma=10 推断为十个真实 opsin 分子。

### A4. 完整数值参数及调用陷阱

数值逐项来自历史 initPhotoreceptorParams.m 的 peripheralPrimateCone 分支：

| 名称 | 数值 | 含义/约定 |
|---|---:|---|
| sigma | 22 | opsin decay，s^-1 |
| phi | 22 | PDE decay，s^-1 |
| eta | 2000 | source effective PDE dark activation constant；保留原归一化 |
| gdark | 35 | cGMP 暗态，µM |
| k | 0.01 | cGMP→current，pA µM^-3 |
| n | 3 | current exponent |
| cdark | 1 | Ca 暗态，µM |
| beta | 9 | Ca extrusion，s^-1 |
| m | 4 | cyclase Hill exponent |
| kGC | 0.5 | cyclase Ca affinity，µM |
| gamma | 10 | source opsin gain |
| timeStep | 0.0001 s | 作者默认 cone 积分步长 |

派生暗电流正幅值 0.01×35^3 = **428.75 pA**。这是源码默认，不是所有单 cone 的实测值。参数均可获得，不需要用未知自由参数填空才能运行 forward kernel。

必须显式处理：
1. **输入单位**：论文连续 u 是 R*/s；MATLAB kernel 的 stm 已是每步 R*，R 更新的 stimulus 项未再乘 dt。因此调用时 stm[k]=u[k]×dt；不能把 R*/s 直接传进去，也不能重复乘 dt。当前 Python docstring 明确相同约定。
2. **绝对电流与差分电流**：kernel 返回 −kG^n 的绝对外段电流。光照关闭通道时向零移动；如需要 ΔI，应另行保存同背景基线并定义相减规则，不能把绝对负向电流直接当 ΔV。
3. **初始适应状态**：作者每次调用从暗态重置，demo 加前置背景让其稳定。持续仿真应在用户批准后选择并核验状态承接/预适应；不能每一帧重置暗态，也不能把暗态瞬变当刺激响应。
4. **离散顺序**：kernel R/P/C/G 使用前一步值，S[k] 由 C[k] 计算而 G[k] 使用 S[k−1]。若直接换 ODE solver，须与来源数值轨迹分开标记和验收；此审计未作求解器改变。
5. **源 demo 不是无瑕验收脚本**：历史 demo 的多个 Example 顺序覆盖 stm；noise 示例已按 MeanPhoFlux=rate×dt 生成，后又乘 dt，存在静态可见的重复缩放风险。以 kernel+明确 stimulus 合同为准，不能未核验就跑整份 demo 并声称生理复现。
6. **动力学与全细胞不同**：C 是 outer-segment Ca，不是 synaptic-terminal Ca。源码没有膜电容、漏/电压门控电流、突触释放或 HC-feedback 输入；没有 pA→mV 的可验证转换。保持 pA 模块独立；不得随手线性缩放为 mV 来冒充有生理来源的 membrane 接口。

### A5. 和早期 Rieke/ISETBio 的关系

[Angueyra et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8883858/) 属同一机制来源系列，比较了有/无慢 Ca-feedback 的生化模型；慢反馈不是唯一机制解释。Chen2024 的 forward kernel 无 C_slow，只有快速 Ca–cyclase feedback，且沿用先前的 primate consensus 选择。这说明**同源且经过后续专文验证**，不表示“完全同一方程”或“官方宣布全面废止 ISETBio”。本次未找到 supersedes 声明。

因此：
- 可以提名 Chen2024 作为更清楚的作者原始 forward 源码基准；
- 不能将 ISETBio 的 foveal 参数、k=0.02 与 C_slow 分母混进 Chen2024 的 k=0.01、无 C_slow 方程并仍声称原模型；
- 不能因后者发表更晚就声称它覆盖早期模型的所有慢适应/region 功能。

## B. 对照候选：ISETBio osBioPhys / Rieke 2014

### B1. 精确来源与公开程度

官方库 [isetbio/isetbio](https://github.com/isetbio/isetbio)，本次冻结 commit：
**d2924c2fe1d41c7dc5af1006d8081735b08f1b35**（2026-09-25）。

| 文件 | 用途 | blob SHA |
|---|---|---|
| [osBioPhys.m](https://github.com/isetbio/isetbio/blob/d2924c2fe1d41c7dc5af1006d8081735b08f1b35/cones/outersegment/%40osBioPhys/osBioPhys.m) | 两套全部参数、10°切换 | 7eee0ff4ff662c6a7a5a97c95779b6a4bcd9ae6a |
| [osAdaptTemporal.m](https://github.com/isetbio/isetbio/blob/d2924c2fe1d41c7dc5af1006d8081735b08f1b35/cones/outersegment/%40osBioPhys/osAdaptTemporal.m) | 动态方程和负向pA输出 | 208948d07069c7581f03913bc6584a5ef3a713f0 |
| [osAdaptSteadyState.m](https://github.com/isetbio/isetbio/blob/d2924c2fe1d41c7dc5af1006d8081735b08f1b35/cones/outersegment/%40osBioPhys/osAdaptSteadyState.m) | 背景初态求解 | ddc0865485c0539b72c22885d87d20889f95e5b6 |
| [LICENSE](https://github.com/isetbio/isetbio/blob/d2924c2fe1d41c7dc5af1006d8081735b08f1b35/LICENSE) | MIT，保留版权/许可 | 7b9cd0d5821a2199ce25aadfac3bdec16e953bea |

[2014-11-02 Rieke model note](https://isetbio.github.io/isetbio_v0.1/cones/adaptation%20model%20-%20rieke.pdf) Eqs.1–10 给出机制。其 primate 定位不能替代每套参数的 specimen/species/background/region 原始拟合元数据。引用的 Cone-Adaptation wiki 本次返回创建页面界面，未取得有效内容。

### B2. 状态、接口和参数

五个动态状态 R/opsin、P/PDE、G/cGMP、C/Ca、Cs/Ca_slow；S 为代数变量。
与 A 的主要差别：
```text
dCs/dt = betaSlow*(C-Cs)
J = k*G^h / (1+Cs/Cdark)
I_OS = -J
q = 2*beta*Cdark/(k*Gdark^h)
Smax = (eta/phi)*Gdark*[1+(Cdark/kGc)^n]
```
其余结构 dR=OpsinGain*u−sigma*R、dP=R+eta−phi*P、dG=S−PG、dC=qJ−beta*C。注意 ISETBio 的 h 是 current exponent、n 是 cyclase exponent，和 A 的命名不同。

代码直接接收 pRate=R*/s；它自己在更新式整体乘 dt。输出 outer-segment pA。osCompute 用 absorptions/integrationTime 转 rate；cMosaic 专用入口可按 cone 接受标量背景、保存状态，并额外返回**扣背景的 differential** current，与 raw temporal 输出有别。

全部参数按 osBioPhys.m：

| 参数 | peripheral (>10°代码分支) | foveal (≤10°代码分支) |
|---|---:|---:|
| sigma | 22 | 10 |
| phi | 22 | 22 |
| eta | 2000 | 700 |
| gdark | 20.5 | 20.5 |
| k | 0.02 | 0.02 |
| h | 3 | 3 |
| cdark | 1 | 1 |
| beta | 9 | 5 |
| betaSlow | 0.4 | 0.4 |
| n | 4 | 4 |
| kGc | 0.5 | 0.5 |
| OpsinGain | 10 | 12 |

### B3. 为什么不优先选它填本次严格来源缺口

- 官方工程实现、机制和全部数值均公开；但参数分支的“foveal/peripheral”不能自动上升为猕猴中心至周边的生理事实。源码 L33–36 自称切换阈值 somewhat arbitrary；L57–62 明示只有两档和未来 intermediate 参数计划。
- 核验的类中没有 L/M/S 专属动力学参数，亦没有各档 species/recording/background 元数据；能对L/M/S数组计算不等于验证过S-cone kinetics。
- **静态一致性风险**：osAdaptTemporal.m L85 用 OpsinGain×pRate；osAdaptSteadyState.m L95 却用 bgR/sigma，背景求解式也未见 OpsinGain。给同一个 R*/s 背景可能不会从真正时间方程稳态开始。未运行，不能把这个静态疑点宣判为实际测试失败，但采用前须核验。
- temporal 更新按新 R/P/C 顺序推进，且末尾有采样裁剪/复制；不能假定与 A 的显式前一步序列逐点相同。
- 一样没有膜电压/terminal Ca/释放/HC-feedback。多出的 Cs 是外段模型慢反馈，不是 HC feedback。

## 下一阶段仅在批准后的验收建议

范围限定于 A 的外周 L/M 外段模块，不替代用户原计划：
1. 锁定上述论文 commit、MIT notice 和参数来源；保留 rate、dt、pA、背景、region/cone_type metadata。
2. 对照作者 kernel 验证黑暗、背景预适应、弱闪、阶跃+闪光、增/减光非对称；输入和验收阈值预先列明。先确保单位/符号/状态/步长正确，再谈生理拟合。
3. 获取并核验真实 source traces 后才宣称重现论文；先前此步仍未完成。
4. separate status: nonlinear source identified；implementation pending approval；numeric parity unrun；regional extension BLOCKED；cone V/HC feedback BLOCKED。
5. 不把“可运行单细胞外段模块”写成“中心至周边RetiPath完成”；任何新增regional假设或跨单位耦合继续向用户说明后确认。

## 本地来源留存

旁边的 cone_dynamics_reference_sources/ 只保存本次审阅的原始参考文件与许可，不属于 RetiPath 实现，不曾执行。manifest.json 记录仓库、commit、原路径、Git blob SHA、来源链接。8 个留存文件已逐字节计算 Git blob SHA，与连接器返回 SHA 全部相符；不存在模型经过数值测试或经本地重写的暗示。
