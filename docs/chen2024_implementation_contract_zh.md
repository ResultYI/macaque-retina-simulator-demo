# 阶段 C：Chen2024 外周 L/M cone 外段电流合同

用户已批准（Sentinel_db5d1bccdbbc8191bed02a864c13f5f4）。主线完整目标及 A/B 保留，本合同只新增独立非线性外段模块。来源为历史 commit `1bcf7c9a6a6814ef39d2c39d82cf8dd1b5b39db2` 的 MIT `BiophysModel.m` 与 `initPhotoreceptorParams.m`。禁止调用 linear/inverse 分支或 demo；已逐行审查两个运行函数只做参数与模型计算，无文件／网络／系统调用。

范围来自已提供文献审计：Macaca fascicularis/nemestrina/mulatta 的混合来源，记录全部 >20°、L/M cone。它不是单一物种默认配置，不涵盖中心／中周边／S cone，不含 bleaching/regeneration 或全部慢适应。输出为绝对负 pA，不连接 HC/BC，不用任意 gain 换成 mV。

## 50k 门禁勘误与已批准的窄修正

2026-10-10，用户批准 Sentinel_68036fb39d9481919461d1837cc67cdf。修改前已重新核对原作者 kernel、Fig.3 p7（80k 刺激轴）、Fig.4 p8（primate cone 100k 刺激轴）与 Discussion p15。正文 <50,000 R*/s 讨论模型不含漂白/再生且限制曝光的适用性，不是逐样本 hard cap，也未给可机械替代的全段均值上限。先前将它实现为所有 reference 输入的瞬时拒绝过强；≥50k 点数不是来源数据不可复现的单独证据。

最小显式区分：仅 RunMode.REFERENCE / component_reference 且构造器显式 `source_reproduction=True` 选择 `author_source_reproduction` 输入策略时，正值高峰不再因 ≥50k 拒绝。默认 `strict_project_reference` 与 numeric_fixture 保留 0≤u<50k 工程封套；full_physiology 仍禁止此组件。来源账本与所有参数字节不变，账本 total_rate 上界只按保守默认工程封套解释，不表示作者源码数值 domain。

作者复现 stream 的 rate_segment 必须显式选择相同 input_policy；跨策略 segment/state/checkpoint 拒绝。负值、非有限、单位、形状、时间网格、物种/区域、数值状态和下游门禁不变。高峰产生 HighLightApplicabilityWarning，并在结果中保存实际消费区间的范围/峰值、描述性均值、持续时间、累计 R*、高峰点数、来源定位和不含漂白的适用性警告；不会以均值建立新接受条件。曝光前史和物理漂白状态不从输入推断。批量最后未消费样本另记录其提供范围，不混入累计曝光。

R/P/G/C/S 更新、原参数、暗态初始化、dt、连接和数据不变。本轮不运行真实记录、不拟合、不处理负值；真实数据仍由负值含义及预历史/相位依据未解而阻塞。整体代码/策略身份 model_hash 随接口修正改变，旧 checkpoint 不静默迁移；动力学、参数、state-spec 和 graph 指纹另核相同。

## 实际离散源码合同

源码 L39 用 `gdark=(darkCurrent/k)^(1/n)` 重算暗态；保留该顺序。L41–42：`q=beta*cdark/darkCurrent`；`Smax=eta/phi*gdark*(1+(cdark/kGC)^m)`。初态 R=0、P=eta/phi、G=重算 gdark、C=cdark、S=gdark*eta/phi。

固定 dt=0.0001 s，float64。公开输入 u 为每 cone 的非负有限 total photoisomerization rate，R*/s，包含背景；上界是否拒绝按上述显式策略区分。传给原 MATLAB kernel 的 stm=u*dt，只乘一次；其时间向量 N 点只执行 N−1 次更新，最后一项 stm 不消费。批量 source-parity 包装返回 N 个样本（暗态＋N−1 次更新）；streaming 则把每个 u 样本解释为一个 dt 区间，消费 N 个样本并返回 N+1 状态，批量对照显式使用前 N−1 项。不能静默错一格或改 ODE 求解器。

对旧状态 R/P/G/C/S 和区间输入 u，源码逐步更新：

```text
Rnew=R+dt*(-sigma*R)+gamma*(u*dt)
Pnew=P+dt*(R+eta-phi*P)
Cnew=C+dt*(q*k*G^n-beta*C)
Snew=Smax/(1+(Cnew/kGC)^m)
Gnew=G+dt*(S-P*G)
I_OS=-k*G^n
```

R/P/C/G 使用旧步值；Snew 依赖 Cnew，但 Gnew 必须使用 Sold。四个主要动态状态 R/P/G/C；S 是源式代数缓存，仍持久保存以保留初态与离散更新的逐点顺序。统一 SimulationState 中显式保存其五块数组、绝对时间、source／parameter／model／state／graph 身份及固定步长合同。持续流不每帧暗态 reset；checkpoint 保存全部缓存并拒绝不兼容恢复。

## 单位与证据

G/C 为 µM，S 为 µM/s，I_OS 为 pA。R/P 为有效活动量，按方程归一化分别使用形式单位 s^-2/s^-1；eta 形式单位 s^-2、gamma 形式单位 s^-2/R* 是量纲推导，不是逐项分子实测。账本同时保留源码 eta 的 1/sec 注释和 gamma 原注释，不能声称其注释已构成严格实测分子单位。数值逐项解析 peripheralPrimateCone 分支，类别 author_assumption，源式派生量类别另注明 derivation；不编参数、不跨物种平均。

显式声明 absolute current 与基线差分：差分仅在提供同背景基线电流 Quantity 后计算 `I-baseline_I`；禁止当成电压。原始作者每次调用暗态初始化；稳定背景单测通过持续背景推进达到预先规定稳态判据，不能任意直接设置适应态或每帧预热。

## 数值验收

固定刺激均 synthetic_fixture：暗态、5000 R*/s 背景、弱闪、阶跃＋闪光、增／减光及近支持域上界。原 MATLAB 历史 kernel 实际执行，仅 biophysFlag=1，从原 initPhotoreceptorParams 取得参数，导出输入与原 response。独立源式 scalar oracle 只用于查看 R/P/G/C/S 序列；原 MATLAB 电流是主要 source-parity 对照。

对 float64 MATLAB 电流预定 abs<=1e-8 pA、relative<=1e-10；纯 streaming chunk/checkpoint 与单段在同运算顺序下要求逐点相同；暗态允许 1e-9 pA 浮点残差；稳定背景末段每步状态差 scaled max<1e-8。容差仅是工程设置，若失败保留原失败并修复，不靠持续放宽过关。必须测试错误单位、shape、负／非有限输入、中心／S cone／错误物种、错误 dt/solver/hash、linear/inverse/电压耦合拒绝。新增策略隔离回归：默认/numeric >=50k 拒绝，显式作者复现正值高峰可计算且有警告，真实负值两策略均拒绝；修改前后域内全部状态和电流逐点一致。

## 真实数据边界与当前阻塞

Dryad v3/325663，MAT file3585689，41,490,462 bytes，发布 SHA256 `6724e2edd8e8708456f32900a14efae88192a5ba597365046d0031eb543d1331`。历史普通下载返回403/API401、Chrome CUA 启动失败的档案保留。用户随后提供真实 ZIP，MAT/README 已安全提取、字节数及官方 SHA256 匹配、全部字段已清点，见真实数据报告。

实际 PrimateConeResponses 为6条、每条80000点的刺激和暗电流归一化平均响应，README 标4–5次平均、R*/s、dt0.1ms；逐cell物种／精确偏心度／绝对暗电流／holdout 不在包内。刺激负值1026点的处理含义及实际前史/相位仍未解，不补字段或裁剪。只能在依据充分后做固定参数来源数据复现，不能叫独立生理验证。

完成后保留 A/B 报告并另写 C 报告，跑 A/B/C 回归并做一次独立限定审查。完整需求 A02 只新增外周 L/M 子项，完整模式仍阻断；无训练、下游回路或新的区域假设。
