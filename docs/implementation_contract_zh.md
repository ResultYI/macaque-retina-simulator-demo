# 猕猴连续中心—周边模拟器：阶段 A/B 实施合同

版本：2026-10-10。用途：交 Codex 实施。本文为已核来源之独立修正附录；不得把 Pro 原稿中的占位方程当作生理实现依据。

## 0. 不变的完整目标与本次交付

完整目标仍为 macaque 中心到周边连续的、含光输入、非均匀 L/M/S cone mosaic、光转导/适应、HC feedback、ON/OFF BC、AC、RGC、眼动及解剖—功能位移的多层模拟器。阶段 A/B 是该目标内的可验证部件，不删减目标，不替代既定研究主线、架构决定或训练实验。

本次只实施有证据可闭合的代码和 fail-closed 基础设施。禁止以任意常数、恒等映射、零输出、匿名 F/G、可运行的玩具链或仅线性参考结果，冒充完整生理模拟完成。不得开始训练、科学对照实验或发布生物学图。以下数值单测不构成生理实验。

已确认的物种政策：用户同意共同框架、按物种与来源分开参数，不做无依据平均；人/猫参数不进入 macaque 主模型默认。确认消息：Sentinel_601db2e51e7c81919e962a4cc23a6f39（用户回复“可以”，确认上下文由父线程记录）。这条政策已定，不再作为等待用户批准的事项。作者混合来源模型仅作为具名、隔离的 source-specific reference 保存，不能继承到主模型默认。

Codex 应保留现有仓库组织和已授权架构。下列 Python 风格类型/签名用于明确语义；无需为符合文件名而重构已有系统。已有代码若不兼容，应报告真实差异，不能偷偷改变主线。

## 1. 两阶段范围与完成定义

### 阶段 A：可依赖的实施底座

1. 版本化来源注册表、参数账本、来源资产校验和禁止未知参数进入生理运行的加载器。
2. 精确导入 ISETMacaque 已取得表值，保留原精度、子午线、单位、证据等级与支持域；不外推、不补 e=0、不生成不存在的二维实测图。
3. 有单位和坐标语义的逐细胞、稀疏有向连接、层状态、刺激端口及运行结果结构。
4. 联合 ODE 积分器与可持续的 state/chunk/checkpoint 接口；先用于 synthetic_fixture 数值单测。
5. 单位正确的光谱/面积积分算子；缺少物种/光学 profile 时只能做 synthetic_fixture 解析测试，不做 RGB 标定、不设置默认光子量。
6. 完整目标需求覆盖表、启动前阻断报告、自动测试及仓库要求的聚合检查。

阶段 A 完成只表述“数据、接口、运行门禁与工程测试完成”，不表述“生理模型已能运行”。

### 阶段 B：具名作者参考组件

实现隔离命名的 `SomaratnaFreeman2025GaussianHCReference`，移植作者固定参数、Gaussian-HC 分支的联合 RHS、空间权重规则、率输出与显式源配置。它不得成为完整模式的默认实现。

两个作者密度 .mat 目前已下载并验 hash，但其 MATLAB MCOS table 尚未读出带列名的数字。故本次阶段 B 使用明确标作 `synthetic_fixture` 的位置/连接做数值单测，不宣称完整作者复现，不生成生物学图，不用 ISET 表替换后仍称作者原模型。

阶段 B 完成只表述“已实现来源可核的线性参考内核，数值测试通过；作者原阵列/论文结果复现未完成”。若能在已有受支持环境不运行模型地精确导出 .mat，可另提交来源数据导出证据；不能静默扩大完成声明。

## 2. 来源证据和资产身份

本合同方程核验来自作者代码静态检查；不是逐符号读取了出版社排印方程的声明。出版社正文直取返回 403；原论文 DOI 与作者来源已独立确认。

Somaratna & Freeman 2025，DOI：https://doi.org/10.1152/jn.00302.2024

作者代码固定 commit：`5406d1533dae1678f94687379ac078a612f6170f`，仓库 https://github.com/AlanFreeman/Colour 。

主要文件及函数：
- `runColFun.m`：calStim/calDrive、calConv/weight、calDeriv、solveT/solveF、cone/gang、calStruct。
- `setColLit.m`：作者模型数值参数与其引用标签。
- `runCol.m`：setDef 及 resp.t；horExp=0 的默认/时间响应设置。
- `runColTask.m`：resp 中的 rate=7.2*max(0,g)。
- `setColFun.m`：calRog/rogPar，仅作单独解析参考，不能与 Gaussian-HC 分支无标记混用。
- License.txt 指定 CC BY 4.0；保留作者、固定版本、许可、改动说明。

完整固定链接以 `https://github.com/AlanFreeman/Colour/blob/5406d1533dae1678f94687379ac078a612f6170f/` 加原文件名。不要引用仓库浮动 main 代替锁定版本。

已校验作者原二进制：
- Cone density.mat，5011 bytes，Git blob SHA-1 `fe56fe90e041d5699538dd2d78e2afa5757c708d`。
- Ganglion cell density.mat，3054 bytes，Git blob SHA-1 `10eb6eca68976793a4db7164fdb6e22850e5330f`。
- 两文件都含 opaque MCOS table `d`。目前只核验来源/身份和格式，未核数值或支持域。

可用交接资产位于共享来源包；消费执行器必须按其支持的文件交接流程取得、核 hash 与实际可读性，不能假定共享云路径存在于用户电脑。

ISETMacaque 来源资产：`ISETMacaque_PackerConeRc_cf22ac9.m` 和 `isetmacaque_packer_tabulation.csv`，沿用已有审计的固定完整 commit/来源 URL，不能把文件名里的短 hash 自行扩写。把它登记为 published-code tabulation，不写成已取得原始实验逐细胞数据。

Macular 官方单位/接口文档：https://macular.gitlabpages.inria.fr/Macular/user_doc/Macular/main.html 。它是可定义细胞方程的框架，不是默认含全部 macaque 参数的数据库。

## 3. 数据字段与严格加载合同

### 3.1 来源记录 SourceRecord

必备字段：
- source_id、title、authors、publication_year、doi_or_canonical_url
- artifact_kind：paper/code/data_table/supplement
- repository_commit（代码必须）、source_path、source_locator（函数/行/表/figure）
- asset_sha256、source_blob_sha1（适用时）、license
- verification_level：identity_verified/code_verified/numeric_table_verified/primary_measurement_verified
- provenance_notes、retrieved_at、upstream_sources_checked（明确哪些上游引用尚未检查）

不得把 `code_verified` 自动提升成 `primary_measurement_verified`。

### 3.2 参数记录 ParameterRecord

必备字段：
- parameter_id、name、value 或 null、unit、shape
- source_id、source_locator、derivation（无转换则 identity）
- evidence_kind：measured/source_fit/author_assumption/project_assumption/synthetic_fixture/unknown
- species、sample_context、retinal_region、meridian、coordinate_frame
- support_domain（每轴名称、单位、闭开边界；未知为 null）
- adaptation/background/stimulus condition、uncertainty（未知显式标记）
- allowed_modes、status：usable_reference/usable_physiology/blocked
- assumption_approval_reference（仅本项目新增且已批准假设适用；不能伪造）

签名：
- `register_source(record: SourceRecord) -> None`
- `load_parameters(profile_id: str, mode: RunMode) -> ParameterSet`
- `validate_parameters(parameters, requirements) -> CoverageReport`

加载器必须检查 source 存在、资产身份、值/单位/shape、适用条件、坐标语义、支持域。完整模式拒绝 unknown/null、缺来源、synthetic_fixture、未批准 project_assumption 及条件不匹配。报错必须包含 parameter_id、缺什么、受阻模块及可补证据；禁止落到 default。

### 3.3 逐细胞与连接

`CellTable`：cell_id、cell_type/subtype、species_profile_id、hemiretina/meridian、soma_position、rf_position、position_unit、coordinate_frame、eccentricity_value/unit/frame、parameter_profile_id、evidence_tags。

soma_position 和 rf_position 是分开的可空字段；无位移模型时不得以同一坐标填充后声称已考虑位移。局部 patch 坐标与全局 eccentricity 必须分开。

`ConnectionTable`：edge_id、pre_cell_id、post_cell_id、synapse_type、weight、weight_unit、sign_convention、delay_seconds（来源不存在时不能设“生理默认”）、source_profile_id、normalization_rule、evidence_tags。

存储可使用 CSR/COO；明确方向为 rows=postsynaptic、columns=presynaptic。不允许为了稀疏而悄悄截断作者全连接 Gaussian 权重。full kernel→sparse approximation 是另一个需公开误差/批准的模型或数值近似。

`StateSpec`：state_name、cell_ids、shape、unit、meaning（absolute_voltage/generator_deviation/generator_with_output_offset/activity 等）、initial_condition_source。

`SimulationState`：time_seconds、ordered_state_arrays、state_spec_hash、graph_hash、parameter_hash、model_id/version、input_interpolation、solver_config、optional_rng_state。

`SimulationResult`：run_mode、model_id、source_manifest_hash、input_contract、outputs_with_units、coverage_report、numeric_checks、claim_level。synthetic 结果须逐对象继承标签，不能只藏于顶层。

### 3.4 ISET 表及接口

保留 CSV 60 行：temporal 28、nasal 32；原 mm 与 cones/mm²。

支持域：temporal=[0.003839230868131,15.72093224557] mm；nasal=[0.003839230868131,19.9277403918] mm。e=0、superior/inferior 均无这份表的证据。

签名：
- `read_density_tabulation(asset, expected_hash) -> MeridianDensityTable`
- `interpolate_meridian_density(table, eccentricity_mm, meridian, method) -> DensityQueryResult`
- `evaluate_packer_rc_reference(table, eccentricity_deg, meridian, magnification_profile) -> RadiusQueryResult`

`method` 必须显式登记为 computational_interpolation，返回 bracketing source rows 与 interpolation provenance。未经配置的插值方式不能偷设为“生理连续函数”；本阶段可明确实现线性插值并命名，不声称唯一生理真值。

PackerConeRc 原参考顺序是 density→spacing→Rc，再线性 interp1(Rc)。不能由插值 density 代替。其 210 µm/degree 与 Somaratna 4.73 degree/mm 分别保留来源；密度转换为 per degree² 时除以 magnification_deg_per_mm²。

## 4. 光输入单位算子：有用但不伪造标定

严格区分 `SpectralRadiance[W/(m² sr nm)]`、`RetinalSpectralIrradiance[W/(m² nm)]`、`ConeContrast[dimensionless]`、`PhotoisomerizationRate[R*/s]`。没有 optics/pupil/geometry profile 时，radiance→retinal irradiance 未闭合，禁止强制转换。

纯单位算子签名：
- `integrate_cone_photoisomerization_rate(irradiance, wavelength_nm, quadrature_weight_nm, aperture, quantum_efficiency_profile) -> PhotoisomerizationRate`

两种 aperture 约定必须显式选一，不能重复乘面积：
1. 空间权重 w 面积积分=1（w 单位 m⁻²），先得 Ebar_λ[W/m²/nm]，再乘 A_eff[m²] 得 P_λ[W/nm]。
2. 空间 aperture a 无量纲、面积积分=A_eff，直接对 E_λ*a*dA 得 P_λ。

输出 `q=sum_λ P_λ * η_λ * λ_m/(h*c) * quadrature_weight_nm`，单位 R*/s；λ_m=λ_nm*1e-9，h、c 用 SI 常量。η 是指定光捕获/异构化链的显式无量纲效率；必须明确它包含哪些介质、吸收与异构化因素，防止重复乘，且本阶段没有任何默认 macaque η 或 A_eff。

检查：波长/积分权重与谱 shape 对齐、坐标面积单位转换、有限非负输入、A_eff 正值、权重归一化与效率定义域。缺 profile 报 `MissingCalibrationError`，不能从 RGB 范围猜 R*/s。synthetic_fixture 可用常值谱和面积检验解析积分，测试记录不得成为生理 default。

## 5. 联合积分与持续状态合同

通用签名：
- `rhs(t_seconds, state_vector, stimulus, parameters, graph) -> StateDerivativeVector`
- `initialize_state(model_id, parameters, graph, initial_conditions) -> SimulationState`
- `advance(state, stimulus_segment, t_stop_seconds, solver_config) -> (SimulationState, OutputSegment)`
- `checkpoint(state) -> Checkpoint`
- `restore(checkpoint, expected_model_hashes) -> SimulationState`
- `validate_runnable(request) -> CoverageReport`

先完整验证再运行。所有有反馈层在同一 RHS 使用同一时刻的联合状态，不能按 cone_new/HC_old/BC_new 加一帧传递延迟；反馈环不是必须加 delay 或必须 implicit 的理由。无已来源化 delay 的模型不得无声加 delay。

数值选项（solver、rtol、atol、dtype、max_step、stimulus interpolation）与生理参数分开保存。输出采样 dt 不是自适应积分内部步长。chunk 不重置状态、绝对时间或周期相位；修改 graph/parameter/model hash 时拒绝不兼容恢复。未授权 stochastic 机制不要创建随机 spike state。

## 6. 阶段 B：Somaratna Gaussian-HC 精确内核

### 6.1 明确的输入/状态/输出

输入 u_i(t) 为经过该作者光学 attenuation 的 cone contrast，无量纲。不是 photon rate，不是绝对光强；不带背景依赖 adaptation。

状态 c,h,b_off,b_on 为 mV generator deviations；g_off/g_on 为含正输出偏置的 generator potentials，并非绝对膜电位。

参数：τ=0.014 s；kSens=17.8 mV/contrast-unit；kSur=1.54；kRect=7.2 Hz/mV；resting rates=[21.8,25.4] Hz；gRest=resting_rates/kRect mV。全部逐项指向 setColLit/gang，不能标为当前实验测得。

### 6.2 联合 RHS

- dc/dt=(-kSens*u-kSur*W_hc*h-c)/τ
- dh/dt=(W_ch*c-h)/τ
- db_off/dt=(c-b_off)/τ
- db_on/dt=(-c-b_on)/τ
- dg_off/dt=(W_off*b_off+gRest_off-g_off)/τ
- dg_on/dt=(W_on*b_on+gRest_on-g_on)/τ

正亮输入首先令 c 负向变化；ON bipolar 反号，OFF 同号。feedback_output=kSur*W_hc*h，仅为代数输出，不是额外适应状态。

初态 c=h=b_off=b_on=0、g=gRest。输出 rate=kRect*max(0,g)，Hz。作者不提供 spike-time、Poisson/Bernoulli/refractory 模型；不得为“输出看起来完整”而加入。

### 6.3 权重规则

`a(post,pre)=exp(-distance²/r²)`；先排除输入类型，后按 postsynaptic 行归一化。零有效输入行报错，不回退均匀。

- W_ch：rHor，排除 S 输入。
- W_hc：由未归一化 a 的转置重新归一化，不排除 S；不是 W_ch 的简单转置。
- W_off：rGang，不排除 S。
- W_on：rGang，排除 S。

签名：`build_somaratna_weights(cone_locations_deg, cone_types, ganglion_locations_deg, eccentricity_functional_deg, profile, evidence_tag) -> ReferenceGraph`。

作者局部 patch 的 e 单位 degree：rOpt=polyval([1.4241e-5,0,0.011663],e)，rHor=exp(polyval([-4.5667e-5,0.0030709,0.029977,-3.261],e))，rGang=polyval([-1.3583e-6,0.00014598,0.0012385,0.012601],e)。系数降幂排列。阶段 B 只为具名 reference/source-config 或 synthetic fixture 求值；无已核支持域则不可给通用 physiology 许可。不得把所有细胞换成不同 e_i 后仍称作者原模型。

### 6.4 数字一步更新与独立 oracle

作者用 ode45 联合积分；可用工程实现中已选择的联合求解器，但必须有明确精度和独立测试。

对于分段常量 u，令 z=g-gRest、x=[c,h,b_off,b_on,z_off,z_on]，上述 RHS 等价 dx/dt=A*x+B*u；A 的块行依次为：
- [-I,-kSur*W_hc,0,0,0,0]/τ
- [W_ch,-I,0,0,0,0]/τ
- [I,0,-I,0,0,0]/τ
- [-I,0,0,-I,0,0]/τ
- [0,0,W_off,0,-I,0]/τ
- [0,0,0,W_on,0,-I]/τ

B 只有 c 块为 -kSens*I/τ。由 block exponential `exp(dt*[A B;0 0])` 得 Ad、Bd，`x_next=Ad*x+Bd*u`。这是从原模型导出的 ZOH 数值 oracle，不是新增生理机制；仅适用于该输入合同。可用它独立检验联合积分器，不要求大型生理图采用稠密 expm。

对于连续正弦，频域 oracle 令 C=1/(1+iτω)：`c_hat=(I+kSur*C²*W_hc*W_ch)^(-1)*(-kSens*C*u_hat)`；b_off_hat=C*c_hat、b_on_hat=-C*c_hat、g_hat=C*W*b_hat。该 oracle 与同一 Gaussian-HC 分支匹配。

`horExp=true` 的作者 solveF 改用 W_coneBack，而 solveT 不跟随此分支。阶段 B 必须拒绝 true，不能默默替换。ROG rSur 的 exponential-HC 版本不得拿来校验本 Gaussian-HC 空间网络。

### 6.5 生理解释限制必须随组件输出

- 作者目标是 macaque，但部分参数取人或猫的研究/缩放；不能标成纯单一 macaque species 实测模型。
- 作者注释的 ganglion jitter 取 Dacey 1993（人）；optics 由人 PSF 缩放；retinal magnification 汇合 M. mulatta 与 M. fascicularis；kRect 的上游采用猫 V1 研究。上游哪些已独立检查、哪些仅由作者来源标签支持，必须分开。
- 此组件是固定 τ 线性状态加输出整流，没有背景亮度依赖、calcium/bleaching/Ih adaptation 状态。
- 只覆盖 H1 与 midget ON/OFF 路径，没有 AC、其他 RGC、完整 cone-type 规则、解剖位移或连续二维全视野。
- 当前 synthetic_fixture 只证明实现数学一致；不代表猕猴神经连接或观测响应。

## 7. 完整主线需求覆盖和硬阻断

维护机器可读 requirements 文件，并生成人可读覆盖摘要；每项 fields 为 requirement_id、required_module、status、implemented_artifact、source_ids、missing_evidence、tests、claim_limit。

必须保留以下条目，不能删掉使覆盖率升高：
- 物种/来源隔离策略：POLICY_CONFIRMED，共同框架、分物种/分来源参数，不无依据平均，人/猫不进入主默认；具体物种 profile 未补齐的参数仍分别 BLOCKED，不以策略已确认代替数据齐备。
- 校准 RGB/光场→retinal irradiance→cone R*/s：算子可测，实际 optics/pupil/spectrum/profile 未闭合为 BLOCKED。
- 连续二维、中心含 e=0 的 cone density/mosaic 与完整 L/M/S、S-cone 区域规则：部分 meridian 表可用，完整项 BLOCKED。
- cone aperture 的生理空间模型：单位接口存在不等于参数存在，缺值 BLOCKED。
- 光转导与非线性光/对比适应：BLOCKED。Saha 区域响应统计只能作约束/未来校验，不能直接写进不明 ODE。
- HC feedback：具名线性参考 AVAILABLE_REFERENCE；主线生理模型未冻结仍 BLOCKED。
- BC subtype/非线性/连接：midget 线性参考不解除全主线 BLOCKED。
- AC 类型/状态/靶细胞/连接：BLOCKED。
- RGC 类型覆盖和所需 spike mechanism：midget rate 参考不解除全主线 BLOCKED。
- anatomical↔functional displacement 和细胞密度 Jacobian/计数守恒：BLOCKED；禁止只换坐标标签。
- gaze/eye movement 到 retinal image 的校准映射：接口可留，完整实际路径缺证据时 BLOCKED。
- 中心—周边连续动态 profile：不能由两个区域端点自动变成生理函数，BLOCKED。
- 作者原 .mat 图结构与论文数值复现：BLOCKED_DATA_DECODE。
- 完整功能/生理验证与训练实验：本阶段 NOT_RUN，不能据单测通过宣称完成。

`validate_runnable(mode=full_physiology)` 必须一次返回全部阻断并拒绝生成完整结果；不能通过选择 reference 模式悄悄满足完整请求。显式 component/numeric_fixture 模式可运行，只返回其有限声明。

## 8. 测试验收（必须实际运行并报告，不是声明已经通过）

### A 组
- 来源账本：缺 source、坏 hash、缺单位、unknown、错误 species/condition、未批准假设、synthetic 混入 physiology 都失败。
- ISET：60 行与分段计数/原数值一致；域内节点恰好返回原数值；e=0、负值、超域、未知 meridian 拒绝；密度单位平方缩放正确；Rc 参考变换顺序与原码一致。
- 类型：contrast、R*/s、mV、mV/s、绝对/偏差电压互传失败；anatomical/functional 互传失败。
- 图：方向/shape/ID 引用正确；空归一化行报错；排除 S 先于归一化；不隐式创建或截断边。
- 光积分：常谱解析结果、面积线性缩放、nm→m 因子、两种 aperture 约定等价、波长积分权重、效率未设置拒绝、RGB 直转拒绝。
- 状态：chunk 与单段一致；checkpoint恢复一致；参数/图 hash 不匹配拒绝；绝对时间连续；无隐藏 reset。
- 全模式：每个完整需求仍存在；任何必需缺口都硬失败且返回完整原因。

### B 组（所有图与刺激均 synthetic_fixture）
- 零输入保持非放电状态 0，g=gRest，rate=[21.8,25.4] Hz。
- 正输入初始 cone derivative<0；ON bipolar 反号、OFF 同号。
- 行和为 1 的均匀恒定输入稳态：c=h=-kSens*u/(1+kSur)，b_off=c，b_on=-c，g_off=gRest_off+c，g_on=gRest_on-c。
- 同一联合 RHS 的积分轨迹与 ZOH block-exponential oracle 收敛一致；记录 rtol、atol、采样方式和最大误差。
- 正弦稳态幅相与本节 Gaussian-HC 频域公式一致；明确忽略初始瞬态所用的数值收敛判据，而不是把不收敛数据调参拟合过去。
- 输出整流后非负、单位 Hz；没有自动 Poisson/Bernoulli sampler。
- horExp=true 明确报 UnsupportedReferenceBranch；未知 adaptation option 报错而非被忽略。
- 连续流状态在 chunk 边界不跳变；重启只在显式 reset。
- 参考模型日志/结果始终带 source commit、cross-species 限制、synthetic 标签和 claim_limit。

测试数值容差属于计算设置，不是生理常数；Codex 需在测试文件固定合理 float64 容差并说明理由，不能为通过测试持续放宽而不报告。还须按仓库说明运行受影响的 lint/typecheck/aggregate test；分别报告通过、失败、未运行。

## 9. Codex 回传内容与停止点

交付：改动文件清单；来源 manifest 与数据 hash；A/B 各测试命令/退出码；完整覆盖表；reference 输出的限制；未完成清单；下一项具体所需证据。

完成本阶段后停止于 A/B 工程交付。不要自行补生理假设、换研究问题、扩大到训练或绘制生物学结论图。依已确认政策直接实施共同框架和分物种/分来源参数，无需重问。若将来提出违背该隔离政策的混物种默认、用 midget 线性模型替代完整目标、改变已确认主线、决定未授权架构或训练流程，先陈述变更/原因/影响并由用户确认。

本合同允许把可证实的部分真正实现完，而不允许用不断审计替代工程交付，也不允许用工程可运行性替代生理完整性。
