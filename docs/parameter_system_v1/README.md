# 全层参数体系 v1：使用与扩展

本版是静态参数定义、解析和绑定迁移，不是新生理模型。原 Population → ComponentSpec →
ModelSpec → Builder → Runner、Python/SciPy、主配置和 effective_dynamics_v1.json 保留。
没有拟合、偏心度曲线生成、数值后端或默认实验读出切换。

## 实际入口

- 普通组件：RetinaBuilder.compile 继续使用原 factory/来源检查，再生成 resolved_parameters，
  parameter_system_binding_id 和实际组件参数快照。原 native 模型、初始规则和 CSR 不改。
- 群模型构造：build_group_construction_seed 在原来源/recipe/派生规则完成后，
  调用新解析器，并将 compatible_inner_parameters 的结果作为真正返回的 inner_parameters。
  因此不是仅报告旁路；旧初态来自原构造，解析结果必须与当前快照精确一致。
- 已有局部快照：RetinaBuilder.bind_group_snapshot(assembly, seed, root, protocol=protocol)。
- 执行：ParameterSystemGroupRunner(assembly, seed, protocol, bundle, mode="reference")。
  它把新解析兼容视图交给原 GroupTimeArm，并在协议和 checkpoint controller 中登记独立绑定身份。
- Cell6：现有 ConditionalTemporalOutput 复用原 TemporalReadout，仅增加参数体系作用域校验和元数据。
  run_retina_gui.cmd 默认仍是原生产/既有读出，实验模式必须显式选择。

当前快照解析命令：

    D:\macaque_retina_simulator_envs\cu126_py311_v1\Scripts\python.exe -B -X utf8 tools/resolve_parameter_system_v1.py --output reports/my_new_parameter_bundle --validate-binding

输出目录必须全新，拒绝覆盖旧结果。该命令加载当前 native0.6/inner0.3 局部资产，
构造原/新 Runner 的初始快照和恢复对象，不推进时间、不做新背景求解或拟合。

交付示例位于 reports/parameter_system_v1_20261028/delivered_snapshot：

- registry.json：完整版本化 ParameterRecord，可用 IntrinsicParameterRegistry.read_json 读入。
- resolved_fp64.npz / manifest.json：局部 FP64 数组、按层/类型 theta schema、来源引用、布局和身份。
- initial_state/：独立 StateBundle 初始状态、原历史/RNG/HC延迟；不是完整运行 checkpoint。
- v1_checkpoint/：原完整模型 checkpoint，新增绑定身份在 controller 内，保持原存储格式。
- ACCEPTANCE.json：逐字段零差异、原/新身份隔离、旧/新恢复与 Cell6 错身份拒绝结果。
- CELL6_CALIBRATION_RECORD_IMMUTABLE.json：不可变、带内容哈希的已有条件校准记录。

## 七类对象

RunContext 明确物种标签、眼、光学、背景、单位、输入合同、时钟、ROI和seed。
缺失的眼/实际解剖物种/物理光学信息保留unknown；源包缺master seed时保留None，
同时登记完整原RNG状态身份，不另造seed或重新采样。

PopulationIndex 复用现有 cell/group ID、类型、坐标、偏心度、方位、成员数和坐标角色。
cone_owner/compartment无法由当前已绑定资产唯一确定时保留None/源模型角色，不用最近邻伪造。
现有源细胞表引用保留在原artifact；本版不为元数据缺口重建全视网膜或复原独立成员轨迹。

IntrinsicParameterRegistry 保存 ParameterRecord、Applicability、Evidence、AggregationRule，
ResolvedParameterBundle保存FP64值及provenance引用。StateBundle独立，不把状态/RNG/历史混成静态theta。
RouteRegistry直接引用原CSR、归一化和contact/gap身份，不重建边。CalibrationRecord独立、不可变，
proposed/diagnostic/conditional/validated_for_scope分开，架构迁移不提升任何资格。

## 规则与单位

规则为 shared_constant、type_constant、eccentricity_function、retinotopic_function、
background_rule、derived、conditional_override。解析顺序为原单位/身份、shared/type、
位置/背景、派生、精确conditional；同级重复来源、单位冲突、缺依赖/循环、越域均报错。
proposed目录项不能直接成为执行绑定，engineering_default要求显式工程预览。

位置规则目前支持有来源的1D分段线性：声明axis、axis_unit、coordinate_frame、anchors、
values和interpolation=linear，禁止外推；支持eccentricity或已声明x/y坐标及背景轴。
不执行任意字符串/Python表达式。派生仅白名单sum/product/ratio、显式依赖和量纲检查。
已有空间函数通过原图/CSR及其来源身份复用，未重新拟合或生成新梯度。

SI F/S/V/A、Td、R*/s、Hz与源模型native/normalized/effective量分开。
原source未给出SI映射的量以vH_native:/Chen_native:/source_native:命名，
不能拿这些标签满足F/S/V等物理端口；此限制没有通过猜测换算消除。
矩阵和每类型不等长kernel保留各自数组，不拼成不合法的全局同宽history。

## 群聚合

C/gL按原成员数只乘一次，均匀组严格保留原N*value计算顺序；异质显式求和。
反转电位不乘N；异质EL只能显式gL加权并声明equipotential。通用聚合API要求成员type/background身份，
混合类型/背景拒绝；阈值/门控/history/kernel不做默认平均，异质不相容直接拒绝。

兼容字典仍保存permember参数。额外的_group_total数组仅用于验证，不作为新的permember C/gL回灌。
普通V+s AC的C/gL是effective pool参数，保持原池语义，不顺带乘target成员数。
原RGC路线保持 N_post*gain*(W_mean@activation)；contact_coefficient_S已聚合，
gap Laplacian维持S，不再次乘contact/member数。W_mean、W_sum、contact和group成员数分开。
保留full-parent归一化与局部边界，禁止crop renormalization；群总率不自动替代代表单细胞输出。

## 来源、可校准状态与 Cell6

源目录导入旧全部profile；旧base值与“实际采用值”是两个命名空间。
active populations参数来自当前已选recipe和实际seed，不能用旧150/−70/5覆盖实际B5或Cell6读出。
缺少类型/偏心度证据的膜、release、history、门控保持原值并标uncalibrated_for_eccentricity。
runtime_eccentricity_deg与evidence_validated_eccentricity_deg独立，0–6°目标不是已验证范围。

Cell6实验输出校验biological_cell_id、target group/类型/column/members、input contract、
background policy、readout model/artifact、训练/评估及source artifact/code身份和资格。
现有GUI/缓存guard仍验证具体stimulus/payload；数学层本身不能识别伪装节点，调用者必须提供声明来源的真实V。
输入合同不等于一条训练波形：已登记原Cell6工程输入类及用户另行批准的627ms受控cone合同。
其他新刺激/背景/类型/group无隐式放行，不能用最近邻替换。约9.036ms是冻结精确有效编码时间，
g0–g5为刺激组，工程约4.73°不是Cell6实测位置。

CalibrationRecord记录现有训练/划分/SHA、输入/模型、损失、范围、参数、不确定性、证据、
性能、资格、限制和授权。write_immutable拒绝覆盖；read_immutable复核内容哈希。
原物理膜、旧放电与实验放电及各自history继续独立。

## 兼容与限制

新binding ID、route ID、协议字段与checkpoint controller身份隔离，不与旧缓存混用。
新Runner默认拒绝旧checkpoint；显式allow_legacy=True仍由原GroupTimeArm完整验证旧控制器身份、
时钟、RNG、前史、frontier与队列，不忽略旧哈希。普通旧GroupTimeArm和旧配置仍可使用原包。
StateBundle初始状态不是动态控制器快照，不能拿来冒充完整恢复。

H2在当前native H1机制中not_implemented；两室Cone参数/矩阵可登记和按现有ComponentSpec使用，
不向当前native H1路径偷偷新增两室。缺失的fine成员/owner/subtype信息明确unknown；
没有以架构工作补造生理数值或全范围资格。见 COVERAGE.md。

本轮只做必要工程检查，所有历史失败保留。FIT_READY=false、absolute monitor=false、
Cell6 INCONCLUSIVE及原生产默认不变；不拟合、不跑长期全链、不Git、不上传。
