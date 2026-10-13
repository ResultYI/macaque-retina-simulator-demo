# 来源、单位与审查规则

合同优先级为最新确认的完整目标和 A/B 实施合同、独立核源附录、实际源文件、最后才是待审的 Pro 三稿。三稿归档不构成可信认证；其中错误方程不得流入实现。

来源账本 v2 将来源对象、参数对象、物种／来源配置与实验条件分开。每项参数保留值、shape、单位、来源及位置、变换、证据类别、物种、样本条件、区域、经线、坐标系、支持域、不确定性与允许模式。缺项用 null 或 unknown 保留并拒绝生理运行，不能临时编值。

证据类别是 `measured`、`source_fit`、`author_assumption`、`project_assumption`、`synthetic_fixture` 或 `unknown`。文献拟合、数学推导、作者假设和项目假设与实测分开；变换写入 derivation。`measured` 需要独立核对的 primary_measurement 来源，代码资产身份校验不能替代原始实测审查。项目假设进入完整模式须有已批准引用。

公开源码转录的 Packer 密度表使用 `published_code_tabulation` 标签。原始测量、具体物种、样本计数与论文不确定性尚未独立核完，不得改称 measured_source_table。查询不能补 e=0、表外范围或未知经线；mm 到 deg 使用显式具名倍率，密度转换使用倍率平方。Rc 按源码先密度转间距／半径，再插值。

共同框架按猕猴物种与来源分开配置。不作无依据的平均；人、猫和未知混合物种不能作 macaque 主线默认。Somaratna–Freeman 是具名作者线性参考，需要显式 opt-in，其 mixed provenance 和上游未核部分保留在输出及账本。

单位同时校验语义和坐标系。contrast、R*/s、mV 和导数不能互传；绝对电压、偏差 generator 和含率偏置 generator 不能互换。soma 与 RF 字段分开，未知位移不能由改标签冒充。CSR 行为 post，列为 pre。真实离散细胞连接和面积求积是不同对象。

绝对光积分的输入是经过光学后的视网膜光谱辐照度 W/(m² nm)。归一化 aperture 的面积加权均值须再乘有效面积；无量纲 mask 则直接面积积分。波长 nm 必须转换 m，光谱求积带宽仍为 nm。效率须有明确概率定义，且排除已计入辐照度的重复透过率。缺真实 optics／spectrum／pupil／area／efficiency profile 拒绝 RGB 到 R*/s；synthetic 校验仅证明单位算子。

状态由联合 RHS 演化，不按层更新制造隐式 delay。checkpoint 保存绝对时间、ordered state、各项 hash、求解器及输入插值合同；恢复拒绝不兼容参数、图、模型或状态注册表。没有获准随机机制，不创建 RNG/spike 状态。

数据用途须分别登记 parameterization、calibration、independent_validation、synthetic_test、reference_only 或 not_assigned。同一曲线用于定参数后不能再充当独立验证。当前作者代码与 Packer 表只作 reference_only；所有数值图、刺激和输出逐对象标 synthetic_fixture。没有进行动物实验、独立生理验证或训练。

Schema 检查不是科学验证。当前依赖无外部 jsonschema，项目实现并测试了实际 schema 使用的 JSON Schema 子集；不声称实现完整标准。没有配置 lint／静态 typechecker，仅实际运行 AST 与运行时合同检查。

完整模式执行前必须同时报告所有阻断并报错，不能静默降级到 reference。结果构造也拒绝本阶段 full_physiology。覆盖状态只使用未开始、局部实现、来源待补、数值验证或生理验证；机器 gate 单列。工程验证不能改写为生理验证。
