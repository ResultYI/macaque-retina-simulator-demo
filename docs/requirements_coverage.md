# 完整需求覆盖表

状态分为未开始、局部实现、来源待补、数值验证、生理验证；gate 控制完整模式。

| ID | 模块 | 状态 | Gate | 缺失证据 / 限制 |
|---|---|---|---|---|
| P01 | species/source policy | 局部实现 | POLICY_CONFIRMED | 逐物种、区域与条件须各自核验，不以混合来源替代主线默认 Human/cat excluded from main defaults; author mixture opt-in only |
| S01 | continuous centre/transition/periphery model | 来源待补 | BLOCKED | 中心、过渡和周边尚无完整生理 profile No complete retina claim |
| S02 | nonuniform L/M/S cone mosaic | 来源待补 | BLOCKED | e=0、各 meridian、真实二维 L/M/S mosaic 与 S cone 参数缺失 60 meridian nodes are not a 2-D mosaic |
| S03 | calibrated light to R*/s | 来源待补 | BLOCKED | 标定 optics、pupil、光谱与区域输入参数未具备 Unit operator tested only with synthetic_fixture |
| S04 | horizontal cells and feedback | 来源待补 | AVAILABLE_REFERENCE | 完整 HC 类型、非线性反馈与连接证据缺失 Named H1 linear reference does not unblock main model |
| S05 | ON/OFF bipolar types, nonlinearity and wiring | 来源待补 | BLOCKED | ON/OFF BC subtype、非线性和连接参数缺失 Midget linear reference only |
| S06 | amacrine types, states and targets | 来源待补 | BLOCKED | AC 类型、状态与目标连接尚未实现 No AC implementation or substitute |
| S07 | RGC type coverage and spike mechanism | 来源待补 | BLOCKED | 完整 RGC 类型与 spike 生理机制缺失 Midget rate only; no invented stochastic spikes |
| S08 | anatomical/functional displacement and Jacobian | 来源待补 | BLOCKED | 生理位移模型、Jacobian 和区域坐标参数缺失 Separate nullable coordinates are not a displacement model |
| S09 | sparse directed connectivity | 数值验证 | ENGINEERING_VALIDATED |  CSR postsynaptic rows; source-free edges cannot enter physiological runs |
| S10 | regional dynamics and continuous profile | 来源待补 | BLOCKED | 区域依赖动力学与连续参数过渡缺失 No default regional interpolation |
| S11 | given gaze and persistent state | 来源待补 | BLOCKED | 给定眼动到标定视网膜光输入的完整映射缺失 Streaming state tested; calibrated gaze pathway absent |
| S12 | layer outputs with units and metadata | 数值验证 | ENGINEERING_VALIDATED |  Reference layer ports; not all physiological cell classes |
| S13 | independent physiological validation | 来源待补 | NOT_RUN | 独立、未用于定参数的生理验证数据与条件缺失 Numerical fixtures never count as physiological observations |
| S14 | separate numerical, coverage and physiology reports | 数值验证 | ENGINEERING_VALIDATED |  Evidence reports do not certify biology |
| E01 | source/parameter/unit gates | 数值验证 | ENGINEERING_VALIDATED |  Schema plus semantic/profile/asset checks; unknown full parameters rejected |
| E02 | cone density primary evidence | 来源待补 | PARTIAL_SOURCE | 真实二维密度、位置、物种和条件证据仍不完整 published_code_tabulation, not measured_source_table |
| A01 | physiological cone aperture | 来源待补 | BLOCKED | 生理 aperture profile 与来源参数缺失 Aperture unit interface is not a biological parameter |
| A02 | phototransduction and nonlinear adaptation | 来源待补 | BLOCKED | 中心、过渡、S cone、电压耦合、terminal Ca/HC，以及真实来源对照和独立验证缺失 Saha fits not converted into an unknown ODE |
| B01 | author arrays and paper reproduction | 来源待补 | BLOCKED_DATA_DECODE | Somaratna–Freeman 作者 MCOS 数值表未解码 Source-faithful kernel only, no original-array/paper reproduction |

完整模式阻塞见 reports/full_mode_blockers.json。数值检查通过不认证生理符合度；本表保留所有完整主线需求。

## A02 外周 L/M 子项

Chen2024 外周 >20deg L/M：数值验证 / SOURCE_PARITY_VALIDATED。A02 父项仍 BLOCKED；真实数据身份、字段与当前对照阻塞见 reports/chen2024_data_inventory.json 和 docs/chen2024_real_data_report_zh.md。
