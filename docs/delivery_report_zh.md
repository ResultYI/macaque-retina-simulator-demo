# 猕猴连续视网膜模拟器：阶段 A/B 交付总报告

2026-10-10。独立项目实际位于 `D:\macaque_retina_simulator`。已完成本次确认的 A/B 工程交付：来源／单位／物种门禁、显式坐标与连接、光积分算子、持续状态求解及隔离作者线性参考。修复后 A 54 项、B 8 项实际通过。完整生理模式仍有 15 项阻断；没有声称完整模拟器或真实生理符合度已经完成。

完整目标保留中心、过渡区与周边同时连续覆盖，非均匀 L/M/S 采样、逐细胞类型／位置、HC/BC/AC/RGC 与 ON/OFF、soma/RF 位移、稀疏连接、区域依赖动力学、给定眼动、逐层输出和独立生理验证。参考 patch 与合成单测仅属于本次有限交付。

## 1. 实际改动与文档

| 文件 | 实际用途 |
|---|---|
| [retina_sim/contracts.py](../retina_sim/contracts.py) | 单位／语义／坐标、CellTable、StateSpec、逐块导数与持续状态 |
| [retina_sim/provenance.py](../retina_sim/provenance.py) | 来源资产、参数／物种／条件门禁，完整需求 preflight |
| [retina_sim/parameter_contracts.py](../retina_sim/parameter_contracts.py) | 作者参数的显式接口单位与 shape，无生理默认值 |
| [retina_sim/graph.py](../retina_sim/graph.py) | 带方向和证据的连接对象与 CSR；空行拒绝 |
| [retina_sim/density.py](../retina_sim/density.py) | 严格密度支持域、逐节点出处、显式插值／倍率、源码 Rc 顺序 |
| [retina_sim/photometry.py](../retina_sim/photometry.py) | 光谱功率及 R*/s 算子、面积／波长换算和效率重复计数拒绝 |
| [retina_sim/engine.py](../retina_sim/engine.py) | 联合 ODE、自适应求解、绝对时间、chunk／checkpoint／身份验证 |
| [retina_sim/reference.py](../retina_sim/reference.py) | 显式 opt-in 的作者 Gaussian-HC 线性参考、输出率、独立数值 oracle |
| [retina_sim/schema.py](../retina_sim/schema.py) | 实际使用的 schema 子集校验 |
| [tools/extract_packer_cone_density.py](../tools/extract_packer_cone_density.py) | 严格 60 行原字面量提取及只读 verify |
| [tools/build_registry.py](../tools/build_registry.py) | 从实际固定源码建立账本和 v2 schema |
| [tools/coverage_report.py](../tools/coverage_report.py) | 20 项需求及完整模式 15 项阻断 |
| [tools/run_checks.py](../tools/run_checks.py) | 按次保留命令、退出码、日志、AST 与误差 |
| [tests/test_stage_a.py](../tests/test_stage_a.py) | 54 项工程／来源／单位／状态负例和数值校验 |
| [tests/test_stage_b.py](../tests/test_stage_b.py) | 8 项 synthetic 作者线性参考验收 |

[实施合同](implementation_contract_zh.md)、[Pro 三稿及审查](pro_consultation_and_review_zh.txt)、[原会话文本](pro_original_message_archive.json)、[来源规则](evidence_review_rules.md)、[独立代码复核及修复](review_report_zh.md)、[逐项覆盖表](requirements_coverage.md) 均已保存。咨询原文保留错误，不作为执行规格。传入咨询消息被读取工具截断，末稿尾部从原 Pro 会话完整取回；重叠前缀一致，三稿正文剔除会话引用格式和空白后的比较全部一致，未删改论述。另存工具返回的三份原始消息 JSON 以保留原引用标记。

## 2. 真实验收与日志

最新验收：`20261010T111806851135Z`。解释器：`C:\Users\PC_User\anaconda3\envs\retipath_v1\python.exe`。

| 检查 | 命令（在项目目录执行） | 退出码 | 真实日志 |
|---|---|---|---|
| 60 行来源提取核对 | `python -B tools\extract_packer_cone_density.py --verify` | 0 | [check_0.log](../reports/20261010T111806851135Z/check_0.log) |
| A 组：54 passed | `python -B -m pytest -p no:cacheprovider tests\test_stage_a.py -q` | 0 | [check_1.log](../reports/20261010T111806851135Z/check_1.log) |
| B 组：8 passed | `python -B -m pytest -p no:cacheprovider tests\test_stage_b.py -q` | 0 | [check_2.log](../reports/20261010T111806851135Z/check_2.log) |

聚合实际命令为上述固定解释器加 `-B .\tools\run_checks.py`，退出码 0。AST 解析无语法错误；未配置 lint 或静态 typechecker，没有安装工具或声称执行了这些检查。所有刺激、几何、单测状态及数值输出均标 `synthetic_fixture`；参考参数来自作者代码，其证据类别仍保留为 source_fit／author_assumption。

[最新机器验收记录](../reports/verification.json)，[本轮不可替换记录](../reports/20261010T111806851135Z/verification.json)。首轮 A 38/B 8 日志仍保留在 `reports/20261010T110622396136Z/`。独立 reviewer 复现非空错单位加载 bug，修复并增加四项双入口回归；另补混合单位导数、支持域与执行 preflight 的工程校验。没有通过放宽容差隐藏失败。

## 3. 数值结果及适用限度

| 验收量 | 实际最大误差 | 单位／含义 |
|---|---|---|
| 常谱光积分解析值 | 2.22045e-16 | 相对误差 |
| 源码 Rc 变换／插值顺序 | 8.67362e-19 | deg |
| 单段与 chunk＋checkpoint | 1.28092e-14 | mV |
| 混合单位合成 ODE | 1.11022e-16 | 按各状态声明单位比较的数值误差 |
| 均匀恒定输入稳态 | 8.88178e-16 | mV |
| ZOH 粗求解与 block-exponential | 3.68491e-06 | mV |
| ZOH 细求解与 block-exponential | 1.60749e-12 | mV |
| ZOH 断点与分段 oracle | 6.37823e-14 | mV |
| 正弦初始瞬态后的残差 | 2.23987e-14 | mV |
| 正弦稳态幅值 | 1.33227e-15 | mV |
| 正弦稳态相位 | 1.40494e-14 | rad |

细求解使用 DOP853、float64、rtol=1e-10、atol=1e-12、max_step=0.002 s，输出采样间隔 0.001 s，与内部自适应步分开。正弦比较忽略 t<0.8 s 的初始瞬态，预先固定残差判据 1e-8 mV；没有为拟合改变参数。这些容差属于计算设置，不是生理常数或动物测量误差。独立数值 oracle 是同一线性系统的数学一致性校验，不是独立生理数据。

## 4. 来源、数据与哈希

当前账本有 12 个来源对象、9 个作者参数、1 个具名参考配置；没有可运行的 macaque 主线默认配置。

| 关键清单／数据 | SHA256（文件字节） |
|---|---|
| [sources/registry.json](../sources/registry.json) | `d0ce10602a753a0a140795637460a3a75b1ac2eab1e1211da7a59ce9cfe16d42` |
| [sources/author_asset_manifest.json](../sources/author_asset_manifest.json) | `228f6913ceb25d825bbc6554e48eaada75a7faf44ab9b84072d64fbe634802a0` |
| [data/published_code/packer_cone_density.csv](../data/published_code/packer_cone_density.csv) | `897ab2056e5a92cd0180b09ee4b97fe01cf010cc17f269fb15d94e352515ba5c` |
| [data/published_code/packer_cone_density.manifest.json](../data/published_code/packer_cone_density.manifest.json) | `b29922ef71816c8f58c85c6cc5226687715db8ebf808e7ae02091b7e484b9c52` |

全部 12 个来源资产的字节身份与位置：

| 来源 ID | 固定资产／代码位置 | SHA256 |
|---|---|---|
| SF2025_runColFun_m | [data/source_snapshots/somaratna_freeman_5406d153/runColFun.m](../data/source_snapshots/somaratna_freeman_5406d153/runColFun.m)；calConv 52-65; calDeriv 67-107; gang 266-298; solveF 413-469; solveT 471-510; weight 512-550 | `2250208391e483f521320163eb9803e2349ed9bd83cb18ef75f36dde8547533b` |
| SF2025_setColLit_m | [data/source_snapshots/somaratna_freeman_5406d153/setColLit.m](../data/source_snapshots/somaratna_freeman_5406d153/setColLit.m)；literature-model constants and fit coefficients, entire file | `a85b2f03535acaabb211feff89fcd3c3f3ed8c012417d7076b0acf447a8abeb8` |
| SF2025_runCol_m | [data/source_snapshots/somaratna_freeman_5406d153/runCol.m](../data/source_snapshots/somaratna_freeman_5406d153/runCol.m)；entire source file | `84a311331cc07c7fa73129453916862f3c25669914004e650a5332d070a70541` |
| SF2025_runColTask_m | [data/source_snapshots/somaratna_freeman_5406d153/runColTask.m](../data/source_snapshots/somaratna_freeman_5406d153/runColTask.m)；resp: output rectification line 666 | `2bb25c70e69f4e4cfdbbefd8d5505cd8322cca192cd5c714ceb24d6f10877c39` |
| SF2025_setColFun_m | [data/source_snapshots/somaratna_freeman_5406d153/setColFun.m](../data/source_snapshots/somaratna_freeman_5406d153/setColFun.m)；entire source file | `625ebc27f1cc5cf2924d1e32b41d7f3a453b035a42e6f9c547a946db9e584293` |
| SF2025_setColTask_m | [data/source_snapshots/somaratna_freeman_5406d153/setColTask.m](../data/source_snapshots/somaratna_freeman_5406d153/setColTask.m)；entire source file | `ca9bddb438e58f01663d5a34e1c15e9cf658d6bffc8e224c444729f573d019de` |
| SF2025_setCol_m | [data/source_snapshots/somaratna_freeman_5406d153/setCol.m](../data/source_snapshots/somaratna_freeman_5406d153/setCol.m)；entire source file | `09664520e16a41022a0a635221ca04c9a94a74f5d05d7039c163c48370fbb242` |
| SF2025_License_txt | [data/source_snapshots/somaratna_freeman_5406d153/License.txt](../data/source_snapshots/somaratna_freeman_5406d153/License.txt)；entire source file | `45df43d765bb3d039232a1279716d3ed4f6089fa24947e9d1ed3b90d61490f5e` |
| SF2025_Cone_density_mat | [data/source_snapshots/somaratna_freeman_5406d153/Cone density.mat](../data/source_snapshots/somaratna_freeman_5406d153/Cone density.mat)；entire source file; binary MCOS table not numerically decoded | `bb13fdc1fe18025e1e05c6382789a828126d125ca2411bf2d4b9a3def27d2c41` |
| SF2025_Ganglion_cell_density_mat | [data/source_snapshots/somaratna_freeman_5406d153/Ganglion cell density.mat](../data/source_snapshots/somaratna_freeman_5406d153/Ganglion cell density.mat)；entire source file; binary MCOS table not numerically decoded | `9eed078b386e09741117f7088bb4b5299e780f61a47fa9c4e51c9159490fe95e` |
| ISETMacaque_PackerConeRc | [data/source_snapshots/isetmacaque_cf22ac9d/PackerConeRc.m](../data/source_snapshots/isetmacaque_cf22ac9d/PackerConeRc.m)；macaqueConeDensity: data 59-73,78-90,103-121,126-138; scaling 75,93,123,142; Rc 14-21,30-31 | `72408017cb5b6c4996b48c95050963057f61aab6e39154ea095062a8b5c35e9d` |
| ISETBio_WatsonSpacing | [data/source_snapshots/isetbio_d2924c2f/WatsonRGCModel.m](../data/source_snapshots/isetbio_d2924c2f/WatsonRGCModel.m)；lines 113-117; spacingFromDensity=sqrt(2/(sqrt(3)*density)) | `796719bb03651887cddd36040b53e6db5aeedf4372888d37bb3450e6e7497a00` |

ISETMacaque commit 为 `cf22ac9d681482ca1fcc365a423ef0c2be7916a3`，保留 PackerConeRc.m 与 MIT 许可。60 节点为颞侧 28、鼻侧 32，mm 位置和源码密度×1e3 的 cones/mm²；原字面量／逐行位置可查。其分类为 published_code_tabulation，不是原始 measured_source_table。Packer 原文测量、确切样本物种与不确定性未独立核完，不能据仓库名补全。不能补 e=0、未知经线或表外域，不能当完整二维模型。

ISETBio commit 为 `d2924c2fe1d41c7dc5af1006d8081735b08f1b35`。仅核到代码中的完美六角 spacingFromDensity 数学关系；没有把它当作猕猴逐位置解剖测量。ISET 代码资产的原始 publication_year 保留 null，2026 只是本次审计日期。

作者代码为 AlanFreeman/Colour 固定 commit `5406d1533dae1678f94687379ac078a612f6170f`，按 License.txt 保留 CC BY 4.0。相关 runColFun、setColLit、runColTask 的函数／常数已静态核对；其他下载文本只标 identity_verified，不冒称完整代码审计。两份 MAT 的 Git blob 为 Cone density `fe56fe90e041d5699538dd2d78e2afa5757c708d`、Ganglion cell density `10eb6eca68976793a4db7164fdb6e22850e5330f`，仅身份通过，MCOS 数值未解码。

九项参数仅独立核到作者 setColLit 的实际字面量和引用注释，没有独立核完上游实测论文、样本条件或拟合不确定性：

| 参数 | 账本值 | 单位 | 证据类别 |
|---|---|---|
| SF2025_tau | `0.014` | s | author_assumption |
| SF2025_kSens | `17.8` | mV/contrast-unit | author_assumption |
| SF2025_kSur | `1.54` | dimensionless | author_assumption |
| SF2025_kRect | `7.2` | Hz/mV | author_assumption |
| SF2025_actRest | `[21.8, 25.4]` | Hz | author_assumption |
| SF2025_magRet | `4.73` | deg/mm | author_assumption |
| SF2025_radOptCoef | `[1.4241e-05, 0.0, 0.011663]` | polynomial coefficients for radius_deg(e_deg), descending powers | source_fit |
| SF2025_radHorCoef | `[-4.5667e-05, 0.0030709, 0.029977, -3.261]` | polynomial coefficients for log(radius_deg)(e_deg), descending powers | source_fit |
| SF2025_radGangCoef | `[-1.3583e-06, 0.00014598, 0.0012385, 0.012601]` | polynomial coefficients for radius_deg(e_deg), descending powers | source_fit |

作者引用含人 PSF／位置 jitter、跨猕猴物种来源以及猫 V1 增益等混合来源；这个生理来源结论沿用提供的独立审计，当前执行只核了作者文件，没有再读完上游论文。整个配置明确标 source_specific_mixed，只能显式选择 component_reference／numeric_fixture，禁止进入完整 macaque 主线默认。具体来源读取程度和未知项保留在 registry.upstream_sources_checked。

## 5. 完整需求逐项状态

| ID | 完整需求 | 状态 | 机器 gate |
|---|---|---|
| P01 | 物种与来源分离政策 | 局部实现 | POLICY_CONFIRMED |
| S01 | 中心—过渡区—周边连续全模型 | 来源待补 | BLOCKED |
| S02 | 非均匀 L/M/S cone 阵列 | 来源待补 | BLOCKED |
| S03 | 真实标定光输入至 R*/s | 来源待补 | BLOCKED |
| S04 | HC 与反馈 | 来源待补 | AVAILABLE_REFERENCE |
| S05 | ON/OFF BC 类型、非线性和连接 | 来源待补 | BLOCKED |
| S06 | AC 类型、状态与靶细胞 | 来源待补 | BLOCKED |
| S07 | RGC 类型覆盖与 spike 机制 | 来源待补 | BLOCKED |
| S08 | 解剖／功能位移及 Jacobian | 来源待补 | BLOCKED |
| S09 | 有向稀疏连接工程 | 数值验证 | ENGINEERING_VALIDATED |
| S10 | 区域依赖连续动力学 | 来源待补 | BLOCKED |
| S11 | 给定眼动与持续状态路径 | 来源待补 | BLOCKED |
| S12 | 带单位和元数据的层输出 | 数值验证 | ENGINEERING_VALIDATED |
| S13 | 独立生理验证 | 来源待补 | NOT_RUN |
| S14 | 数值／覆盖／生理分开报告 | 数值验证 | ENGINEERING_VALIDATED |
| E01 | 来源、参数与单位门禁 | 数值验证 | ENGINEERING_VALIDATED |
| E02 | cone 密度原始测量证据 | 来源待补 | PARTIAL_SOURCE |
| A01 | 生理 cone aperture | 来源待补 | BLOCKED |
| A02 | 光转导与非线性适应 | 来源待补 | BLOCKED |
| B01 | 作者原阵列及论文复现 | 来源待补 | BLOCKED_DATA_DECODE |

完整模式的 15 项阻断是：S01、S02、S03、S04、S05、S06、S07、S08、S10、S11、S13、E02、A01、A02、B01。每项缺证据、实现范围与有限声明见机器表和中文覆盖表；require_runnable 同时返回全部原因并抛错，SimulationResult 也拒绝本阶段 full_physiology。工程连接和层输出通过不代表真实突触图或全细胞类型覆盖。

## 6. 当前参考能力与未实现机制

参考内核含 cone、horizontal、ON/OFF bipolar 和 ON/OFF ganglion 六组联合状态，固定 tau，mV 偏差及明确 resting offset，输出 Hz 整流 rate。W_ch 和 W_on 排 S 后归一化；W_hc 从原始 Gaussian a 转置后重新归一化并保留 S。没有把 W_ch 的转置当反馈，没有按层顺序创造 dt delay，没有截断全 Gaussian 核或补未知边。

输入只是已包含光学 attenuation 的 cone contrast。rOpt 作为作者具名半径记录，并未在此输入端再次施加光学滤波。没有自动 R*/s 输入、光照依赖适应、AC、完整 BC/RGC subtype、spike-time／Poisson／Bernoulli／refractory、完整 LMS 区域规则、解剖位移、二维连续全视网膜或标定眼动投影。horExp=true 与未知 adaptation 选项明确报错。

光积分是严格单位算子，通过的物理解析单测使用合成有效面积／效率。缺实际校准时 RGB/radiance 路径拒绝，不靠虚构光学参数生成绝对生理刺激。密度表没有用于虚构全场 mosaic，作者 MAT 没有用合成阵列代替并宣称复现成功。

## 7. 环境、下一项证据与停止点

使用已有 retipath_v1：Python 3.11.16、NumPy 2.4.6、SciPy 1.17.1、pytest 9.1.1、h5py 3.16.0，PyTorch 2.14 + cu130 已导入且 CUDA 可用。GPU 先前预检为 RTX 4070 Ti SUPER 16 GB；本次 A/B 实际是 CPU float64 数值验收，没有运行 GPU 科学实验。Matplotlib 和外部 jsonschema 缺失但没有安装；不影响当前验收。D 盘写入经正常审批成功，固定来源下载成功，没有遗留权限或下载阻断。旧 Macular、ISET 项目只读借鉴，未恢复旧实验。

下一项需要的具体证据：先核定具名猕猴物种及照明条件下的 phototransduction／adaptation 机制原方程、状态变量、参数单位／样本条件／支持域与独立验证数据，不能把 Saha flash-response 描述性拟合直接写成未知 ODE。并继续取得 e=0 与其他经线的密度／LMS 规则、真实 aperture 与光学校准、soma↔RF 位移与细胞守恒、BC/AC/RGC subtype／连接证据。作者原阵列复现另需可靠解码和逐列单位／坐标审计，不能仅凭 MAT 字节身份解除阻断。

本阶段停止于已授权 A/B。真实生理符合度：未验证；数值一致性：以上真实测试通过；实现完整度：基础工程与具名参考局部实现，完整目标仍阻断。没有训练、动物实验、生物学结论图或独立生理成功声明。
