# 猕猴水平细胞空间分布候选方案

2026年10月11日更新。用户随后明确要求“你别只写方案，你写代码啊”，已按这项授权在既有项目实现下述具名合成路径，并接入配置、GUI、完整网络和导出。实施阶段状态原为IMPLEMENTED_NOT_RUN。用户随后明确授权“测试一下”，已完成下文记录的定向工程回归及一次小ROI运行；未恢复拟合、对照实验或生理统计验证。此前Pro讨论与后续源码比较的证据范围分别保留；新增实现不归为Pro已验证结果。

## 结论与范围

建议采用“连续总HC密度与H1比例分开定义，联合生成带短程排斥的H1/H2非均匀mosaic”。中央不再铺满18400/4600的峰值，也不从cone-pedicle-free区域推断HC-free孔。推荐首版合成密度支持物理偏心度0–7.1mm；这是跨动物来源约束下的可替换合成空间场，不是同一只猴的全眼实测图。

中央连接的推荐组织方式是：沿同一个cone ID，从感光输入位置追踪到移位后的OPL pedicle，再与末端附近HC的局部arbor建立接触。中央视野cone不必对应一个位于RF原点的HC，也不必要求中心HC跨越全部Henle位移。仍需形态支持域及接触约束，不能以mosaic完成冒充功能连接闭合，更不能冒充HC动力学标定。既有2–3mm和4–5mm完整halo独立patch模式保留原有来源和身份；新方案已作为另一个明确命名的hc_joint_synthetic_v1实现；原source_patch_v1仍是默认HC模式，不自动替换已有配置。

## Pro讨论与审查结论

实际使用我们的云浏览器，在[研究方案评估会话](https://chatgpt.com/c/6ac70c68-0814-83ee-a87c-1077bf332d51)中核实并选择“Pro，第5项，共5项”，完成本题讨论。提交内容仅含本项目已授权默认参数、实现缺口和公开文献。

Pro最终认可中央log-smoothstep、分型比例、分区积分配额、联合soft-core及当时提出的候选contact构造，同时强调中央contact未闭合。初稿中的以下错误已在讨论中提出并纠正：固定全局N不保证局部密度；新插值不是published_fit；arbor anchor可以暂与soma同位，但功能RF不能直接等于soma；divergence分母是cone数，convergence分母是HC数；逐ring独立round不能保证全域计数。后续实现比较还纠正了一项判断：anchor处eligible密度为零只使局部均匀半径公式失效，不能据此判定整个有限arbor没有可接触的pedicle。该修正和下节连接路线是新增审查结果，不归为Pro已核结论。

## 其他模拟器怎样建模及本方案的取舍

以下结论来自2026年10月11日对官方源码、文档和作者论文的静态核查。所核路径未提供经过验证的“foveal输入cone→Henle位移pedicle→当地HC arbor”通用现成解；不能把引擎能力或外围实现当成中央已验证配置。

| 实现 | 实际采用的空间组织 | 可借鉴部分与限制 |
|---|---|---|
| ISETBio，commit d2924c2fe1d41c7dc5af1006d8081735b08f1b35 | 在cone RF坐标计算RGC surround，H1双指数核输出cone索引与权重；该路径没有逐HC soma/arbor及独立pedicle几何 | 复用稳定cone ID、类型过滤、稀疏权重；双指数作独立功能参照，不替换显式HC |
| ISETMacaque，commit cf22ac9d681482ca1fcc365a423ef0c2be7916a3 | 在cone RF坐标做Gaussian center/surround pooling，拟合中央RGC调谐 | 借鉴索引与加权组织；中央功能拟合不等于中央cone–HC解剖连接 |
| Macular 1.5官方文档及作者论文 | PR/HC作用合入BC时空滤波；点细胞和等距层网格 | 有效通路可作独立对照；官方文档说明现有均匀密度不支持fovea。本次未逐行核源码 |
| NeuronC的primate H1模型，Packer与Dacey2005 | 外围实测cone/HC位置，独立cone pedicle、HC soma及两者间简化cable；直接接触和HC gap分开 | 最接近可借鉴的显式接触组织。其120μm树突直径属于该外围模型，不能作为中央默认半径 |
| Retsim官方连接器 | 可用真实分支节点，或在允许范围内向photoreceptor接触点延伸HC树突；约束x/y、z、pre/post区域及间距 | 借鉴pedicle→允许arbor分支的连接原语；不能声称手册已验证中央Henle/pedicle环配置 |
| Virtual Human Retina，ModelDB 2018247，commit 0c99e094e5cff98ae43264e5f7ee48011e1a08e3 | 密度驱动HC位置和有效半径；示例HC及cone模板只有soma，无独立pedicle/HC dendrite | 借鉴将位置和连接规则分离。作者范围1–5mm；不移植其密度floor或半径到猕猴foveola |

关键一手入口：[ISETBio实际RF距离缓存](https://github.com/isetbio/isetbio/blob/d2924c2fe1d41c7dc5af1006d8081735b08f1b35/ganglioncells/%2BRGCMosaicConstructor/%2Bhelper/%2BsurroundPoolingOptimizerEngine/cache.m#L13-L47)、[H1双指数实现](https://github.com/isetbio/isetbio/blob/d2924c2fe1d41c7dc5af1006d8081735b08f1b35/ganglioncells/%2BRGCMosaicConstructor/%2Bcompute/%2BsurroundConePoolingWeightsUsingComputeFunctionFor/PackerDaceyDoubleExponentModel.m#L30-L72)、[ISETMacaque坐标来源](https://github.com/isetbio/ISETMacaque/blob/cf22ac9d681482ca1fcc365a423ef0c2be7916a3/toolbox/%2Bsimulator/%2Bfit/conePoolingDoGModelToSTF.m#L18-L28)、[Macular官方文档](https://macular.gitlabpages.inria.fr/macular/user_doc/Macular/main.html)、[Packer与Dacey2005原文](https://bpb-us-e1.wpmucdn.com/sites.uw.edu/dist/6/8438/files/2020/10/orin-dacey-2005.pdf)、[Retsim连接器手册](https://retina.anatomy.upenn.edu/~rob/ncman6.html)、[Virtual Human Retina接触代码](https://github.com/ModelDBRepository/2018247/blob/0c99e094e5cff98ae43264e5f7ee48011e1a08e3/Full%20Field%20Light%20Stimulation/create_Connections.hoc#L699-L741)。

**本方案取舍：**保留显式HC，采用两平面稳定身份和Retsim式局部接触构造。ISET的effective核只保留为独立参考或未来经批准的对照：ISETBio使用Kwide·exp(−2.3d/Rwide)+Knarrow·exp(−2.3d/Rnarrow)，R对应约10%响应半径；ISETMacaque使用exp[−(d/R)²]。两者尺度定义不能混用。宽RF成分包含网络耦合解释，不能当单细胞长树突半径；显式HC↔HC gap启用时，不再无说明叠加代表同一效应的宽核，避免重复计入。ISETBio启用S-cone surround的教程也未引入H2特有空间形态，不能作为完整H2替代。

以后提出方案及请求用户选择前，先完成相关已发表模型和可用实现的比较，说明实际坐标、状态、连接器、适用域和可迁移部分；未核源码时明确证据来自文档或论文，再说明本项目新增了哪些假设。

## 来源和适用范围

- [Röhrenbeck等1989](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1460-9568.1989.tb00349.x)，DOI 10.1111/j.1460-9568.1989.tb00349.x：摘要报告中心总HC约250/mm²、半径约0.6mm环峰23000/mm²。这里核到的是摘要级约束，没有获得中央逐型完整数字曲线。0.6mm是原文报告的近似峰位；把它固定成模型节点、指定节点间函数，才是项目近似。
- [Wässle等2000原文](https://bpb-us-e1.wpmucdn.com/sites.uw.edu/dist/6/8438/files/2020/09/Wassle-et-al.-2000.pdf)：Table2给分型密度；近fovea 18400/4600是作者以总峰和4:1比例推算。连接divergence同样包含作者的密度和收敛数推导，不能全部标签为直接测量。Table1远端patch保持独立，首版不与Table2直接拼接。
- [Eglen与Wong2008原文](https://arxiv.org/pdf/1208.0986)，DOI 10.1017/S0952523808080176；arXiv上传于2012：联合规整性可以由同型规则性加胞体尺度异型排斥解释，不必引入长程跨型配准。field B源自Wässle Fig7，206个H1和86个H2、298×300μm，局部dmin参数为21±4、32±8、12±2.5μm，正态抽样低于5μm舍弃。这些是局部模型参数，不是全眼硬核半径；原算法也不是本文的新soft-core算法。

证据字段保持measurement、published_fit、author_assumption、project_approximation四类。作者算术推导另存derivation字段并指向输入来源，不硬说成直接测量或作者拟合。cross_animal_transfer、cross_species_transfer、meridian_transfer单独存储。Table2为fascicularis来源；若目标配置为另一macaque物种，必须明确跨物种属性。全场轴对称属于项目假设。

## 连续密度场

所有坐标、半径、面积均在HC soma投影物理平面；r单位mm，密度单位cells/mm²。先定义总密度ρ和H1比例p，再令ρ1=pρ、ρ2=(1−p)ρ。

中央0≤r≤rp，推荐：

- rp=0.6mm，ρc=250，ρp=23000
- u=r/rp，s(u)=3u²−2u³
- ρ(r)=exp[logρc+(logρp−logρc)s(u)]
- p(r)=0.8

这给出有限正值，中心和峰端径向一阶导数为0，且单调升至峰值，不出现中心奇点。曲线形状、精确固定峰位以及把0.8延入中心都是project_approximation。原文0.5–1mm描述不可包装成峰位统计置信区间。ρ1(0)=200、ρ2(0)=50只是此假设的推导值，不能另列为测量。

外段使用下列原始记录。2.5和4.5mm只是区间代表位置；保留原区间，不把它们写成准确测点。

| 动物或patch标识 | 原偏心度mm | 模型代表mm | H1/mm² | H2/mm² |
|---|---:|---:|---:|---:|
| 9.8.99 | 2–3 | 2.5 | 6360 | 1586 |
| 99.5018.1 | 3.5 | 3.5 | 4700 | 1600 |
| 29.7.99 | 4–5 | 4.5 | 2665 | 947 |
| 97.3017.3 | 5.6 | 5.6 | 2278 | 933 |
| 96.5018.2 | 6.3 | 6.3 | 1950 | 1150 |
| 101.9026.7 | 7.1 | 7.1 | 1914 | 1133 |

从(rp,ρp)到各代表节点，对log(ρ1+ρ2)作保形三次Hermite插值，以PCHIP内点导数规则为基础，峰端导数显式设0。p节点取H1/(H1+H2)，对logit(p)作同样保形插值，左端导数0以衔接中央常比例。插值函数和任何平滑处理均单独记录，不修改原始记录。

有限域以0–7.1mm为首版有效计算范围。超域返回明确状态，不夹到末节点、不恒值延伸、不把800–1000的“外围”摘要描述安放到虚构偏心度。区间中点选择、不同动物组合和缺少二维方向信息意味着此曲线没有可宣称的测量置信带；敏感性方案须另获运行授权。

## 联合mosaic及计数

推荐合成模式为带固定分区配额的有限温度marked soft-core过程；目的为可审查的非晶格点集，不宣称已复现Eglen统计。旧通用等质量生成器只能复用积分、坐标、空间索引及ID设施，不能直接把其位置输出叫作已验证HC mosaic。

1. 固定canonical生成域、物理mask、源profile版本、seed和不随tile改变的全域分区。采用seed确定的不规则分区，避免将规则网格中心直接放成细胞；分区仅用于质量配额，不是cell位置。
2. 各型各区Ak的质量mk=∫Akρt(x)dx。固定分区排序，Mk为累计质量；每型从seed取一个U∈[0,1)，令Nk=floor(Mk+U)−floor(Mk−1+U)。全域该型总数为floor(Mtotal+U)，分区误差严格小于1个细胞；不逐tile或逐ring独立round。
3. 区内初始化从ρt/∫Akρt抽样。允许Nk=0或1，绝不补最低1或通过降低ρ“适配预算”。低计数窗口应报告绝对计数误差，不能用无意义的相对误差制造结论。
4. 固定数量和类型，区内提议新位置按同一归一化ρt抽样。以目标分布∝[∏iρti(xi)]exp(−E/T)且分区计数固定为定义，此独立提议下接受率为min(1,exp(−ΔE/T))。如果改为对称局部位移，必须补ρnew/ρold及一般Hastings提议比，不能沿用相同接受率。跨区操作若加入，必须保持相关配额并正确计算提议比；首版可不启用。
5. T保持正值，不用退火至零温追求最规整晶格。终止诊断、足够混合、区界伪影和局部密度残差待授权后验收；不能任意跑固定少量步数就宣称达到目标分布。

它精确保留选定分区尺度的整数配额，不保证任意细窗口的理论一阶强度恰好等于输入ρ。排斥仍会改变区内位置分布；这是需要明示的近似。分区尺寸必须随密度梯度设计并进入manifest，验收要检查不与分区对齐的窗口、多个尺度以及区界附近统计。更严格的一阶强度补偿需要专门统计标定，属于后续决定，当前未执行。

不同tile只能裁剪同一全域实例并共享global IDs。相同seed但独立生成不同大小patch，不自动保证重叠区一致；不作这种承诺。若扩大生成域，必须保留同一个预定义canonical域/实例，或明确标为重新生成。

## 同型及异型作用距离

从Eglen field B本身取ρ1,ref=206/(0.298×0.300)、ρ2,ref=86/(0.298×0.300)，ρtot,ref为两者之和。它们是该digitized窗口的经验归一化密度，不是无限总体密度；不悄悄替换成Table2的另一组计数。

候选长度字段为：

- L11(x)=21μm·sqrt[ρ1,ref/ρ1(x)]
- L22(x)=32μm·sqrt[ρ2,ref/ρ2(x)]
- L12(x)=12μm·sqrt[ρtot,ref/ρtot(x)]
- 任一pair的作用尺度Lij=[Lab(xi)+Lab(xj)]/2

以q=dij/Lij，ψ(q)=(1−q)²（0<q<1），q≥1时ψ=0。候选无量纲能量E=Σi<j aabψ(q)，首版数值选择a11=a22=a12=4、T=1；仅比值a/T有意义。精确重合位置拒绝。μm长度进入mm坐标前转换一次。

这些强度、势形、有效作用长度与密度缩放全部是新project_approximation。21/32/12来自原文局部dmin均值，但把它们转换为soft-core支撑尺度不是作者拟合；原文±SD及5μm截断仅保留为来源参数，不暗示新算法使用或复现了它们。异型没有长程配准或一一对应项；此候选的异型作用尺度较短是建模选择。

soft-core允许很近的pair，不能声称真实胞体非重叠。中央将固定10–12μm硬盘照搬全域会发生packing矛盾；本文不以虚构胞体直径解决。严格形态非重叠模式须另有偏心度相关soma/层深数据，缺参即拒绝该模式，不能把soft-core替换称为严格模式完成。

## 各坐标及接触关系

每个HC保存soma_xy_physical、arbor_anchor_xy_OPL及type/global ID。首版可设arbor_anchor=soma投影，名称为co_located_lumped_anchor；没有数据时不凭空生成轴突/树突长度或电缆delay。

HC soma由HC密度独立生成，不从cone抽样，不套cone的RF→pedicle位移图。每个cone i保留输入坐标q_i、pedicle坐标p_i=F(q_i)和同一个coneID；每个HC j保留soma s_j、OPL anchor a_j及有限arbor支持域A_j。接触查询以p_i与A_j为准，不能以q_i到s_j的距离替代。中央视野cone的pedicle可位于周围OPL环，接受该环附近HC的局部接触；是否有中心soma是另一项空间人口问题。

若有morphology，采用Retsim可借鉴的分支/末端接触判据：在允许的pre/post区域中查最近分支，施加x/y接触阈值、z层位及间距约束。若采用生成形态，先给有限骨架和允许延伸域，再向域内pedicle构造接触；不允许为了配额无界长枝。若只用参数化footprint，则按pedicle-in-footprint或有限核支持域筛选，显式标注有效arbor近似。原文外围120μm直径和任意中心radius floor均不作为中央默认值；中央跨度仍需对应形态来源或单独批准的有限假设。

有效权重可写为w_ji=K_j(p_i−a_j)=K_j(F(q_i)−a_j)。这是将OPL接触拉回输入面后的数学关系，中央通常不再是RF平面中的同心各向同性核；不是声称ISET已实现此变换。离散逐coneID求和不额外乘Jacobian；连续密度积分才同时变换密度与面积。HC功能RF由connected cone输入与网络响应确定，不能直接赋为soma位置；输入坐标加权中心只能作为另行命名的几何诊断。

中央pedicle-free区域没有terminal对象，但它不是HC soma排除区，也不直接决定跨区arbor能否接触。必须检查整个A_j覆盖的terminal和来源范围：anchor处密度零而A_j延入有pedicle区域，仍可能有合法边；输入q_i在中心也不意味着其terminal p_i位于空洞。

## 有限contact候选和gap图

先分开定义Dall,t=Et/Ncone,all（每cone平均连接的t型HC数）与Ct=Et/NHC,t（每HC平均接收的cone数）。来源divergence是patch平均及作者推导，不是每个cone必须有相同整数degree。

优先由已给出的morphology或有限footprint建立候选。下面密度半径仅保留为局部近均匀patch的可选proposal，用于比较和预算，不作为中央构造的先决条件。在同一个OPL物理面积定义ρcone,all、ρcone,eligible,t及ρHC,t；有来源D且eligible密度非零时可评议：

Rproposal,t(x)=α sqrt[Dall,t(r)ρcone,all(r)/(πρHC,t(r)ρcone,eligible,t(r))]，α=1.5。

其推导是局部平稳近似下的边数守恒：Ct≈Dall,tρcone,all/ρHC,t；对应候选面积容纳α²倍的期望输入。α=1.5是尚未批准的工程候选值，不是形态测量，也不保证离散匹配必定可行。这个半径以HC anchor为圆心，仅表示有限候选搜索范围，不能称真实圆盘树突场。若密度变化在该尺度上很大或点上密度为零，该近似不适用，应回到完整footprint内的离散terminal查询，不由此宣布中央contact不可能。

H1候选L/M，稀少S连接首版置零是具名简化；H2候选L/M/S。对明确patch/分区，将全cone分母的总边数目标设为Σcone Dall,t，并对eligible cone做平衡整数配额、有限arbor支持域内的稀疏匹配。来源D的偏心度应按来源的组织位置解释，不能无说明改为q_i的视觉偏心度。若H1排除S，eligible平均目标应为Et/NLM，不能仍用Et/Nall造成边数损失。原来源不提供S-specific degree或生理增益，geometry-only分配是近似，S权重另由已确认生理profile提供。

不再同时硬指定一个矛盾的HC convergence；以实际边数反算并报告。每个pair最多一条接触；cone→HC与反馈返回边共享接触拓扑，权重和单位分别定义。边界patch的边数统计必须说明cone-owner、HC halo及裁剪约定。

divergence provider本身仍有限：原文近中心、5–6mm及远端独立patch锚点不能冒充连续同眼实测曲线。中间插值、5–6mm H2区间取值均需具名近似；未选定时只输出候选图，不输出假定完成的生理contact。

零接触或配额不足时依次检查：coneID及q_i→p_i映射；terminal是否落到输入ROI之外；当地HC及其A_j是否已生成；完整支持域是否被计算边界裁剪；形态、层位、subtype和来源域是否覆盖。扩大halo只能找回裁剪掉的允许邻居，不改变arbor长度。

状态须按证据区分：支持域未生成是边界缺失；A_j无来源或未选定是形态假设缺失；所需坐标/区域超出来源是CONTACT_DOMAIN_UNSUPPORTED；完整已支持A_j内无合法terminal则是零候选事实，若与要求的degree冲突再报告配额不可行。不得仅凭ρeligible(a_j)=0返回CONTACT_DOMAIN_UNSUPPORTED，也不得凭空补中心HC、无限nearest、复制点或扩大中央半径来满足配额。已有局部候选成立仍不等于全域中央形态已验证。

HC↔HC只生成homotypic候选邻接，可用Delaunay边再施加显式有限距离门限；门限需单独profile，未给则保持候选状态。耦合电导来自独立动力学来源/已确认配置，必须对称、非负、有单位。几何核不能直接当nS，不能用网络RF长度常数替代单细胞树突半径。该讨论不修改现有HC动力学。

## 边界与后续验收

分别保存输入owner ROI、OPL候选域、canonical生成域、来源支持域及真实组织mask。先将owner cone沿ID映射到terminal，再覆盖这些terminal邻近HC的完整A_j及其所需cone输入；不能把相同数值的输入ROI原样当OPL ROI。mosaic作用halo覆盖最大的pair相互作用支撑；contact halo覆盖有限arbor和位移映射；网络gap传播需另考虑多跳及边界条件。单边有限长度不意味着gap网络总影响严格有限。

生成域若需要跨出0–7.1mm来源支持域，必须提示不支持/需新的来源profile，不能以裁掉halo保留运行名义。真实视网膜边界不补镜像或复制细胞；有限计算域的封闭/固定边界只能明确作为近似。静态空间统计边界校正与日后动力学halo收敛是不同验收，不互相代替。

后续经运行授权拟验收：节点回代、有限正值、单位与坐标、积分配额、低计数行为、seed/ID/tile一致性、H1/H2及联合最近邻与pair统计、非均匀密度梯度和区界伪影、非法接触/配额缺额、halo截断。连接诊断同时按q_i的输入偏心度统计cone HC度数，按p_i和A_j的OPL位置统计接触距离、HC输入数与空洞；加入“anchor密度为零但有限arbor跨入terminal区域”的边界情形，避免恢复错误拒绝逻辑。小样本不得硬报regularity index或置信统计。来源patch拟合复现须在相同窗口/边界规则下单独验证；中央无实测mosaic，不能标为已验证。以上本轮均未运行。

## 已实施的路径与明确缺口

| 文件/入口 | 实际职责 |
|---|---|
| `retina_sim/hc_mosaic.py` | 物理HC总密度与H1比例分开定义；中央log-smoothstep、外围log-total/logit-fraction Hermite；来源记录和近似分开；联合积分配额与有限温度soft-core；canonical实例、稳定ID与同实例crop |
| `configurations/hc_joint_synthetic_v1.json` | 版本化数值、预算、来源域与具名假设；历史Wässle转录的冻结SHA256；H1/H2 footprint均为null，未设中央半径 |
| `retina_sim/hc_arbor.py` | 同coneID的输入q→移位pedicle p→有限HC arbor；disk/ellipse或显式形态节点、有限延伸与层位/区域约束；配额、缺额、A/B共同拓扑和独立归一化；homotypic有限gap候选导出 |
| `retina_sim/hc_geometry.py` + `retina_sim/spatial_profiles.py` | 配置解析、OPL候选域、完整canonical组织域、arbor与pair支撑halo、共享ID局部视图；接入既有source ROI |
| `examples/full_target_network.py` | 实际构建上述几何/接触，再交给既有原生方程；不另设孤立演示，不用effective RF核替代HC |
| `retina_platform/config.py`、`app.py`、`backend.py` | RunConfig和GUI可选“联合HC合成·未跑”、显式H1/H2有限半径或HC几何JSON；默认单动物patch保留；每次运行导出`hc_geometry.json`及manifest中的完整几何审计 |
| `tools/run_full_target.py` | 相同配置的CLI入口：`--hc-geometry-profile`、`--hc-geometry-json`、`--hc-h1-radius-mm`和`--hc-h2-radius-mm`，单位为mm |

canonical上下文覆盖最大的同型/异型pair有限支撑及完整arbor包络；超出7.1mm支持域或工作预算即报告，不裁halo、不降低密度。显式固定canonical域中的多个view共享细胞ID、类型和位置；自动由输入patch解析的canonical域若改变，则形成另一个实例，不能称跨tile同一全眼实例。来源identity检查冻结的是本地历史转录，不把它升级为原页的独立核实。

完整组织mosaic、局部几何view和动态参与view分别保存。原生模型要求实际A/B单位汇聚行：无已生成cone接触的HC保留在完整实例、arbor/缺额审计及generated_active_population中，作为geometry-only节点，不分配动态状态；HC gap候选未接入动力学，因此这些节点在本有限图中不会影响cone。此操作不降低解剖实例密度、不重采样，不声明这些HC在真实组织中没有输入或响应。若cone没有有限反馈接触，当前原生装配明确报告HC_CONTACT_RUNTIME_UNCOVERED_CONES，不补nearest，也不加隐式背景反馈。既有HC动力学源码、参数及RGC动力学未因本次几何修改而改变。

footprint缺省仍为HC_ARBOR_UNCONFIGURED。GUI输入的半径只建立具名project_approximation；有形态来源时可提交disk/ellipse或morphology_nodes JSON，仍须遵守来源域。新geometry若声明measurement/published_fit，必须同时提供primary_verification：verified、material_path、material_sha256、locator、condition、geometry_sha256及radial_support_mm。geometry_sha256冻结该条记录除source_profile_id/evidence_class/approximation_id/source_locator/primary_verification与跨动物、跨物种、meridian转移元数据外的物理描述，以排序、无额外空白的UTF-8 JSON计算SHA256。集成加载器检查实际材料字节与该描述，拒绝只填来源名称的科学标签；身份绑定仍不等于生理模型验证。divergence未配置时标记CANDIDATE_GRAPH_UNCALIBRATED，归一化接触不是实测degree或nS。gap距离未配置时保持GAP_DISTANCE_UNCONFIGURED；即使配置也只导出候选，conductance为UNBOUND，connected_to_dynamics=false。

配置无需重写密度或生成器。GUI选择联合HC模式后可直接填写H1/H2两个显式mm半径；复杂形态使用HC几何JSON，顶层允许`profile_id`、`canonical_bounds_mm`、`footprints`、`generator`、`divergence_profile`、`require_owner_quota`、`gap_candidate_max_distance_mm`。`footprints`按H1/H2配置，可有`by_hc_id`稳定ID覆盖。省略canonical域时由OPL输入与必要halo自动解析；生成器已有完整具名默认与预算。

| footprint字段 | 精确接口 |
|---|---|
| 通用必需项 | `kind`、`source_profile_id`、`evidence_class`；project配置还须`approximation_id`。单位`unit=mm`，坐标`coordinate_frame=OPL`；`radial_support_mm=[下界,上界]`为来源限制，科学标签必需 |
| `kind=disk` | `radius_mm`为正有限数；这是显式有效footprint，默认未配置 |
| `kind=ellipse` | `axes_mm=[半轴x,半轴y]`为两个正有限数，`angle_rad`为有限旋转角，默认0 |
| `kind=morphology_nodes` | `nodes_mm`为N×2或N×3节点；`node_coordinates=relative_to_anchor`或`absolute_OPL`；`contact_threshold_mm`为正有限值。绝对节点另须`envelope_mm`，xyz节点另须`z_tolerance_mm` |
| 有限延伸（可选） | `allowed_extension`为disk/ellipse有限域，另给正`max_extension_mm`及`extension_policy=nearest_branch_to_pedicle_in_finite_domain`；不生成无限枝条 |
| 接触约束（可选） | `min_contact_spacing_mm`非负；`kernel.kind=uniform`或`gaussian`加正`scale_mm`；`z_interval_mm`、`node_regions`、`allowed_pre_regions`/`allowed_post_regions`须有对应节点/terminal元数据，缺失明确拒绝 |

当前完整网络的McGregor pedicle输入是二维；它能直接接disk/ellipse及二维形态节点。低层arbor接口接受真实xyz/region数据，但完整网络尚无真实cone terminal深度及分支区室标签输入，配置依赖这些数据的z/region约束会报告缺失，不凭空补零或把导航区域称为形态区室。这个限制不阻止已有二维有限接触路径交付。

已写30项待执行的密度/mosaic、arbor和集成回归，覆盖局部/全域配额、0/1计数、crop身份、坐标/单位、中央anchor无terminal但有限arbor可接触、缺额、输入/反馈共同拓扑、GUI配置往返、canonical上下文与动态参与选择。79个项目Python文件AST语法、相关JSON、构建/接触调用参数及既有冻结转录identity检查通过；没有执行产品导入、测试或数值生成，不声称回归通过、mosaic混合收敛、局部统计已验证、中央全层配置已跑或生理标定完成。

未知科学参数继续保留：中央真实arbor/层深、S/H2独立动力学、全型别BC/AC/RGC内层参数与绝对光学/release标定。ISET/Macular功能核仍是独立参考，没有叠加；停止的DeepGaze/SC、研究比较、长适应及独立动物验证没有恢复。

## 授权后的实际定向测试

30项HC回归全部通过；3项必要旧source-patch回归在修复regional导出读取legacy Profile字段错误后通过。新joint profile经过真实GUI配置读取和后台构建/导出，完成0.2ms、39组件、20层、1窗口；76664个数值有限。canonical H1/H2=659/236，逐分区积分配额、稳定ID/位置、cone q/p身份和共同反馈拓扑校验通过；原生HC源码参数SOURCE_VALUES保持不变。工程fixture采用显式H1 disk和H2 ellipse，未修改null生理arbor默认。

此次仅2个有限MCMC sweep，接受率约.5721，mixing_certified=false，不能解释为局部统计、中央形态或生理分布验证。可见窗口点击、长期适应、0–7.1mm全域生成及研究对照未测。完成run在测试脚本更正后经同assembly hash确定性重建复核，没有延长模拟。精确配置、数值、日志、版本hash和完成run路径见[HC定向验收](../reports/hc_acceptance/latest.json)。

中央补充已接入可选 `central_OPL_equal_area_W2000`，其来源/公式与实际适用范围见 [profile](../configurations/hc_arbor_central_W2000_v1.json)。H1/H2 .00804/.01222mm 等面积disk均为具名project support；实际完整OPL arbor/anchor/soma/pedicle须在.5..1mm，不向内延，不强制逐细胞C/D、不改动力学、不替换默认。GUI/RunConfig与CLI的 `hc_arbor_profile` 接到同一个解析器；外部JSON可选named profile，与手填足迹/半径冲突拒绝。15项配置/小接触fixture通过。中央AII→midget零边随后定位到RF/IPL生成域混用并修正，真实小构建已有1254条FMB和315条DB1 glycine边，原密度和.04mm硬支持保留；合法完整有限域的空和算子4项回归通过。进一步修复AII的上游rod/RBC生成halo和三类AC的RF→IPL点接入后，471个RBC各40条rod输入、675个AII各3条RBC输入及三类AC既定BC输入数全部满足。中央仍有9处RGC路径的有限halo/typed IPL点形态冲突，未补假边或扩大连接半径，未进行中央积分；新中央导航preset不等于整网已通过。背景流程的真实小预算端到端在已有周边source ROI完成；统一结论、逐路径坐标数值和证据入口记录于 [core_design](core_design.md)。
