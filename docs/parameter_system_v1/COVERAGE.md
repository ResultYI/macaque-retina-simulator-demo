# 覆盖矩阵

|层/机制|定义与解析来源|当前执行语义/限制|
|---|---|---|
|运行输入/光学/单位|RunContext + 实际input_reference/native/rod帧合同、原外部端口|Td与R*/s分开；unknown物理标定不补值，未进行视频/光学重建|
|Cone L/M/S与多室|全部旧effective_physical_cone/terminal profile、实际component参数矩阵、native SOURCE_VALUES|不新增光谱/亚型差异；当前group保持原native H1模型，缺fine subtype/owner表不伪造|
|Rod|Chen source-native参数、暗背景/膜/release、原snapshot派生参数|源单位不伪装SI；既有R*/s合同不变|
|HC H1/H2|SOURCE_VALUES、HC profile、原Cone–HC CSR/反馈与延迟|H1照旧；H2当前机制not_implemented，不制造参数|
|BC midget/diffuse/S-ON|实际kernel/bias/膜/native gain/release/adaptation/基线参数|独立类型数组，当前kernel不改变；无新偏心度曲线|
|RBC|TRPM1与原contact参数、膜/池/resource/事件率profile|contact component维数与population不同，保留原向量，不按cell数再乘|
|AII|原AMPA/glycine contact参数、膜/resource、gap Laplacian|CSR和S单位保留；不添加AII直达RGC|
|普通AC V+s|实际4功能池及原legacy S-ON路径，C/gL/reversal/gate/G_BA/G_AR|每effective pool的C/gL不再乘N；owner沿冻结拓扑，非已验证生物亚型|
|RGC五类|当前B5/各类型recipe实际采用参数、原膜/反转/history/路线|原数值不变，类型分别登记但不冒充各类型独立生理校准|
|Cell6实验输出|现有冻结model/core/源码SHA、训练标准化、条件CalibrationRecord|显式opt-in，INCONCLUSIVE不升级；原GUI和数学读出复用|
|空间/route/edge/contact|既有group/CSR/坐标/成员/contact与gap身份|只解析当前ROI，不重建连接、不crop归一化；几何证据不升级|
|数值/输出设置|原protocol/clock/recording/phase和原Runner|FP64与100us/1ms时钟保留；本版不改求解器、精度、观察时序|
|动态状态|StateBundle及既有完整checkpoint字段|与静态theta分离；history/RNG/delay/控制器不丢弃|
|校准来源|不可变CalibrationRecord与旧来源catalog|proposed/diagnostic/conditional/validated_for_scope分离，不运行优化|

当前局部交付快照：22个population、1697个group节点、34条route；
335个解析FP64数组、563条参数记录。额外gap registry引用现有全局局部gap CSR。
支持的矩阵/不等长数组是结构能力，不表示已给缺失机制赋值或拟合。

新旧参数、初始化、RHS、全部端口逐字段零差异；CSR未重建，路线身份由原稀疏数据/索引/indptr登记。
细粒度原Builder保留实际cell索引；群体group没有展开复制为独立细胞。

尚未具备而明确保留的限制：H2 native机制、当前group源fine表缺失时的L/M/S/owner精细映射、
未给SI转换的source-native量、缺独立类型/偏心度校准、Cell6物理输入/解剖位置与条件资格。
这些不是架构迁移自动解除的资格。没有新增0–6°全范围测试或生理数值。
