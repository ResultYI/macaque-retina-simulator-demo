# 冻结 atlas 分块存储与接入说明

新增模块 `retina_sim/atlas_chunks.py`，schema 为 `frozen_atlas_spatial_chunks_v1`。
它只改变既有固定 master 的存储布局，不运行 joint/equal-mass 生成器，不按块分配配额，不改变密度、HC 类型/形态、坐标或细胞身份。
原 JSON 全部保留，继续充当完整来源快照与备份。未修改运行、背景初始化、checkpoint、GUI、CLI 或共享配置。

## 文件与身份

每个人口一个独立目录，位于 `data/atlas/spatial_chunks_v1/`。

- `manifest.json`：原 JSON 的完整 SHA256、完整原表 `table_hash`、atlas/source/population/reference 身份、坐标类型与单位、固定域、块宽、空间索引及所有块/元数据哈希。
- `metadata.json`：原 recipe、密度参数、来源、生成参数、诊断、claims、provenance 和 boundary；按需加载。
- `block_XXXXXXX.npz`：原 master 的行号、source keys、global cell IDs、cell types，以及每种已提供坐标的 float64 数组。文件不含 pickle。

默认块宽 0.1 mm，网格以全局原点为基准，包含负坐标。索引先定位网格块，再以块内实际点范围剔除无交集块。
精确半开查询遵循原 `AtlasROI` 的 `[x0,x1) × [y0,y1)` 规则；最终按原 master 行号排序。存储块不能成为生成域或 owner 配额域。

`source_json_sha256` 是整个原 master 的逻辑身份，`full_table_hash` 绑定全部原始细胞表内容。`storage_manifest_hash` 另外绑定分块布局、全部块哈希和元数据哈希。
HC 的逐行 `source.types` 从元数据移入块内；查询通过 `cell_types` 返回原 H1/H2 标记，其他来源字段不变。equal-mass JSON 没有独立 subtype 数组，其类型标记采用原 population 名。

转换使用新目录且拒绝覆盖；manifest 最后写入，未完成的目录不能作为有效存储。每次读取一个块都核对其 SHA256；只读取 manifest 不代表所有块已经验证。
旧 JSON 不会在运行查询时打开。一次性转换目前仍全量解析原 JSON/table，**未实现百万行 JSON 流式转换**。

## 接入 API

```python
from retina_sim.atlas import AtlasROI
from retina_sim.atlas_chunks import ChunkedAtlasReader

# 由工程内部绑定提供目录和两个冻结哈希，具体值见验证报告 local_masters。
reader = ChunkedAtlasReader(
    binding["path"],
    expected_source_sha256=binding["source_json_sha256"],
    expected_manifest_hash=binding["storage_manifest_hash"],
)
roi = AtlasROI(
    (.34, .36, -.01, .01), (.30, .40, -.05, .05),
    reader.reference_id, coordinate_role=reader.index_role,
)
view = reader.query(roi, max_cells=60000)
table = view.table
ids = view.global_cell_ids
types = view.cell_types
metadata = reader.metadata_record()  # 需要来源/密度信息时再读取
```

- equal-mass 索引角色为 `rf`；HC 索引角色为 `soma`。所有原有坐标字段均保留，但查询必须使用索引角色，禁止隐式坐标回退。
- `query()` 返回 `ChunkedSelection`：裁剪后的 `CanonicalCellTable`、原 master 行号、types、owner mask/status、ROI 和 I/O 统计。**`row_indices` 是原完整 master 的行号，不能再用它索引已经裁剪的 `view.table`。**
- `crop(roi, max_cells)` 返回裁剪表，适用于 equal-mass 调用模式；HC 原接口返回表、types、metadata，接入方应从 `query()` 的结果组装原接口，而不是丢失 H1/H2 标记。
- 默认保留 HC 原 0.14050220147139075 mm pair-context 检查，越出固定域则失败。`require_pair_context=False` 仅用于原始存储行/身份查询，不能据此宣称 joint 相互作用边界完整。
- 查询预算限制实际返回数量，超预算报错，不稀释人口。此接口没有完整 master `.table` 常驻缓存，也没有完整类型数组常驻缓存。
- 存储器只覆盖细胞表；现有连接图、ODE 状态、背景及 checkpoint 的内存不包含在查询内存统计中，运行集成由主任务处理。

转换函数为 `convert_frozen_master(source_json, destination, expected_source_sha256=..., block_size_mm=.1)`。
支持实际存在的 equal-mass master JSON 和单人口 `canonical_cell_asset_v1`，不接受任意多人口图作为一个 master。
参考读取函数 `read_frozen_master()` 只供离线转换/验证，不应被按需运行路径调用。

## 一次验证与实测

工具 `tools/convert_atlas_chunks.py` 已执行一个验证批次，结果见 `reports/atlas_chunks_validation_20261012.json`。
六个既有局部 master 共 16,483 个细胞。全表、两次重叠查询和穿过一个实际细胞坐标的左右半开边界查询均通过。
比较了 global IDs、source keys、H1/H2 或人口类型、原行顺序、owner 标志及坐标 float64 字节；完整表 hash 和非逐行来源元数据相同。
越出 master 的查询被拒绝；HC 缺少 pair context 的完整域查询被拒绝。六个原 JSON 的 SHA256 未改变。

| 人口 | 细胞数 | 块数 | 原 JSON 字节 | 分块副本字节 | 重叠查询返回/解码行 | 读取块数 | 查询峰值 traced 字节 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AII_AC | 310 | 8 | 47,805 | 39,645 | 49 / 155 | 4 | 185,226 |
| parasol OFF | 249 | 6 | 39,410 | 32,597 | 43 / 185 | 4 | 200,499 |
| parasol ON | 249 | 6 | 39,399 | 32,590 | 39 / 185 | 4 | 198,565 |
| rod_BC | 148 | 20 | 27,085 | 48,789 | 23 / 58 | 4 | 126,553 |
| rods | 4,208 | 24 | 626,374 | 351,246 | 670 / 1,883 | 8 | 1,661,427 |
| HC | 11,319 | 104 | 3,227,930 | 1,982,646 | 705 / 1,704 | 24 | 1,599,708 |

原 JSON 合计 **4,008,003 字节**，分块副本合计 **2,487,513 字节**；保留两份后的总量为 **6,495,516 字节**。
小且稀疏的 RBC master 因 ZIP/索引开销增加体积，不能宣称每个人口都压缩。
查询内存是 reader 初始化完成后、单个 overlap_A 查询的 `tracemalloc` 峰值，**不是进程 RSS**，不含完整原 JSON、转换、其他读入表、manifest 初始化或连接图。
各查询的耗时、压缩字节读取量、解码数组量、最大块等实测值都保存在 JSON 报告；仅有一次局部验证，未做全域性能测试。

## RF 0–3 生成与存储预算

以下为**估计**：在各字段声明的平面上，对半径 3 mm 圆盘做 64×64 极坐标 Gauss–Legendre 密度积分；AII 常量密度使用精确圆盘面积。
这不是最终整数配额、实际生成域或生理人口计数。未测积分误差界；实际 guard、Henle/IPL 映射、joint 边界上下文和 typed 内层人口会增加或改变预算。

| 人口 | 圆盘密度计数估计 | 既有生成器核心内存估计 | 分块磁盘估计（局部总字节/细胞外推） |
| --- | ---: | ---: | ---: |
| cones | 548,986 | 尚未确定全域生成器，不能仅从密度授权生成 | 未实测同格式 master；XY 下限 8,783,776 字节 |
| rods | 2,768,569 | 2,038,250,544 字节 | 231,095,245 字节 |
| rod_BC | 153,809 | 113,333,664 字节 | 50,703,969 字节 |
| AII_AC | 141,372 | 104,139,392 字节 | 18,079,655 字节 |
| parasol ON | 30,726 | 22,630,976 字节 | 4,021,528 字节 |
| parasol OFF | 30,726 | 22,630,976 字节 | 4,022,392 字节 |
| HC | 322,307 | 128 字节/细胞约 41,255,296 字节；joint 工作空间未量测 | 56,455,578 字节 |

这七类合计约 **3,996,495** 个细胞；不包括其他 BC/RGC/AC/WAC、位移 guard、连接边、动态状态、背景、checkpoint 或原 JSON 备份。
六类已转换格式的磁盘粗估合计 **364,378,367 字节**，不含 cones 和其他内层人口。稀疏局部 RBC 比例开销大，外推偏粗；压缩率、全域字符串长度和索引开销会改变该值，不能作为磁盘保证。
0.1 mm 网格覆盖半径 3 圆盘的包围方形，每个人口索引块数保守工程估计为 3,844，实际只写非空块。

equal-mass 内存公式沿用现有 `side=max(64,ceil(sqrt(8N)))`、`side²×80 + N×96`。rod 的约 1.90 GiB 核心估计已超过现有 512 MiB 生成器上限；rod/RBC/AII 的估计数量也超过现有 60,000 人口预算。
**分块存储不会解除生成器的全域耦合或内存上限，不能用分块独立生成绕过。**
HC 两轮 sweep 提议约 644,614，超过现有 500,000 提议预算；按 target mass 8 的规模约 40,288 个分区，也超过现有 8,192 分区设置。上述为工作量尺度估算，不代表已生成或 MCMC 已混合。
全域生成时间未测量，不能按局部定时线性承诺。未来大域 JSON 的一次性转换仍需全量内存预算，或先设计保留全域身份的流式导出器。

## Source 与物理域边界

- Packer rod 首节点分别为 nasal 0.402268、superior 0.430108、temporal 0.403018、inferior 0.418160 mm；外围末节点约 6.260–6.290 mm。首节点以内只使用已经批准的二次补全；原数据是作者地图数字化，角向混合与名义 RF 注册仍为项目转移。
- Packer cone 四向首节点约 0.048917–0.065056 mm、末节点约 6.231638–6.253005 mm。中央沿用已有 published-code bridge；不增加新的 L/M/S 证据或源外人口。
- parasol 模型应用半径严格为 **0–3 mm**。圆盘外的 guard 未获支持，`[-3,3]²` 的角点也超界，不能用 clamp 或外推补齐。
- RBC 使用已有 rod/18 effective 转移；AII 使用已有 5000/mm² physical 转移。两者都不是全域独立实测人口曲线，实际物理域必须从支持的目标及连接半径推导。
- HC 密度项目场应用半径 0–7.1 mm，不等于直接测量的完整场。现有 joint master 固定域仍为 `[-.4,.9]×[-.4,.4]` mm；pair context 为 0.14050220147139075 mm。直接中央/周边 arbor 曲线仍缺失，本任务未改变 morphology、density 或 joint 生成参数。
- Henle 原数字化曲线与已经批准的 `Schein1988_M1_bounded_bridge_v1` 保持分离。该 bridge 应用为 `0<r<=6 mm`，原点方向未定义，mapped radial limit 为 0.15 mm；中央/tail/outer identity 仍为有界项目桥接。应先从 RF 域映射实际 OPL/BC IPL 目标，再确定 HC/rod/RBC/AII 域。
- S 路由保持已有 species/direction/effective 约束；缺失的全域曲线不能由存储器填补。其他 midget/diffuse/S/AC/WAC 人口由实际 cone parents 和 typed sites 推导，本次不提供虚构的独立全域密度。
- 固定域和有限 pair/连接支持并不证明 recurrent 外部闭合或完整生理验证。源数据节点范围、application 域和 query 支持分别保留。

## Master 版本与背景 cache

对同一冻结 JSON 重新分块，仅 storage layout hash 改变，原 source SHA、完整表 hash、cell IDs 和坐标应不变。
新生成更大域会改变 canonical bounds、recipe/canonical instance、source namespace 和 master 版本；即使 seed 相同也不能拼接旧域身份或复用旧局部配额。
背景/checkpoint cache 必须绑定原 master 身份、实际细胞 ID/坐标、连接图与动力学/初始化配置。新大域 master 应使旧背景 cache 失效，禁止把局部缓存标成全域背景。
工程集成继续使用一套共享配置；本模块不添加用户版本菜单或新的生理模型。
