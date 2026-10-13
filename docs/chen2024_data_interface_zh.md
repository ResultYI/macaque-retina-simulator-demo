# Chen2024 / Dryad 只读数据接口

`retina_sim.chen2024_data.ChenPublishedData` 复用现有导入器的冻结文件大小/SHA256，以及既有 MAT 结构清单；不重新下载或建立另一份数据管线。加载器只读原文件，按需读取一个 primate MAT 变量，并返回不可写的原样配对切片。未接入 GUI、运行配置或模型刺激驱动。

```python
from pathlib import Path
from retina_sim.chen2024_data import ChenPublishedData

data = ChenPublishedData(Path(r"D:\macaque_retina_simulator"))
rod_catalog = data.catalog("PrimateRodResponses")
record = data.read_record(
    "PrimateRodResponses", 1, start_sample=0, stop_sample=256
)
# record.sample_index_zero_based：原记录的零基索引
# record.relative_index_seconds：索引 × README dt；不是实验起始时间
# record.stimulus_Rstar_per_s：原 AdaptingStepStim
# record.response_normalized：同一 struct 的原 AdaptingStepResponse
# record.metadata：资产身份、单位、重复平均和未解的比较/标定缺口
```

记录序号从 1 开始，切片停止索引不包含在内；数组序号不代表生物学 cell ID。`catalog()` 不解析 MAT 数组；结构、范围及数值哈希来自已经绑定当前 MAT 的旧清单。首次 `read_record()` 会解析并缓存该变量的六条记录，后续切片复用它；这是只读加载接口，尚非 MAT 随机磁盘切片。

| 发布变量 | 记录与样本数 | README 采样间隔 | 响应与重复 |
| --- | --- | --- | --- |
| PrimateConeResponses | 6 × 80,000 | 0.1 ms | 有符号电流 / 饱和闪光估计暗电流，无量纲；4–5 次重复平均 |
| PrimateRodResponses | 6 × 123,060 | 1 ms | 同上；具体逐记录重复数和原始 trial 缺失 |

刺激单位均按原 README 保留为 R*/s；不重新解释为 contrast。rod 名义索引轴为 0–123.059 s，间隔总长 123.060 s。README 依次描述 step+flash、dark flash 和均值每 4 s 改变的 noise，但 MAT 未标这些段落的边界。rod 第 1 条刺激与其余五条的数值哈希不同，不能共享一条刺激替代所有记录。

现用 rod 参数继续通过已有 `retina_sim.rod_pathway.rod_source_values(root)` 读取作者 commit `1bcf7c9a6a6814ef39d2c39d82cf8dd1b5b39db2` 的 `primateRod` 分支；来源校验和更新顺序未改。其作者步长为 0.1 ms，输出是模型 pA。由 `k × gdark³` 得到的 37.23875 pA 是模型参数推导值，不是公开记录各自的实测暗电流，不能拿它补填发布数据的绝对电流分母。

## 尚不能进行的标定或重放

- 全部 rod 原刺激中存在负的名义 R*/s（最小 −1080.6756），语义未解；不裁剪、加 offset 或改作 contrast。前 256 个零刺激样本通过解析核验，不解除整条记录的输入缺口。
- 没有实验 time/onset、phase、预刺激历史或导出预处理说明；1 ms 数据与 0.1 ms 作者更新的 hold/插值/对齐规则也未确定。
- 缺逐记录实测暗电流、原始重复、明确 cell ID/species/eccentricity。绝对 pA 标定受缺分母阻塞；缺分母本身不禁止将来有依据的无量纲比较，但其余输入/历史缺口仍阻止当前重放。
- rod README 的 voltage-clamp 表述与云侧论文方法复核的 suction electrodes >20° 存在冲突，未确定逐记录映射。接口同时保留两种表述，不补猜。

`record.require_model_comparison_ready()` 当前会明确拒绝重放。数据是作者参数化来源，今后成功比较也不构成独立外部生理验证。

## 本轮核验与后续工程范围

只运行一次 `tools/check_chen_published_loader.py`：rod 第 1 条前 256 对样本、索引轴及只读属性与原 MAT 一致；六条记录的结构复用既有清单。结果见 `reports/chen2024_readonly_loader_check_20261012.json`（PASS）。两份当前配置和 cone/rod 实现哈希保持不变；模型、背景、拟合及重采样运行数均为零。

随后已获批完成 `atlas_runtime.py` 的整段正弦输入内存修正：保持原 100 µs 网格及线性插值，只按需缓存相邻两行；未接入 Chen 实验刺激。一次最小核验 351 次旧新值及索引比较完全相同，14 个背景平移/时间边界检查通过。1,465 个细胞在 11/1,100 s 声明时长下通道分配峰值约 88 KB（分配追踪，不是整个模型 RSS），缓存恒为两行。见 `reports/native_light_input_check_20261012.json`；没有重跑背景或模型。

补充只读 rod 清单见 `reports/chen2024_rod_input_metadata.json`：每条只有两个 1×123,060 float64 字段，未含 darkCurrent/offset/time/prePts/epoch/segment 或嵌套附加元数据。第 1 条 7,635 个负点、第 2–6 条各 7,155 个；第 2–6 条刺激逐值相同，第 1 条不同。首负值在零基索引 49,800（MATLAB 49,801），约 −3×10⁻¹⁵；索引 49,800–63,058 内共有 6,600 个负点、5,526 个细碎连续段，该包络含间隙，不能当作整段负刺激。JSON 提供这一早期包络/段计数及所有后续精确负值段、首负值前后各 10 点；原 MAT 保留全部细碎点，不依据 demo 的 prePts 补实验预历史。

全域 0–3 mm 尚未生成或接入。当前运行仍限定冻结 r≈0.35 mm 的 owner ROI `(.34,.36,-.01,.01)` 与 0.04 mm halo；已有 graph 分块仅改变这份局部图的存储。更大域仍需全域身份/预算/生成器规划和源支持边界，不能独立分块生成后拼接。现有数量与内存估算、parasol 3 mm 边界和 HC 中央缺口见 `atlas_chunks_integration_zh.md`；缺失的测量不由加载器或存储器补齐。
