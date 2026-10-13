# Chen2024 瞬时 50k 门禁窄修正报告

2026-10-10，按已批准 Sentinel_68036fb39d9481919461d1837cc67cdf 完成本轮修正并停止。显式作者来源复现模式允许非负有限的正值高峰，保留高光生理适用性警告；默认工程封套与完整模式阻塞保留。没有处理真实负值、运行真实模型对照或调整参数。

## 依据与改动

固定作者 commit 为 `1bcf7c9a6a6814ef39d2c39d82cf8dd1b5b39db2`。原 BiophysModel.m 无瞬时50k上限、裁剪或整流。已重新核对[官方论文](https://cdn.elifesciences.org/articles/93795/elife-93795-v1.pdf) Fig.3 p7 的80k刺激轴、Fig.4 p8的primate cone 100k刺激轴及 Discussion p15 Photopigment bleaching。轴上限不当作已测峰值。正文的 <50k 讨论不含漂白/再生且限制曝光的适用性，不能作为所有作者复现输入的逐样本拒绝条件，也未规定可机械替代的全段均值上限。

|模式|选择方式|正值 ≥50k R*/s|
|---|---|---|
|默认 component_reference|source_reproduction=False|仍按 strict_project_reference 工程封套拒绝|
|显式作者来源复现|component_reference 且 source_reproduction=True|允许计算，记录警告与适用性未建立|
|numeric_fixture|不可选择作者策略|仍拒绝|
|full_physiology|不可使用该孤立组件|仍阻塞|

负值、NaN/Inf、单位、形状、固定dt、scope、solver、状态和下游门禁不变。stream 的 rate_segment 须显式选择同一输入策略；跨策略 segment/state/checkpoint 拒绝。实际实现与调用细节见 [合同](chen2024_implementation_contract_zh.md)。

警告同时以 Python HighLightApplicabilityWarning 与结构化 metadata 输出：实际消费区间的逐细胞最小值/峰值/描述性均值（R*/s）、dt区间数、持续时间（s）、累计曝光（R*）、≥50k点数、论文及commit定位。不以均值增加新门禁，包含背景，只统计实际消费区间；批量最后未消费样本另记提供范围。曝光前史、实际漂白与物理照明条件不能从这些数值推断，生理适用性仍为 NOT_ESTABLISHED。账本中的旧上界保留为默认保守工程封套，未改账本参数来伪装来源。

## 参数、动力学与数据保持

修改前快照见 reports/chen2024_gate_correction_before/；[逐项指纹比较](../reports/chen2024_gate_correction/fingerprints_comparison.json) 及 [模块差异](../reports/chen2024_gate_correction/chen2024.diff.patch) 可复查。R/P/G/C/S更新式、initial_conditions、initialize、unpack、暗态派生块指纹相同；12项作者数值、参数profile、state-spec、graph、engine、contracts、来源账本与作者两文件字节相同。

整个模块/策略 model_hash 有意改变；旧及跨策略 checkpoint 不能静默续跑。这个身份变化不等于生理方程变化。

ZIP SHA256 `c7e39748f11cf2903167968b610f360844ce4a91254d14823b8953572a878de6`；MAT SHA256 `6724e2edd8e8708456f32900a14efae88192a5ba597365046d0031eb543d1331`；README SHA256 `f25cb72716327f5fc08cfe26cc423bc3364f8f331900dd1972eeb664c1911d03` 均与修改前一致。[真实曲线及统计保持证据](../reports/chen2024_gate_correction/real_data_preservation.json) 确认原NPZ、全部6细胞统计CSV/JSON、历史图PNG/PDF未重做未改写。图中旧50k标线现在仅解释为默认工程封套，不能用于断言作者数值域。

## 实际验证与独立复核

最新实际执行 `reports/20261010T133335020336Z`：A54、B8、C45全部通过，AST解析无错误。见 [verification.json](../reports/20261010T133335020336Z/verification.json)。C原36项保留，新增9项覆盖高峰警告、模式与策略隔离、负值/非有限拒绝、状态身份、逐块曝光及修改前后指纹/轨迹。容差没有放宽。

合成暗态、背景、阶跃的修正前后 R/P/G/C/S/电流/时间逐点一致，最大误差0；正值100k高峰 synthetic_fixture 与独立源式 scalar计算全部状态最大误差0。该高峰检查不是新的真实动物记录或高峰MATLAB执行。五个原 synthetic_fixture 与先前真实执行的原MATLAB输出再次比对，最大电流误差7.389644451905042e-13 pA；分块/checkpoint误差0。数值结果不认证生理符合度。

首次本轮聚合 reports/20261010T132921555035Z 留档：A53通过/1失败，B8/C45通过；失败为覆盖报告 P01.status=待审计 不在合同五种状态中。只修复生成器映射为局部实现、保留原测试后重跑通过，不删除失败记录。

限定独立代理复核已核查更新式、指纹、模式门禁、负值及高光警告，未发现确定的动力学或绕过问题，且先行指出上述报告状态回归。复核说明见 [独立复核记录](chen2024_gate_correction_review_zh.md)。

## 剩余阻塞与停止点

真实 PrimateConeResponses 6条刺激各有1026个负值（最低-767.7310526189535 R*/s），已超出舍入噪声；没有来源支持的恢复/裁剪/加背景规则。预刺激历史及实验/内核相位依据仍缺。≥50k 的12538点仅作描述统计，不再单独阻断显式作者复现。缺逐cell暗电流阻止恢复真实pA，但不是固定模型无量纲归一化比较的单独阻塞理由。

全部6条 model_ran=false，误差指标和noise ceiling为空，没有参数拟合、gain调整、裁剪、时间平移或初态补造。此包属于作者定参来源，不是独立验证。完整主线仍为20项需求、15项阻塞；中心/过渡/S cone、下游回路和独立生理验证未完成。本轮只完成窄门禁修正，到此停止。
