# 阶段 A/B 独立实现复核

2026-10-10。独立 reviewer `/root/contract_review` 只读检查实际 stage 源码、固定源文件和首轮日志，没有写文件、运行旧实验或新增科学实验。复核范围限定于已确认 A/B 合同。

发现并内存复现一个确定 bug：原参数门禁仅拒绝空单位；把 `SF2025_tau.unit` 从 `s` 改成非空的 `mm` 后，参数加载曾接受，参考入口直接数值化而未核单位。首轮 A 38/B 8 的测试没有覆盖这个情况。

修复新增 `retina_sim/parameter_contracts.py`，为实际使用的九项作者参数规定接口单位和 shape；来源加载器和参考内核入口均严格核对。增加四项非空错误单位回归，每项同时检查加载器及绕过加载器的模型入口。

独立 reviewer 对修复再次只读复核，实际内存复现结果：

- `tau: s → mm`：加载器抛出 `GateError: Parameter interface unit/shape mismatch`。
- 绕过加载器构造错误 ParameterSet：参考入口抛出 `ContractError: Author parameter unit/shape mismatch`。

本轮修复复核未发现剩余确定缺陷。这是限定检查的结论，不能推广成没有其他未知缺陷。

其他指定核对项符合合同：W_hc 是原始 a.T 独立归一化并保留 S；W_ch/W_on 先排除 S 再归一化；六块状态使用联合 RHS，cone drive、feedback、ON/OFF 符号及 gRest 初态正确；光积分保留有效面积、mm²/nm 换算并拒绝重复效率；chunk 保持绝对时间与状态，checkpoint 核对六项身份／hash；full 模式、缺校准、未知 profile、unsupported adaptation、horExp=true 均拒绝，无默认补数。

同时补齐工程契约：StateDerivativeVector 对混合状态单位逐块核验，支持显式 dimensionless activity 与 mV generator；支持域拒绝空对象、坏类型、非有限边界、缺单位和错误开闭语义；require_runnable 在执行前携带全部缺口抛出 GateError。这些是接口及门禁修复，没有加入新生理机制。

修复后的真实重跑日志在 `reports/20261010T111806851135Z/`：A 54、B 8 通过，提取核对及聚合退出码均 0。首轮日志仍保留在 `reports/20261010T110622396136Z/`，没有覆盖或隐瞒审查发现。没有发生 pytest 失败后放宽容差的过程；bug 来自独立负例复现并通过新增回归解决。

作者 MAT 未解码、上游实测未独立核完、完整视网膜及生理验证的缺口均是明确阻断，不因参考数值验收通过而解除。
