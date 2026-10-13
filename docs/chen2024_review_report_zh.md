# 阶段 C 独立限定审查

独立审查代理 `/root/contract_review` 读取发布后的真实代码、验收记录、MATLAB 日志和 MATLAB 导出的原始数值结果。审查限于已批准的 Chen2024 外周 L/M 外段电流模块，不增加生理参数、模型范围或数据实验。

初次审查发现一个确定问题：`rate_segment` 使用 `round` 选择刺激样本，非网格时刻表现为最近采样而不是声明的零阶保持。输入 [10,20]、dt=0.0001 s 时，t=0.000075 s 原返回 20，应返回 10。离散推进仅查询网格左端，因此此前 MATLAB parity 不受影响；此前测试没有覆盖非网格接口查询。

修复为预计算绝对区间边界，并用 `searchsorted(side='right')-1` 选择半开区间。新增回归覆盖区间 75% 处、精确边界、nextafter 左极限、origin=0/1.3 和区间外拒绝。审查代理随后独立读取发布文件并实际执行这些查询，确认唯一发现已修复。本次限定审查未发现剩余确定缺陷。

其余指定项核对通过：rate×dt 仅一次；R/P/C/G 使用旧步状态；Snew 使用 Cnew，Gnew 使用 Sold；暗态重算及负 pA 符号正确；参数、单位、>20° L/M 与三 Macaca 混合来源限制明确；S 缓存跨 chunk/checkpoint 保存；拒绝 adaptive、linear/inverse 和 pA→mV。

审查确认五组原 MATLAB 电流对照全部标为 `synthetic_fixture`，最大数值误差约 7.39e-13 pA。真实 Dryad MAT 尚未取得，inventory 为 BLOCKED_DOWNLOAD；来源数据复现和独立生理验证均未运行。

最终完整 A/B/C 聚合结果由主代理另行保存在 `reports/verification.json` 与不可覆盖的时间戳目录；此审查结论只支持有限范围的代码、接口和源码数值一致性，不认证生理符合度。
