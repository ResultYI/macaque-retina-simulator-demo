# Cell6 实验动态放电输出

原生产/既有读出仍为每次启动默认。新增选项仅为冻结的 temporal_readout_v1
条件性候选，资格仍为 **INCONCLUSIVE**；FIT_READY=false、原 absolute monitor=false 不变。

## 软件操作

1. 双击项目根目录 run_retina_gui.cmd。
2. 保持 cell6_g3，下方 RGC 选择 parasol_RGC_ON [1] · pool 2。
3. 顶部“放电输出”选择“实验 temporal_readout_v1（INCONCLUSIVE）”。

紫色为实验动态期望率，蓝色为原生产基线，橙色仍为原保存拟合输出；
灰色为实测 PSTH。物理膜电压保持独立 mV 图，未被替换。显示按原20ms矩形箱，
底层输出仍为1ms概率和期望计数率。切回默认即移除紫色叠加，重新启动不会自动采用实验模式。

其他类型/节点、图、刺激或缓存身份不匹配时明确拒绝。当前只支持原六个 Cell6 条件案例；
group含两个模型成员，不是两条独立实测细胞状态。g3/g4/g5均为已触碰的开发数据。

## 独立导出

项目目录下，使用原环境及尚不存在的新输出路径：

    & 'D:\macaque_retina_simulator_envs\cu126_py311_v1\Scripts\python.exe' -B -X utf8 tools/export_temporal_output_v1.py --experimental --case cell6_g3 --output reports/cell6_temporal_output_integration_20261028_v1/user_g3_experimental.npz

只读原缓存，输出独立 NPZ 和同名 JSON，不覆盖原曲线。必须显式传 --experimental。
精确tau、训练mean/SD、参数和模型ID从原 MODEL.json 加载，不重新估计或取整。

## 流式观察

新增适配器直接复用原 TemporalReadout，不重造滤波或八态算法：

    from pathlib import Path
    from retina_platform.replay_data import ReplayData
    from retina_platform.temporal_output import (
        ConditionalTemporalOutput, EXPERIMENTAL, PHASE, binding_for_replay)
    root = Path(r"D:\macaque_retina_simulator")
    data = ReplayData(root).load_case("cell6_g3")
    node = data.nodes[data.target]
    binding = binding_for_replay(data, node)
    out = ConditionalTemporalOutput(root, binding, mode=EXPERIMENTAL)
    voltage = data.arrays["parasol_RGC_ON:V"][:-1, 1]
    first = out.push(voltage[:1237], data.time[:1237], phase=PHASE)
    saved_state = out.state()
    restored = ConditionalTemporalOutput(root, binding, mode=EXPERIMENTAL,
                                        state=saved_state)
    rest = restored.push(voltage[1237:], data.time[1237:-1], phase=PHASE)

输入必须是原生100μs、SI V、匹配group/表示/源时间索引的真实膜节点。适配器核对原科学合同、
case、刺激SHA、缓存payload SHA、模型及恢复身份；调用方仍负责提供其声明节点的真实电压。
本轮只验证缓存到此流式接口，没有推进全链来验证新的实时求解器接线。

源缓存为离散commit前节点：初始right在initial_commit前，后续遇输入frame取left，否则连续right。
1ms bin-start使用尚未消费当前100μs样本的a；7.1s末端孤立点没有后续bin，不输入。
各组共用原灰背景a0/pi，评分前1s为原记录上下文；不在评估或拖动位置重置、重新归一化。

自由输出没有真实spike/history参数，严格过去三个bin通过八态概率边缘化。
probability为每1ms事件概率，expected_count_rate_Hz=p/0.001，不是危险率λ。
当前仅支持自由期望；采样spike明确不可用，不创建伪造事件或RNG。
saved_state可JSON保存，但只覆盖输出状态，不是全链checkpoint。

## 证据与限制

最新验收目录：

    reports/cell6_temporal_output_integration_20261028_v1/validation_r2/

REPORT.md、CONSISTENCY_RESULT.json、GUI_RESULT.json及experimental_mode.png包含实际结果。
不规则分块在native索引35437由新进程恢复；与冻结g3离线预测比较，沿用原概率/状态1e-12
及期望率1e-9Hz预算，检查bin无重复/缺失及错误身份拒绝。
根目录第一版回执保留；r2只加强payload绑定并等待真实字段加载后截图。

有效编码时间常数不是已验证动态阈值或通道参数。输入物理标定、局部边界、解剖注册及g5误差
区间问题仍存在。本次不重训、不重新评分、不推进上游，不自动推广其他细胞/偏心度。
完成后停止。
