# 2026-10-20 RGC材料入口

本页对应既有 `review/physiology-wip-20261017T2356Z` 分支，供参数方案审阅。统一设计说明在 [core_design.md](core_design.md)。这些材料是来源核查、独立参考和未采用候选的工程结果，不代表整网实测标定完成。

| 材料 | 入口与结论 |
|---|---|
| 电位—hazard来源核查 | [来源对照](../reports/modeling_scope_20261020/rgc_voltage_hazard_source_audit_v1/primary_source_review.json)、[8状态精确代数](../reports/modeling_scope_20261020/rgc_voltage_hazard_source_audit_v1/result.json)：旧参数在零突触电导、EL处产生约135.38Hz，未证明为单位或公式BUG。 |
| 原始V—spike数据检查 | [逐源检查](../reports/modeling_scope_20261020/rgc_v_spike_intake_v1/source_inventory.json)、[最小数据清单](../reports/modeling_scope_20261020/rgc_v_spike_intake_v1/minimum_primary_data_handoff.json)：公开Cell_6不是图3两只动态钳细胞的配对V—spike记录。 |
| 作者参考复算 | [比较图](../reports/modeling_scope_20261020/on_parasol_author_reference_v1/on_parasol_author_reference.png)、[完整结果](../reports/modeling_scope_20261020/on_parasol_author_reference_v1/result.json)：NumPy/SciPy逐式复算；2020/2022版本差异分别保留，没有直接执行JAX notebook或MATLAB。 |
| 真实整网候选对照 | [比较图](../reports/modeling_scope_20261020/on_parasol_candidate_network_v1/on_parasol_candidate_network.png)、[汇总](../reports/modeling_scope_20261020/on_parasol_candidate_network_v1/ON_parasol_summary.csv)、[配对控制](../reports/modeling_scope_20261020/on_parasol_candidate_network_v1/matched_control_result.json)。 |

[候选配置](../configurations/on_parasol_literature_candidate_v1.json)仅将ON-parasol的α/μ/β迁移为90Hz/−53mV/1.67mV，保留项目膜条件、history和dt。背景期望计数率由273.2变为42.0Hz；亮暗增量仍不足1Hz，单种子计数未分辨出该响应。候选没有改变生产默认或其他四类参数。实际运行42组件、20,422细胞，共推进2.2s模型时间；神经路径为完整耦合整网，输入为明确的工程根通道阶跃。真实ISET标准视频捕获没有执行。

## 数据与复核

- [发布清单](rgc_oct20_publication_manifest.json)保存本次精确路径、字节和SHA256，以及必要但按项目约定保留本地的外部资产身份。
- 已包含四组报告的摘要、完整合并轨迹、必要状态检查点、参数/输入身份、统计数组、回归收据和图表；重复分段及`state_latest`副本不重复上传。
- 公共原始Cell_6保留其Figshare v1身份和CC BY4.0元数据，用于重现数据检查；它没有被当成配对V—spike或新拟合输入。
- 原文XML保留自身CC BY许可；缓存作者Python源码固定在commit `7beb7e72b0678192f0bc98392a678cdff9601791`，附[原MIT许可](../data/source_data/latimer2019_figshare_v1/LICENSE.CBEM-MIT.txt)。
- 本机ISET设置、捕获缓存、私人视频、历史视频的大型状态包、`vendor_pinned`及生成的atlas资产继续按既有发布约定留在本地。没有把不完整包称为完整光学/神经环境安装包。

在含NumPy/SciPy的既有核心环境中可运行：

```text
python -B -X utf8 tools/verify_published_rgc_results.py
```

该命令验证发布文件身份、来源、实际参数差异和记录数组，不启动神经模拟、MATLAB或拟合。完整神经重跑还需要发布清单列出的原冻结atlas/来源资产及相同运行环境；原实验脚本保留历史视频保护检查，因此缺少本地历史证据时会显式停止，不能将缺失检查冒充通过。恢复这些原资产后，使用已提交的`run_on_parasol_candidate_network.py`和`run_on_parasol_matched_gray_controls.py`，不混用参数检查点。

已查看过的Cell_6验证结果保持拒绝状态。原始配对V—spike、独立生理测试、膜条件/区域等效性仍是未闭合问题。请以本分支或明确commit为准读取；推送不证明任何Pro会话已经取得或读过内容。
