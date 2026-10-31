# 完整响应的桌面分层回放

双击工程根目录 `run_retina_gui.cmd`。它现在打开原 Tkinter 软件的完整响应模式，不打开网页。Cell6 既有回放保留在左侧按钮中；原模拟设置界面也有“完整响应 / 视频分层回放”入口。

命令行启动：

```powershell
& '.\.venv\Scripts\python.exe' -B -X utf8 -m retina_platform --response-replay --result 'reports\result_packages\retina_113_4s_298fa2ef1524.response.zip'
```

## 使用

1. 选择“导入结果ZIP”，直接选择生成的结果包。也可导入已完成运行目录或其 JSON manifest。
2. 选择 near / far，再选择 RGC 类型和实际组 ID。默认只显示该目标的真实连接祖先；反馈、电耦合会扩大祖先范围。勾选“全部patch作灰色对照”可看其余组。
3. 底部播放、暂停、逐视频帧步进、拖动和速度共用物理时间轴。视频来自包内的原始 RGB 帧。本例为113.AVI原视频的0–4秒、120帧，而非整个14.9秒。
4. 点击任一层的点，查看组 ID、类型、坐标及坐标语义、代表成员数、当前状态与单位。“细胞 / 状态详情”提供完整内容；“响应曲线”独立打开，能切换实际保存的字段。
5. 左栏可滚动。窗口缩小时播放条固定保留，各层说明随宽度换行。

导入刺激视频时先用完整文件 SHA 匹配现有结果。当前已有113.AVI的对应4秒结果。没有匹配响应时清空旧热图并提示“需要计算”；不会静默启动长计算，也不会伪造热图。任意新视频的物理输入定义与计算仍由对应计算入口负责，本回放器不发明这些参数。

## 各层实际可看内容

|层面|主图状态|单位与身份|
|---|---|---|
|Cone|release|dimensionless；实际代表组，L/M/S身份保留|
|HC|Vh|vH_native_voltage，不能标作物理mV；H1/H2身份和soma坐标保留|
|BC五类|voltage|保存V，显示可换为mV；类型分别保留|
|普通AC四类|voltage|V→mV；每个目标组的有效功能RF池，不冒称解剖AC位置|
|S通路AC|inhibition_state|dimensionless；没有伪造未保存的膜电压|
|Rod|voltage|V→mV；本实验是固定工程背景|
|RBC / AII|voltage|V→mV，另有已保存release、activation等字段|
|RGC五类|voltage或期望二元率|V→mV或Hz；曲线还可选gE/gI、各电流、导数和hazard|

精确的每位置、每群、字段和单位清单见 `reports/response_viewer_delivery_v1_20261030/FIELD_COVERAGE.json`。原响应块保留所有保存字段；接触电导或选定细胞辅助状态若没有完整细胞坐标对应，不冒充逐组空间热图。没有保存的状态不重建。

主视图使用实际保存的1ms节点，选择不晚于当前时刻的样本；不插值。原始9024条记录及左右事件侧仍完整留在响应块中。Δ显示只减各组t=0状态；色标在整个显示片段内固定，绝不逐帧自动拉伸。全片色标是只读显示信息，不参与模型计算。各卡片用各自固定空间范围，坐标单位为度。

点代表模型组或有效功能池。多成员组不是多个独立记录，也不把组事件复制成成员spike。“祖先”指实际非零连接拓扑，不表示已证明每条连接对本视频的独立因果贡献。外部S0储库冻结、Rod工程背景、Sbi高基线/混合输入、混合参考几何等研究限制不变。三项资格仍为false。

## 一个ZIP包含什么

版本化 `manifest.json` 按以下类别绑定相对路径、SHA、大小、用途、字段单位、时间、组身份、范围和缺失项：

- `stimulus/`：120帧原始RGB及帧率、曝光、时间定义；不把有损预览当模型输入。
- `responses/near`、`responses/far`：完整原始NPZ时间分块、原身份、单位、事件和数值/资源结果。
- `cells/`：实际局部组ID、成员数、坐标/角色和非零连接CSR，包括有效AC池与HC同型耦合。
- `provenance/`：参数、来源、初始化和选择身份。历史绝对路径只作来源描述，回放不沿这些路径取文件。
- `reports/`、`logs/`：必要报告、指标、曲线和执行日志，包括原失败记录。

示例ZIP约1.49GB，主要是保留原字节身份的已压缩响应NPZ；没有打入vendor、全工程或无关光学缓存。包用于完整已保存响应的回放，不是完整可恢复模型checkpoint。

导入不执行脚本、不加载pickle；拒绝绝对路径、路径穿越、大小写重复路径、符号链接、哈希不符和超限解压。限制为4096文件、总8GiB、单文件2GiB，NPZ内部数组另有限额。导出先完成临时文件，再原子改名；已有包不会覆盖，原始结果不会删除。

## 导出与后续自动整理

界面“导出结果ZIP”与以下命令使用同一导出器：

```powershell
& '.\.venv\Scripts\python.exe' -B -X utf8 tools/export_response_bundle.py --experiment 'reports\external_video_two_positions_4s_20261029_v1' --output 'reports\result_packages\new_unique_name.response.zip'
```

已接入的自动范围是现有完整视频入口 `tools/run_external_video_pipeline_v1.py`：两位置响应、统计及交付成功后自动写入该次运行目录的 `packages/*.response.zip`。失败或部分运行写 `PARTIAL_RESPONSE_EXPORT.json`，不冒充完整包。此改动没有启动该昂贵流程；导出器已用本次真实完成数据单独执行并导入验证。旧Cell6/其他历史结果格式尚不自动转换为此新包，仍保留其原入口。

## 验证入口

小型合成fixture只用于软件测试，不作为科研结果：

```powershell
& '.\.venv\Scripts\python.exe' -B -X utf8 -m unittest discover -s tests -p test_response_bundle.py -v
```

真实桌面测试：`tools/verify_response_desktop_v1.py`。它只导入现成ZIP，测试近/远组选择、真实祖先变化、播放/拖动/步进/速度、细胞点击、曲线、缩放及未知刺激提示，不运行模型。证据位于 `reports/response_viewer_delivery_v1_20261030/`。本次截图来自实际Tk窗口；未使用或绕过被禁止的本地HTML浏览器入口。
