# 2026-11-02 组会资料

固定目录：`D:/macaque retinal simulator/semi_reports/2026-11-02_HC`。
后续直接修改本目录的 `main.tex`，编译覆盖 `main.pdf`，沿用同一个资料文件夹。

2026-10-31 根据当前代码精简为两页日文资料。标题无副标题，姓名与日期分行，报告日期仍为 2026 年 11 月 2 日。

## 当前结构

开头：现有 0–6° HC 分布与密度图。已核对图片所用保存坐标与当前 HC 表逐项一致：H1 82,456 个、H2 20,597 个；6° 对应 1.338 mm。

1. **分布来源和密度方程**：Röhrenbeck（1989）的中央总密度锚点，Wässle（2000）的分型密度与中央比例；列出中央 smoothstep 密度、外侧插值和面积积分生成坐标的思路。
2. **ISETBio 光学输入到 HC 响应**：期望光异构化数除以时间窗，使用固定 L/M/S 参照系数得到 Cone 模型输入；再说明 Cone 放出、HC 输入平均、H1/H2 时间方程、同型侧向耦合和回到 Cone 的反馈。

正文保留接触范围与功能性 RF 的区别、中央半径外推和工程输入尺度的说明。原有详细连接统计、局部验证图和原文裁切资产继续保存在本目录，便于独立复核。

## 代码对应

- 密度与坐标生成：`retina_sim/hc_mosaic.py`。
- HC 接触范围：`retina_sim/atlas_hc.py`、`retina_sim/disk_hc_functional.py`。
- ISETBio 光学采样：`tools/export_cone_video_aperture_tile.m`。
- R*/s 到参照等价输入：`retina_sim/isetbio_input.py` 的 `reference_equivalent_drive()`。
- 代表群的输入平均：`tools/prepare_external_video_v1.py`。
- Cone 与 H1/H2 方程：`retina_sim/h1_h2_effective.py`、`retina_sim/vh2005.py`。
- 同型侧向项：`retina_sim/hc_lateral_candidate.py`。

本次仅重写资料并核对公式，模型参数、结构和既有模拟结果保持原状。

## 编译与图像

在本目录运行：

```powershell
.\build.cmd
```

构建使用本机既有 upLaTeX → dvipdfmx 流程。TeXstudio 继续使用已配置的“构建并查看”；文件头保留 document-id，不设置编译器覆盖命令。

正文复用 `hc_distribution_density.pdf`。需要从保存数据重绘分布图时，在本目录运行：

```powershell
& 'D:/anaconda/python.exe' ./prepare_assets.py plot
```

`QA.json` 记录本次编译、两节结构、开头图像、代码核对和版面检查。连接与局部响应的原验证仍分别见 `hc_connectivity_validation.json`、`hc_response_validation.json`。
