# 2026-11-02 组会资料

固定目录：`D:/macaque retinal simulator/semi_reports/2026-11-02_HC`。
后续直接修改本目录的 `main.tex`，编译覆盖 `main.pdf`，沿用同一个资料文件夹。

2026-10-31 根据当前代码精简为两页日文资料。标题无副标题，姓名与日期分行，报告日期仍为 2026 年 11 月 2 日。
2026-11-01 加入 Cone 大小同尺度对比图及简短日文图注后，当前资料共三页。
同日将 HC 预览更新为完整有限 Cone–HC 网络的首帧应答，H1/H2 合并显示，仍为三页。
随后将应答图统一为视野坐标，按直接 Cone 输入的重心定位；计算结果沿用原数据。
最新版本删除第二节的公式与详细说明，在第一节末尾用一句话交代 van Hateren（2005）响应模型，资料共两页。
随后在第一节补入 HC 接触范围随偏心度的变化及出处，仍为两页。

## 当前结构

开头图像依次为：ISETBio Cone 分布预览及 0°、3°、5.8° 的同尺度大小对比图 → HC 分布与密度图 → 全网络 HC 应答结果图。前两组位于第一页，应答结果位于第二页正文之前。Cone 两图共用简短日文图注“加入 Cone 大小变化的分布生成预览图”。HC 细胞体分布图保存坐标与当前表一致：H1 82,456 个、H2 20,597 个；6° 对应 1.338 mm。

1. **分布来源和密度方程**：Röhrenbeck（1989）的中央总密度锚点，Wässle（2000）的分型密度与中央比例；列出中央 smoothstep 密度、外侧插值和面积积分生成坐标的思路。
第一节末尾补充：按密度生成 HC 后，使用基于 van Hateren（2005）的模型计算响应，并保留该文献引用。第二节及其公式已删除。

1.1 合并说明来源：总 HC 密度参考 Röhrenbeck（1989）；H1/H2 的密度、个数比和随偏心度变化的 RF 尺寸参考 Wässle（2000），正文将 RF 限定为直接 Cone 输入范围。1.4 直接列出现行接触径函数：0–1 mm 固定，1–5 mm 按对数径 PCHIP 的等价三次表达式增长，5–6 mm 固定；q=(r−1)/4，g(q)=1.8q−0.6q²−0.2q³，D=D0(Dp/D0)^g。中央径 H1/H2 为 16.08/24.44 μm，外侧径约 61.8/63.5 μm。已按当前配置与代码核对该表达式；延长和插值属于本研究设定。5–6 mm 锚点位于目标解剖 0–6° 范围之外；完整功能性 RF 尚未定量评估。末句总结按上述密度分布配置 HC，再使用 van Hateren（2005）的模型生成 HC 响应。

正文保留密度来源、插值和坐标生成的说明。原有详细连接统计、局部验证图及来源记录继续保存在本目录，便于独立复核。

全网络应答使用最新偏心度接触结构及 fine 行先分型归一化、再取群均值的反馈矩阵，计算整个有限 Cone–HC 网络。190,498 个 0–6° Cone 接受 ISETBio 完整首帧光学输入，持续 33.3 ms；图像范围外的光输入保留 100 参照等价 Td，外侧神经状态也共同演化。48,620 个 Cone 代表群和 32,652 个有输入 HC 代表群参与计算，重新求得相容背景。标准与更紧时间精度对照的最大 HC 差为约 `3.27e-13`，验证通过。

当前应答图与输入图统一使用视野坐标。每个 HC 的显示位置取实际接触 Cone 感光位置的连接加权重心，权重来自冻结的完整 fine Cone→HC 矩阵，与图像内容和应答强弱无关；该重心是显示位置的近似。半径 6° 视野内共显示 127,268 个 HC，H1/H2 合并使用同一色标；颜色仍逐项取自原 `h(33.3ms)-h(0)`，各细胞继承已有代表群状态。新增显示的 24,244 个 HC 胞体在原解剖 6° 窗口之外，其状态已在原全网络运行中计算。无接触的 29 个 HC 无法定义输入重心，在视野图中省略，原始解剖数据保留。未重算动态、改动响应数值或进行空间插值。

开头的细胞体分布图仍按解剖位置统计：半径 6° 对应 103,053 个 HC。它与应答图的选取坐标不同。旧解剖位置应答图保留为 `hc_fullfield_anatomical_preview.png/pdf`，对应数据与来源也以此名称保存。

计算身份及原始数据保存在 `reports/HC_FULLFIELD_FIRST_FRAME_0_6DEG_20261101_v1`。本轮完成首帧全网络应答，尚未计算完整 4 秒视频。旧局部预览资产和既有研究结果保留。

## 代码对应

- 密度与坐标生成：`retina_sim/hc_mosaic.py`。
- HC 接触范围：`retina_sim/atlas_hc.py`、`retina_sim/disk_hc_functional.py`。
- ISETBio 光学采样：`tools/export_cone_video_aperture_tile.m`。
- R*/s 到参照等价输入：`retina_sim/isetbio_input.py` 的 `reference_equivalent_drive()`。
- 代表群的输入平均：`tools/prepare_external_video_v1.py`。
- Cone 与 H1/H2 方程：`retina_sim/h1_h2_effective.py`、`retina_sim/vh2005.py`。
- 同型侧向项：`retina_sim/hc_lateral_candidate.py`。

本次沿用原有方程与参数，新增首帧全网络计算及资料预览。既有模拟结果保留。

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

2026-11-01 新增 `isetbio_cone_distribution.pdf`，来源为 `reports/cone_video_113_4s_eccentric_aperture_0_6deg_20261029_v1/first_frame/HD/frame_000.png`。保留原图像素，封装为 PDF 并裁去外围空白，便于本地编译。原始图片保留在原目录。
同目录的 `diameter_same_scale.png` 原样复制为 `isetbio_cone_diameter_same_scale.png`，展示三个偏心度的相同尺度局部图。

当前 HC 预览为 `hc_fullfield_response_preview.png/pdf`，对应数据和来源分别为同名 `.npz`、`.json`。图像来自原始 ISET 输入中的裁切首帧。以下命令只提取已完成的全网络结果和重绘，不调用求解器：

```powershell
& .\.venv\Scripts\python.exe .\semi_reports\2026-11-02_HC\prepare_hc_preview.py extract-fullfield
& D:\anaconda\python.exe .\semi_reports\2026-11-02_HC\prepare_hc_preview.py plot-fullfield
```

`QA.json` 记录本次编译、单节结构、开头图像、代码核对和版面检查。连接与局部响应的原验证仍分别见 `hc_connectivity_validation.json`、`hc_response_validation.json`。
