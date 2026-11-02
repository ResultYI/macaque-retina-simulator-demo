# 2026-11-02 组会资料

资料目录：`D:/macaque retinal simulator/semi_reports/2026-11-02_HC`。直接编辑 `main.tex`，在同一目录编译 `main.pdf`。

## 资料内容

日文资料正文两页、参考文献一页，共三页。图像顺序为 ISETBio Cone 分布预览、同尺度 Cone 大小图、HC 分布与密度图、合并 HC 应答图、H1/H2 接触半径对比图。图3说明为“Coneからの入力に対するHCの応答。”。图4展示0–6°半径曲线和0°、3°、6°处的同尺度输入圆，数值为半径，单位μm；文献依据在图注中引用。正文包含数据来源和三项指数密度函数两个小节，最后说明采用 van Hateren（2005）的模型生成 HC 响应。

密度函数形式参考 Virtual Human Retina（Ly et al., 2025；DOI `10.1016/j.brs.2025.01.013`）的三项指数式；系数依据 Röhrenbeck（1989）、Krebs与Krebs（1989）、Ahmad（2003）及Wässle（2000）的密度数据设定。密度使用 `configurations/hc_density_rf_joint_v2.json` 的全精度系数，H1/H2按0.8/0.2拆分。HC生成区域为解剖半径0–1.75 mm，含H1 **138,369** 个、H2 **34,592** 个，共 **172,961** 个。图2显示其中解剖半径0–1.338 mm（6°相当）的H1 **90,616** 个、H2 **22,669** 个。整个生成区域的积分期望分别为138,368.5442和34,592.1361，整数化误差均小于1个细胞。生成采用固定seed20261010、联合短程排斥和2个sweep。

直接接触半径在0–0.58 mm取H1 8.04 μm、H2 12.22 μm；0.58–0.60 mm按固定20 μm三次Hermite段连接；外围使用 `R_t=sqrt(d_t/(pi*rho_t))`。1.338–1.75 mm沿用同一密度、比例和接触方程，作为有记录的局部支持区域外延；全局冻结配置的适用范围保持原声明。图4横轴为HC胞体偏心度，0–6°对应0–1.338 mm。在1.338 mm处，H1/H2半径约10.0090/14.3649 μm；1.75 mm处约11.7667/16.2893 μm。该量是直接Cone接触范围；完整功能性RF尚未定量拟合。H1排除S输入、H2接收L/M/S，反馈为H1→L/M、H2→S，同型侧向耦合独立计算。

## 计算结果与范围

运行数据：`reports/HC_VISUAL6DEG_SUPPORT_REPAIR_20261101_v1`。

- 几何生成启动：2026-11-01 22:58:42 JST；完成：23:08:38 JST；阶段用时597.26 s。数值合同记录23:11:38–23:22:23 JST，响应阶段含装配用时651.34 s。
- 有限网络含545,658个原始Cone、48,620个Cone代表群；172,333个有直接输入的HC各有独立状态。
- 实际接触1,452,969条；H1/H2平均汇聚数（含零输入细胞）为6.6356/15.4604。H1没有S输入，最大接触距离/半径比为0.9999992308。
- 515个H1和113个H2无直接输入，共628个，保留在结构数据中。应答图显示输入重心位于0–6°视野内的137,094个HC。
- 190,498个Cone接受ISETBio首帧输入，曝光1/30 s；未捕获成员使用100参照等价Td。完整成员平均和已有Cone分组用于输入转换。
- 背景重新求解，最大原生RHS为9.09495e-12。标准DOP853参数为rtol=1e-6、atol=1e-9、最大步长100 μs；紧精度为1e-7、1e-10、50 μs。最大HC差为4.83169e-13，全状态缩放误差比为2.13603e-08，数值门槛通过。
- 响应阶段记录的峰值内存约2.542 GiB；生成/连接阶段的采样峰值不作为完整连接峰值声明。

观察范围为0–6°视野，HC计算域为解剖半径1.75 mm的有限开边界圆盘。输入Cone的最大突触端偏心距为1.552196 mm；计入最大接触半径后，距边界仍有0.181515 mm余量。全部190,498个输入Cone具有完整几何邻域，其中190,223个有HC直接接触，190,006个有分型反馈。5.5–6°视野内18,894个Cone有18,858个接入HC，接触率99.8095%。几何筛选产生的未接触细胞按实际结果保留。

应答图采用实际Cone输入位置的重心，最大显示偏心度5.99996°，坐标轴±6°；5.5–6°内12,846个显示HC均有非零应答。位置和应答逐细胞取自保存数据。共享同一内部几何、分别重求背景的1.65/1.75 mm边界检查通过：137,094个观察HC的最大应答差1.97863e-4，低于1e-3模型原生单位门槛。较小区域的精度复核最大HC差6.96332e-13。检查结束于2026-11-01 23:31:43 JST，详见 `BOUNDARY_CONVERGENCE.json` 和 `FINAL_ACCEPTANCE.json`。

密度数据包含跨动物记录和二手混合灵长类补助点；固定H1/H2比例、中心弱约束段、接触圆盘、C1过渡与外围支持区域的函数延拓属于模型近似。D1的全Cone/可接触L/M分母问题保留在配置中。`FIT_READY`、`production_adopted`和生理验证标记均为false；结果对应33.3 ms首帧。

## 数据与出图

`GEOMETRY_CONTRACT.json`、`HC_GEOMETRY.npz`、`MOSAIC_DIAGNOSTICS.json`和`BUILD_RESULT.json`记录生成范围、参数、坐标及校验。`RAW_HC_BY_CONE.npz`、`A.npz`、`B.npz`与`CONNECTIVITY_AUDIT.json`记录接触与投影。`VISUAL_COVERAGE.json`、`COVERAGE_TYPED.json`记录逐环带的实际输入覆盖；`BACKGROUND.npz`、`STANDARD.npz`、`TIGHT.npz`、`HC_RESPONSE.npz`、`RESULT.json`记录背景、动态和数值验收。运行终态与资源记录保存在对应运行目录。

本目录的 `hc_density_model.csv` 和 `hc_density_annuli.csv` 分别保存模型密度与实际环带统计；`hc_fullfield_response_preview.npz/json`保存图中坐标、应答值和来源。图像的ISETBio输入与冻结光学记录一致。`hc_contact_radius_model.csv/json`记录直接调用 `JointV2HCDensityProfile.contact_radius_mm` 得到的半径及来源，绘为 `hc_contact_radius_comparison.pdf/png`。1203个采样点及三个示例与模型返回值逐点一致；密度和接触半径的源码、参数与保存网络的相应绑定一致。

计算入口为 `refresh_hc_current.py` 的 `build`、`audit`、`respond`、`boundary` 四个阶段，均使用 `--visual-support`。已有结果目录拒绝覆盖。边界对照的单线程BLAS环境记录在 `BOUNDARY_EXECUTION.json`，数值容差与主计算一致。`before_visual_support_repair.zip`保存资料及图的归档，独立研究运行的数据按原身份保留。

在项目根目录生成接触半径图：

```powershell
& ./.venv/Scripts/python.exe ./semi_reports/2026-11-02_HC/prepare_assets.py size-data
& D:/anaconda/python.exe ./semi_reports/2026-11-02_HC/prepare_assets.py size
```

已完成运行的应答与密度图入口如下，保留来源身份校验，配置身份须与保存运行匹配：

```powershell
& ./.venv/Scripts/python.exe ./semi_reports/2026-11-02_HC/prepare_hc_preview.py extract-fullfield
& D:/anaconda/python.exe ./semi_reports/2026-11-02_HC/prepare_assets.py plot
& D:/anaconda/python.exe ./semi_reports/2026-11-02_HC/prepare_hc_preview.py plot-fullfield
```

在资料目录编译：

```powershell
./build.cmd
```

编译使用既有upLaTeX → dvipdfmx链，TeXstudio使用其“构建并查看”。`QA.json`保存文件hash、源码绑定、数量、精度、图序及三页版面检查结果。

正式参考文献共8条，均在正文及图注中引用：Ly2025、Röhrenbeck1989、Krebs1989、Ahmad2003、Wässle2000、Wässle1989、Boycott1987、vanHateren2005。
