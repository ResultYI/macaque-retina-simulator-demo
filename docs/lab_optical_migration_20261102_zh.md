# 研究室光学依赖与缓存移交

## 源码基线

- 笔记本工程：`D:\macaque retinal simulator`
- 分支：`review/physiology-wip-20261017T2356Z`
- 本次移交依据的源码提交：`5b974384112f47688bc8ae6b972f491a7ac62161`

原光学入口和项目自有 MATLAB 依赖已在上述提交内。研究室先对照这个固定版本；若一致，源码部分无需另搬。本次新增资产提交保留该基线的模型代码和旧配置，不代表切换科学参数。

最初的版本说明核实的是笔记本及其本地远端记录，不能解释为对 GitHub 所有分支最新状态的证明。资产提交后的推送核验只针对本分支。

## 此次随主工程 Git 交付的实际文件

| 仓库内路径 | 文件数 | 字节数 | 说明 |
| --- | ---: | ---: | --- |
| `vendor_pinned/isetcam/` | 2,571 | 518,094,525 | 完整当前工作目录，排除各层 `.git` |
| `vendor_pinned/isetbio/` | 1,762 | 88,387,816 | 当前工作目录及原换行规则留档，排除各层 `.git` |
| `optical/cache/otf_001.mat` 至 `otf_101.mat` | 101 | 151,612,613 | 本次指定的冻结 OTF 缓存 |
| `optical/cache/model.mat`、`metadata.json` | 2 | 7,966,662 | 同一缓存的必要模型和元数据 |

两套依赖交付合计 **606,482,341 字节，约 606.5 MB**；完整缓存共 103 个文件、159,579,275 字节。MB 在此按十进制计算。

这两个 vendor 目录现在由主工程按实际文件记录，不依赖子模块或嵌套 Git 仓库才能取得内容。`data`、`external`、版本标识、根目录与子目录原有许可证均保留；依赖自己的 `.git` 未上传。固定上游提交标识为：

- ISETCam：`17146a9f0470cc89b58d583851f680187268506d`
- ISETBio：`d2924c2fe1d41c7dc5af1006d8081735b08f1b35`

上述提交号记录来源版本，不单独保证本机工作目录等同于干净的上游克隆。实际交付文件以 [vendor 文件清单](../vendor_pinned/OPTICAL_DEPENDENCIES_MANIFEST.json) 的逐文件 SHA256、长度和 Git 模式为准。

ISETBio 原 `.gitattributes` 的完整字节保存在 `.gitattributes.upstream`；有效 `.gitattributes` 末尾追加 `* -text`，避免主工程在检出时改写原 CRLF/LF 并破坏冻结 SHA256。该打包元数据调整单独记录在清单中，算法源码、数据、原模型配置和许可证未改动。清单中的 `120000` 模式表示原相对符号链接；Linux 核验这类条目时应对链接目标文本计算 SHA256，而非解引用目录。

研究室将它们放在**本项目自己的** `vendor_pinned/isetcam` 和 `vendor_pinned/isetbio` 下，不覆盖研究室原有的 ISETBio 安装或工作目录。

## 缓存来源与完整性

此次缓存原目录为：

```text
reports/GIN_69_6_HC2_BC1_SUPPORT_V2/optical/cache
```

仓库交付副本位于 `optical/cache`，原件保留。完整 SHA256、大小与来源见 [缓存资产清单](../optical/ASSETS_MANIFEST.json)。

该缓存元数据为 `769 × 769` 场景、`961 × 961` OI、`380:4:780 nm`、4 mm 瞳孔，并采用来源 CRT/文献亮度近似。不要仅凭相同的 `otf_*.mat` 文件名，把它与早期 LCD-Apple 演示缓存或其他运行缓存混用；必须保持 `model.mat`、`metadata.json` 与 101 个 OTF 文件属于同一完整目录。

`optical/reference/CONFIG.windows.json` 是原 `optical/CONFIG.json` 的原字节副本，保存原 SHA256 身份，供核对使用。它仍含笔记本路径，不是可直接在 Linux 执行的新配置。此副本也不表示其他 atlas、数据集、输入媒体或神经状态已包含在此次资产交付中。

## 必须保留的孔径语义

**这次实际 Cone 空间采样使用 Gaussian，`sigma = 0.204 × reference_diameter`。不能根据配置里初始的 `Pillbox` 字段，把实际 Cone 采样改回 Pillbox。**

实际采样路径使用项目的 Gaussian 分区适配和 ISETBio 核函数；逐细胞有效采光面积按对应高斯约定计算一次。缓存元数据中另有固定虚拟中央凹 Pillbox 参照面积，它服务于原参考标定，与实际局部 Cone 的 Gaussian 采样是不同角色。不要通过全局替换 `Pillbox` 或 `Gaussian` 来“统一”两者。

相关入口和适配代码包括：

- `tools/export_gin_crt_tile_v1.m`
- `tools/cone_video_gaussian_sampling.m`
- `tools/cone_native_aperture.m`
- `tools/matlab/sample_cone_aperture_zones.m`

## Linux 路径与身份

1. 先核对源码基线及两份资产清单，确认完整文件已拉取。保留旧项目、旧 ISETBio 和旧结果。
2. 把新的 MATLAB 可执行文件、项目根目录、两套 vendor 根目录、`optics_cache`/`cache_directory` 等路径写入独立迁移配置，例如 `configurations/local/lab_optical_paths_20261102.json`；这个示例文件名不是已有运行配置。
3. 运行时如需完整配置，另存一份新配置并按入口要求绑定这些路径。不得覆盖原 Windows 配置、原报告中的配置或旧配置哈希；记录新配置 SHA256 和它对应的原配置 SHA256。
4. 把路径指向本项目自带的 vendor 目录，确认 Linux MATLAB 能找到所需入口、数据和 `external`。平台相关二进制能否使用需要在研究室另做就绪检查。
5. 先做导入、路径与小文件读取检查，再按研究室明确授权开展计算；这份文件交付本身不启动模拟、训练或拟合。

## 原算法对照门限的范围

原同次运行的算法对照标准为：

| 对照项 | 原门限 |
| --- | --- |
| OI 各波段峰值归一化误差 | `< 1e-6` |
| OI 相对 L1 误差 | `< 1e-6` |
| Cone 捕获对照误差 | `< 3e-6` |
| 采样权重行和误差 | `< 1e-12` |

OI 这里指 `cached_vs_native_wave_peak_relative_error` 和 `relative_L1_error`，不包含近零填充点的逐点相对误差。它们是**同次运行的算法对照门限**，不是已经建立的跨机器复算标准。本次入库做文件与哈希核验，不宣称研究室 Linux 已通过这些数值对照。

模型、运行与科学资格维持原有记录；不能因依赖和缓存齐全就把 `FIT_READY` 或 `background_qualified` 改为 true。
