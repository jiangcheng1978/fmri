# fMRI 自动处理系统

从 DICOM 到分析报告的完整自动化管线，无需 FSL 或 AFNI 等专业软件。

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/jiangcheng1978/fmri/actions/workflows/test.yml/badge.svg)](https://github.com/jiangcheng1978/fmri/actions)

## 目录

- [简介](#简介)
- [系统要求](#系统要求)
- [从零部署](#从零部署)
- [使用数据](#使用数据)
- [一键运行](#一键运行)
- [查看结果](#查看结果)
- [技术细节](#技术细节)
- [测试](#测试)
- [常见问题](#常见问题)
- [项目结构](#项目结构)
- [许可](#许可)

---

## 简介

本系统面向 **Siemens Prisma 3T** 扫描仪采集的功能性磁共振成像 (fMRI) 数据，实现了从原始 DICOM 到质量评估报告的完整自动化处理管线。

**核心能力：**
- 自动解包 Siemens MOSAIC 多 band 编码（4×4 = 16 band）
- FFT 加速的亚体素精度头动校正（0.05 mm）
- 空间平滑、去趋势、带通滤波
- 多维度质量评估：FD / tSNR / SNR
- 自动生成 Markdown 报告和 HTML 文档

## 系统要求

- **操作系统：** macOS / Linux（已测试 macOS 和 Ubuntu）
- **Python：** 3.12 或更高版本
- **内存：** 建议 ≥ 16 GB RAM（处理过程中可能需要较大内存）
- **磁盘空间：** 至少 2 GB 可用空间

### 依赖工具

| 工具 | 用途 | 安装方式 |
|------|------|----------|
| Python 3.12+ | 运行环境 | `brew install python` (macOS) 或 `apt install python3.12` (Linux) |
| uv | 包管理工具 | `curl -LsSf https://astral.sh/uv/install.sh | sh` |
| dcm2niix | DICOM → NIfTI 转换 | `uv pip install dcm2niix`（自动下载） |

## 从零部署

以下步骤假设你从一台全新的电脑上开始，需要从零搭建整个项目。

### 步骤 1：安装 Python

**macOS：**
```bash
# 方法一：使用 Homebrew（推荐）
brew install python@3.12

# 验证
python3 --version
# 应输出: Python 3.12.x
```

**Linux (Ubuntu/Debian)：**
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev

# 验证
python3.12 --version
```

### 步骤 2：安装 uv

uv 是一个极快的 Python 包管理工具，比 pip 快 10-100 倍。

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 重启终端或执行 source
source $HOME/.local/bin/env

# 验证
uv --version
# 应输出: uv 0.x.x
```

### 步骤 3：克隆项目

```bash
# 克隆到本地
git clone https://github.com/jiangcheng1978/fmri.git
cd fmri

# 查看目录
ls
# 应看到: data/ scripts/ tests/
```

### 步骤 4：创建虚拟环境

```bash
# 创建虚拟环境
uv venv .venv --python 3.12

# 激活虚拟环境
source .venv/bin/activate

# 验证激活成功
which python
# macOS: /Users/xxx/fmri/.venv/bin/python
# Linux: /home/xxx/fmri/.venv/bin/python
```

### 步骤 5：安装 Python 依赖

```bash
# 一次性安装所有依赖（约 2 GB，需要几分钟）
uv pip install pydicom nibabel nilearn numpy matplotlib scipy scikit-learn pandas

# 安装 dcm2niix（会自动下载可执行文件）
uv pip install dcm2niix

# 验证所有包安装成功
python -c "import pydicom, nibabel, nilearn, numpy, scipy, matplotlib, sklearn, pandas; print('✅ 所有依赖安装成功')"
```

### 步骤 6：准备数据

将你的 DICOM 数据放入数据目录：

```bash
# 目录结构
data/
├── sub_003/           # fMRI 数据（BOLD 序列）
│   ├── IMA_0001.IMA   # DICOM 文件（至少 244 个）
│   ├── IMA_0002.IMA
│   └── ...
└── t1_original/       # T1 结构像（可选）
    ├── IMA_0001.IMA
    └── ...
```

**数据格式说明：**
- DICOM 文件扩展名通常为 `.IMA`（Siemens）或 `.dcm`（通用）
- fMRI 数据应为一个 BOLD 序列，推荐 TR ≈ 2.0s，时间点数 ≥ 200
- 如果你还没有数据，可以从 [OpenNeuro](https://openneuro.org/) 获取公开数据集

**如果你有自己的数据，但目录名不同（例如 `sub-001`），需要修改 `scripts/run_pipeline.py` 中的目录名。**

### 步骤 7：运行管线

```bash
# 确保虚拟环境已激活
source .venv/bin/activate

# 一键运行完整管线
python scripts/run_pipeline.py
```

**完整管线包含 4 个步骤：**

```
Step 1: DICOM → NIfTI 转换 (~30-60s)
  将 Siemens MOSAIC 格式 DICOM 文件转换为 NIfTI 格式
  自动解包 4×4 MOSAIC 网格为 64 个独立切片

Step 2: fMRI 预处理 (~1-3min)
  - FFT 头动校正（亚体素精度）
  - 空间平滑 (FWHM=4mm)
  - 线性去趋势
  - FFT 带通滤波 (0.01-0.1 Hz)

Step 3: 质量评估 (~10s)
  计算 FD、tSNR、SNR 等指标

Step 4: 报告生成 (~30s)
  生成 Markdown 报告和可视化图片
```

## 使用数据

**如果你只想体验，没有真实数据，可以使用模拟数据：**

```bash
# 创建一个简单的模拟数据集用于测试（可选）
# 下面是一个快速生成模拟 BOLD 数据的方法

source .venv/bin/activate

# 创建测试目录
mkdir -p data/sub_003

# 运行测试（会自动创建模拟数据目录结构）
python tests/test_project.py
```

> ⚠️ 模拟数据仅用于测试管线是否运行正常，不能替代真实 fMRI 数据。

## 一键运行

如果上面的 `run_pipeline.py` 运行遇到问题，可以分步执行：

```bash
source .venv/bin/activate

# 分步执行
python scripts/dicom/dicom_convert.py              # Step 1: DICOM → NIfTI
python scripts/preprocess/fmri_preprocess.py        # Step 2: fMRI 预处理
python scripts/analysis/report_generator.py         # Step 3: 质量评估与报告

# 生成 HTML 文档
python scripts/generate_docs.py
```

## 查看结果

```bash
# macOS 用户
open output/output_report/fmri_analysis_report.md     # Markdown 报告
open output/output_report/spatial_maps.png            # 空间分布图
open output/output_report/temporal_analysis.png       # 时间序列分析图
open html/fmri-analysis.html                          # HTML 综合文档

# Linux 用户
xdg-open output/output_report/fmri_analysis_report.md
# 或直接查看文件
cat output/output_fsl/preprocessing_stats.json        # 统计信息
head output/output_fsl/motion_params.txt              # 运动参数
cat output/output_fsl/framewise_displacement.txt      # 帧位移
```

**每个输出文件的含义：**

| 文件 | 说明 |
|------|------|
| `output/output_fsl/sub-003_preproc.nii.gz` | 预处理后的 4D 数据 |
| `output/output_fsl/motion_params.txt` | 6 个运动参数（3 平移 + 3 旋转） |
| `output/output_fsl/framewise_displacement.txt` | 每帧的位移量 (FD) |
| `output/output_fsl/preprocessing_stats.json` | 所有质量指标的 JSON 汇总 |
| `output/output_report/fmri_analysis_report.md` | 完整的质量分析报告 |
| `output/output_report/spatial_maps.png` | 空间分布图（均值、标准差、tSNR） |
| `output/output_report/temporal_analysis.png` | 时间序列图（FD、运动参数、PSD） |
| `html/fmri-analysis.html` | 综合 HTML 文档（含术语解释和公式） |

## 技术细节

### 预处理管线

| 步骤 | 方法 | 关键参数 |
|------|------|----------|
| 头动校正 | FFT 互相关（两阶段） | 前 5 帧均值参考，亚体素精度 |
| 空间平滑 | 高斯滤波 | FWHM = 4.0 mm |
| 去趋势 | 线性回归 | 减均值 + 减线性斜率 |
| 带通滤波 | FFT 频域滤波 | 0.01 – 0.10 Hz |

### 质量评估指标

- **FD (Framewise Displacement)**：相邻帧之间的位移量，< 0.1 mm 为优秀
- **tSNR (temporal Signal-to-Noise Ratio)**：时间序列均值/标准差，> 50 为良好
- **SNR (Signal-to-Noise Ratio)**：空间信噪比，信号均值/标准差

## 测试

```bash
source .venv/bin/activate

# 运行全部测试
python tests/test_project.py -v

# 测试覆盖
#   - 项目结构 (6 tests)
#   - 依赖导入 (8 tests)
#   - 脚本语法 (1 test)
#   - 数据完整性 (5 tests)
#   - 预处理函数 (5 tests)
#   - 输出合理性 (5 tests)
#   - HTML 文档 (4 tests)
```

## 常见问题

### Q: 找不到 dcm2niix？

确保已激活虚拟环境并安装：
```bash
source .venv/bin/activate
uv pip install dcm2niix
```

### Q: 找不到 .IMA 文件？

确认数据在 `data/sub_003/` 目录下：
```bash
ls data/sub_003/*.IMA
```

### Q: 质量评级为"较差"？

被试头动过大。建议：
1. 在 GLM 分析中加入 FD > 0.5mm 的 volumes 作为 censoring regressors
2. 或增大空间平滑 FWHM（如从 4mm 增至 6mm）

### Q: tSNR 偏低？

可尝试增大空间平滑 FWHM（如从 4mm 增至 6mm），但会降低空间精度。

### Q: 内存不足？

fMRI 处理需要较多内存。建议：
- 使用更小的数据（较少时间点数）
- 减少数据维度（如降低空间分辨率）

### Q: 代理网络问题？

如果从国外源下载依赖较慢：
```bash
# 设置 pip 镜像（以清华源为例）
uv pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 设置代理
export http_proxy=http://localhost:7897
export https_proxy=http://localhost:7897
```

## 项目结构

```
fmri/
├── .venv/                          # Python 3.12 虚拟环境
├── scripts/                        # 核心脚本
│   ├── run_pipeline.py             # 主控脚本（一键运行）
│   ├── dicom/
│   │   └── dicom_convert.py        # DICOM → NIfTI 转换
│   ├── preprocess/
│   │   └── fmri_preprocess.py      # fMRI 预处理管线
│   └── analysis/
│       └── report_generator.py     # 质量评估与报告生成
├── data/                           # 原始数据（从 .gitignore 排除）
│   ├── sub_003/                    # fMRI BOLD DICOM
│   └── t1_original/                # T1 结构像 DICOM
├── output/                         # 处理输出（从 .gitignore 排除）
├── tests/                          # 测试套件
│   └── test_project.py             # 34 个综合测试
├── .gitignore
└── README.md
```

## 许可

本项目采用 [MIT 许可证](LICENSE) 开源。
