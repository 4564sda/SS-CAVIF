# Self-Supervised Video Stabilization: A Confidence-Aware Visual-Inertial Fusion Approach (CAVIF)

This repository contains the code for our paper:

**Self-Supervised Video Stabilization: A Confidence-Aware Visual-Inertial Fusion Approach**  
(Currently under review at *The Visual Computer*.)

本仓库包含论文 **Self-Supervised Video Stabilization: A Confidence-Aware Visual-Inertial Fusion Approach（CAVIF）** 的相关代码实现（目前投稿至 *The Visual Computer* 审稿中）。

> Note: The training pipeline, inference scripts, and pretrained models will be released after further整理与清理。  
> 注意：训练/推理脚本与训练好的模型后续整理完成后会发布。

---

## Repository Structure / 仓库结构

- `flow_CAVIF/`
  - Generate dense optical flow and confidence maps used by CAVIF.
  - 生成光流与置信图（confidence map），供后续视觉-惯性融合稳定使用。
- `gyrofield/`
  - Gyroscope-based motion field / rolling-shutter related processing utilities.
  - 基于陀螺仪的运动场（gyro field）相关处理与数据对齐工具。
- `CAVIF/`
  - `models/`: core implementation of our proposed CAVIF method (main network modules).
  - `losses/`: loss functions used during training.

---

## 1) Optical Flow + Confidence Map (`flow_CAVIF/`)

`flow_CAVIF` contains the code for generating **dense optical flow** and a **confidence map**.

### Origin / 原始代码来源

The flow/confidence estimation code is adapted from the excellent DenseMatching project:

- Original repository: https://github.com/PruneTruong/DenseMatching

In our repo, we only include **a single modified Python file** (instead of the full original project) to keep this repository lightweight.  
本仓库只保留了**一个被我修改过的核心 .py 文件**（未完整搬运原仓库），用于生成光流与置信图。

### What is modified? / 我修改了什么？

- We keep the original DenseMatching logic as the base.
- We make minimal modifications to better fit the CAVIF pipeline (e.g., I/O format, integration with later stages, naming, or output conventions).

> If you need the full DenseMatching environment, please refer to the upstream repository and follow its installation instructions.  
> 若需要完整依赖与训练/推理环境，请直接参考 DenseMatching 原仓库的说明。

### Output / 输出说明

Typical outputs include:
- **Optical flow** between adjacent frames (dense 2D motion field).
- **Confidence map** indicating the reliability of the estimated flow (used to weight visual measurements in fusion).

（具体输出文件命名与存储位置以该 .py 文件内的参数/保存逻辑为准。）

---

## 2) Gyro Field (`gyrofield/`)

`gyrofield` implements utilities for constructing motion fields from IMU gyroscope signals, aligning IMU timestamps to camera frames, and optionally handling rolling shutter effects.

### Entry point / 入口

- `gyrofield/main.py` is the entry point.  
  `main.py` 是入口文件。

### Configuration / 配置说明

The main configuration is defined in `gyrofield/config.py` (or an equivalent config file in the same folder).  
你提供的配置示例如下（字段含义说明见后）：

Parameters / 参数含义
K: Camera intrinsic matrix.
相机内参矩阵（应包含 fx, fy, cx, cy 等有效值；当前示例为占位）。
R_CAM_IMU: Rotation from IMU coordinate system to camera coordinate system.
IMU 坐标系到相机坐标系的旋转矩阵（用于将陀螺仪角速度/姿态增量转换到相机坐标）。
IMAGE_WIDTH, IMAGE_HEIGHT: Frame resolution.
图像分辨率。
IMU_TIME_OFFSET (seconds): Time shift between IMU timestamps and camera timestamps.
IMU 时间相对相机时间的偏移（单位秒）。正值表示 IMU 时间戳比相机晚。
READOUT_TIME (seconds): Rolling shutter readout time.
Rolling Shutter 行读出总时间。若为 None，一般表示按全局快门/忽略 rolling shutter 处理。
GYRO_SAMPLE_RATE (Hz): Used for basic data quality checks (optional).
陀螺仪采样率，用于验证数据质量或插值间隔合理性（可选）。
NUM_BLOCKS: Number of row blocks used to approximate rolling shutter warping.
将图像按行分块（block）数量，用于近似 rolling shutter 下的逐行时间差/逐块变形建模。
DATA_ROOT: Root directory of the dataset.
数据集根目录。
SCENES: Scene list / mapping. You can register different sequences here.
场景配置（可在此登记不同序列/不同数据路径）。
OUTPUT_DIR: Output directory for generated gyro fields or intermediate results.
输出目录。
3) CAVIF Core (CAVIF/)
CAVIF/models/
This folder contains the core model code of our proposed CAVIF approach, including modules for confidence-aware visual-inertial fusion and stabilization-related networks.

该目录是你提出方法的核心实现代码，包含 CAVIF 的关键网络结构与融合模块。

CAVIF/losses/
This folder implements the training losses used by CAVIF (e.g., self-supervised consistency losses, smoothness, confidence-related weighting, etc.).
该目录实现训练阶段使用的损失函数（例如自监督一致性约束、平滑项、置信权重相关损失等）。

Training scripts and pretrained checkpoints will be provided in a future update.
训练脚本与预训练模型将在后续整理后发布。

Installation / 安装
This repository is under active preparation. At this stage, please:

Prepare your Python environment.
Install required dependencies according to your local setup and the upstream DenseMatching requirements if you plan to run flow_CAVIF.
当前仓库仍在整理中。现阶段建议：

准备 Python 环境；
若要运行 flow_CAVIF，请参考 DenseMatching 原仓库安装依赖。
A requirements.txt / environment file will be added later when the training/inference code is finalized.
等训练/推理代码整理完后会补充 requirements.txt 或环境配置文件。

Data / 数据准备
Put your dataset under DATA_ROOT (default: new_data).
Register sequences in SCENES in the gyrofield config if needed.
将数据放到 DATA_ROOT 下（默认 new_data），并按需要在 SCENES 中登记序列信息。

How to Run (partial) / 运行方式（当前阶段）
Gyro field generation / 生成 gyro field
Example (to be adapted to your actual file names):

python gyrofield/main.py
Flow + confidence generation / 生成光流与置信图
Run the modified script inside flow_CAVIF/ (adapt arguments to your setup).
在 flow_CAVIF/ 中运行对应的修改脚本（根据你代码中的参数进行调整）。

Citation / 引用
If you find this repository useful, please cite our paper (BibTeX will be provided after acceptance / after the preprint is public).
如对你有帮助，欢迎引用本文（接收或公开预印本后补充 BibTeX）。

Acknowledgements / 致谢
DenseMatching (optical flow & confidence estimation base): https://github.com/PruneTruong/DenseMatching
