# Self-Supervised Video Stabilization: A Confidence-Aware Visual-Inertial Fusion Approach (CAVIF)

This repository contains the code for our paper:

**Self-Supervised Video Stabilization: A Confidence-Aware Visual-Inertial Fusion Approach**  
(Currently under review at *The Visual Computer*.)

> Note: The training pipeline, inference scripts, and pretrained models will be released after further cleanup and organization.

---

## Repository Structure

- `flow_CAVIF/`
  - Code for generating dense optical flow and confidence maps used by CAVIF.
- `gyrofield/`
  - Gyroscope-based motion field utilities, including IMU-to-camera timestamp alignment and optional rolling-shutter handling.
- `CAVIF/`
  - `models/`: core implementation of our proposed CAVIF method (main network modules).
  - `losses/`: implementations of the training losses.

---

## 1) Optical Flow + Confidence Map (`flow_CAVIF/`)

`flow_CAVIF` contains code for generating **dense optical flow** and a corresponding **confidence map**.

### Origin

The flow/confidence estimation code is adapted from the DenseMatching project:

- Upstream repository: https://github.com/PruneTruong/DenseMatching

To keep this repository lightweight, we only include **one modified Python file** (instead of the full upstream codebase). The included file is a minimal adaptation of the original implementation for better integration with the CAVIF pipeline.

### What is modified?

- The upstream DenseMatching logic is used as the base.
- Minimal changes were made to better match the CAVIF workflow (e.g., input/output conventions, integration hooks, or output formatting).

> If you need the full DenseMatching environment or additional utilities, please refer to the upstream repository and follow its installation instructions.

### Output

Typical outputs include:
- **Optical flow** between adjacent frames (dense 2D motion field).
- **Confidence map** that indicates the reliability of the estimated flow (used to weight visual measurements in fusion).

(Exact file naming and output directories follow the logic defined in the included Python script.)

---

## 2) Gyro Field (`gyrofield/`)

`gyrofield` provides utilities for constructing motion fields from IMU gyroscope signals, aligning IMU timestamps to camera frames, and optionally modeling rolling shutter effects.

### Entry Point

- `gyrofield/main.py` is the entry point.

### Configuration

Configuration is defined in `gyrofield/config.py` (or an equivalent config file in the same folder). Typical configuration includes:

- `K`: camera intrinsic matrix.
- `R_CAM_IMU`: rotation matrix from the IMU coordinate frame to the camera coordinate frame.
- `IMAGE_WIDTH`, `IMAGE_HEIGHT`: image resolution.
- `IMU_TIME_OFFSET`: time offset between IMU timestamps and camera timestamps (seconds).
- `READOUT_TIME`: rolling shutter readout time (seconds). If `None`, rolling shutter is ignored.
- `GYRO_SAMPLE_RATE`: gyroscope sampling rate (Hz), used for optional data validation.
- `NUM_BLOCKS`: number of row blocks used to approximate rolling-shutter warping.
- `DATA_ROOT`: dataset root directory.
- `SCENES`: scene/sequence registry (configure sequences here).
- `OUTPUT_DIR`: output directory for generated gyro fields and intermediate results.

---

## 3) CAVIF Core (`CAVIF/`)

### `CAVIF/models/`

This folder contains the **core model code** of our proposed CAVIF approach, including modules for confidence-aware visual-inertial fusion and stabilization-related networks.

### `CAVIF/losses/`

This folder implements the training losses used by CAVIF (e.g., self-supervised consistency losses, smoothness regularization, and confidence-weighted objectives).

> Training scripts and pretrained checkpoints will be provided in a future update.

---

## Installation

This repository is under active preparation. At this stage:

1. Prepare your Python environment.
2. Install required dependencies according to your local setup.
3. If you plan to run `flow_CAVIF`, follow the dependency requirements of the upstream DenseMatching repository.

> A `requirements.txt` (or an environment file) will be added later when the training and inference code is finalized.

---

## Data

- Place your dataset under `DATA_ROOT` (default: `new_data`).
- Register sequences in `SCENES` in the gyrofield config if needed.

---

## How to Run (Partial)

### Gyro field generation

```bash
python gyrofield/main.py
