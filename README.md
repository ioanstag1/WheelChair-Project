# WheelchairPose

This repository contains the code for pose estimation, feature extraction, and classification
for impairment level assessment in wheelchair users.

## Structure

- `Code/pose_estimation/` – scripts for YOLO / ViTPose / SpinePose
- `Code/feature_extraction/` – ADL1, ADL2, ADL3 biomechanical feature scripts
- `Code/classification/` – machine-learning pipelines and evaluation

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

