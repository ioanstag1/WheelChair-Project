#!/bin/bash
set -e

echo "=== Creating virtual environment ==="
mkdir -p ~/WheelchairPose
cd ~/WheelchairPose
python3 -m venv .venv
source .venv/bin/activate

echo "=== Upgrading pip ==="
pip install --upgrade pip

echo "=== Installing PyTorch (CUDA 12.1) ==="
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

echo "=== Installing system dependencies ==="
sudo apt-get update -y
sudo apt-get install -y ffmpeg

echo "=== Cloning ViTPose-enhanced Transformers fork ==="
git clone -b add_vitpose_autobackbone https://github.com/nielsrogge/transformers.git
pip install -e ./transformers

echo "=== Installing Python dependencies ==="
pip install ultralytics supervision deep_sort_realtime accelerate safetensors sympy
pip install opencv-python pillow matplotlib pandas numpy scipy hf_transfer

echo "=== Installing SpinePose (GPU ONNX) ==="
pip uninstall -y onnxruntime onnxruntime-gpu || true
pip install onnxruntime-gpu
pip install --no-deps spinepose

echo "=== Testing installation ==="
python3 - << 'EOF'
import torch
from ultralytics import YOLO
import onnxruntime as ort

print("CUDA available:", torch.cuda.is_available())
print("ONNX providers:", ort.get_available_providers())

m = YOLO("yolo11n.pt")
print("Ultralytics OK")

from spinepose import SpinePoseEstimator
sp = SpinePoseEstimator(device="cuda")
print("SpinePose OK")
EOF

echo "=== Setup complete ==="
