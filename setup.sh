# Pull the optimized ROCm Docker image
docker pull rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_2.3.0

# Inside the container, install Unsloth (ROCm branch)
pip install "unsloth[rocm] @ git+https://github.com/unslothai/unsloth.git"

# 1. Update your system and install ROCm requirements
sudo apt-get update && sudo apt-get install -y rocm-libs

# 2. Install Unsloth for AMD (2026 Nightly)
pip install "unsloth[rocm] @ git+https://github.com/unslothai/unsloth.git"

# 3. Install PDF processing tool
pip install pymupdf4llm


