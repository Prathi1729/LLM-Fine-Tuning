mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

conda create --name unsloth_env python=3.11 
conda activate unsloth_env

pip install unsloth

sudo apt install apt install python3.10-venv python3.11-venv python3.12-venv python3.13-venv -y

# Pull the optimized ROCm Docker image
docker pull rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_2.3.0

# Inside the container, install Unsloth (ROCm branch)
pip install "unsloth[rocm] @ git+https://github.com/unslothai/unsloth.git"

# 1. Update your system and install ROCm requirements
sudo apt-get update && sudo apt-get install -y rocm-libs
curl -fsSL https://ollama.com/install.sh | sh   

# 3. Install PDF processing tool
pip install pymupdf4llm
pip install --upgrade torch==2.8.0 pytorch-triton-rocm torchvision torchaudio torchao==0.13.0 xformers --index-url https://download.pytorch.org/whl/rocm6.4

pip install --no-deps unsloth unsloth-zoo
pip install --no-deps git+https://github.com/unslothai/unsloth-zoo.git
pip install "unsloth[amd] @ git+https://github.com/unslothai/unsloth"


pip install numpy

pip install huggingface_hub

pip install unsloth_zoo

pip install sentencepiece gguf

cd /root/LLM-Fine-Tuning/llama.cpp
# Remove the cache files that were accidentally written to the root
rm -rf CMakeCache.txt CMakeFiles/

git clone --recursive https://github.com/ggerganov/llama.cpp
cd llama.cpp

# 1. Configure (Point to the 'build' directory explicitly)
HIPCXX="$(hipconfig -l)/clang" \
HIP_PATH="$(hipconfig -R)" \
cmake -S . -B build \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS=gfx942 \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIPBLAS=ON \
    -DLLAMA_CURL=OFF

# 2. Compile (Target the 'build' directory)
cmake --build build --config Release -j $(nproc)

make clean && make -j


pip uninstall unsloth unsloth-zoo triton -y
pip install --no-cache-dir "unsloth[rocm] @ git+https://github.com/unslothai/unsloth.git"

export TRITON_HIP_LLD_PATH=/opt/rocm/llvm/bin/ld.lld