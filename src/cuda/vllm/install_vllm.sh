#!/bin/bash
# exit when any command fails
set -e

ENV_NAME="vllm-env"
VLLM_REPO_DIR="vllm"

if [ ! -d "./$ENV_NAME" ]; then
echo "create virtual environment"
python3 -m venv $ENV_NAME
fi
export PATH="$PWD/$ENV_NAME/bin:$PATH"
. $ENV_NAME/bin/activate 
pip install --upgrade pip 
pip install vllm 
pip install -U "huggingface_hub[cli]"
echo "[+] Verifying installation..."
python -c "import torch; print('Torch:', torch.__version__); import vllm; print('vLLM:', vllm.__version__)"
echo "[+] Cloning vllm repo and setting up sparse-checkout"
if [ ! -d "$VLLM_REPO_DIR" ]; then
    # Clone vllm repository with sparse-checkout
    git clone --depth 1 --filter=blob:none --sparse https://github.com/vllm-project/vllm.git
    cd vllm

    # Initialize sparse-checkout and set to only checkout 'benchmark' directory
    git sparse-checkout init --cone
    git sparse-checkout set benchmarks examples .buildkite

else
    echo "[=] Repo '$VLLM_REPO_DIR' already exists, skipping clone"
fi

echo "[+] Downloading the benchmark directory..."
cd ..

echo "[+] Deactivating virtual environment"
deactivate
echo "[✓] Setup complete. Virtual environment deactivated, and 'benchmark/' downloaded."
