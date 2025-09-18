ENV_NAME="vllm-env"
VLLM_REPO_DIR="vllm"

FILE=ShareGPT_V3_unfiltered_cleaned_split.json
if [ ! -f "$FILE" ]; then
    echo "$FILE does not exist. Downloading..."
    if [ -z "$HF_TOKEN" ]; then
        echo "HF_TOKEN environment variable is not set. Please set it to your Hugging Face token."
        exit 1
    fi

    wget --header="Authorization: Bearer $HF_TOKEN" \
    https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
else
    echo "$FILE already exists."
fi

if [ ! -d "./$ENV_NAME" ]; then
    echo "You need to install vLLM first. Run install_vllm.sh"
    exit 1
fi
export PATH="$PWD/$ENV_NAME/bin:$PATH"
. $ENV_NAME/bin/activate
if [ ! -d "$VLLM_REPO_DIR" ]; then
    echo "You need to install vLLM first. Run install_vllm.sh"
    exit 1
fi

MODEL_NAME="${1:-meta-llama/Meta-Llama-3.1-8B-Instruct}"



vllm bench throughput  --enforce-eager --model "$MODEL_NAME" --tensor-parallel-size 1 --load-format dummy --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 50 --backend vllm