# Huggingface example

Run huggingface models

## Loading gated model

Make sure to create a `.env` file in the folder with the following content:

```bash
HF_TOKEN=hf_...
```

## Setup environment

```
./setup_environment.sh
```

## Run example

```bash
./helloworld.py --model_name "openai-community/gpt2-large" --prompt "Hello World! In a galaxy far, far away..."
```