# Common flags for running huggingface examples
import argparse

def getCommonParser(is_parent=True):
    # If is_parent is True, don't add the help flag to avoid conflict
    parser = argparse.ArgumentParser(add_help=not is_parent)
    parser.add_argument("--model_name", type=str, default="openai-community/gpt2-large")
    parser.add_argument("--device", type=str, default="cuda")
    return parser