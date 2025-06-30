# Hello world for huggingface
import argparse
import transformers
from dotenv import load_dotenv

from util import getCommonParser

def main():
    load_dotenv()
    
    helloworldParser = argparse.ArgumentParser(parents=[getCommonParser()])
    helloworldParser.add_argument("--prompt", type=str, default="Hello World! In a galaxy far, far away...")
    helloworldParser.add_argument("--max_length", type=int, default=256)
    args = helloworldParser.parse_args()
    
    helloworld_str = f"Running Hello world for \"{args.model_name}\" with prompt: \"{args.prompt}\""
    print(helloworld_str)
    print("=" * len(helloworld_str))
    
    pipe = transformers.pipeline(task="text-generation", model=args.model_name, device=args.device)
    results = pipe(args.prompt, max_length=args.max_length, truncation=True)
    for result in results:
        print(result["generated_text"])

if __name__ == "__main__":
    main()