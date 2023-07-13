import argparse
import json
from tqdm import tqdm


def format_example(example: dict) -> dict:
    context = f"Question: {example['instruction'].strip()}\n\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/test.json")
    parser.add_argument("--save_path", type=str, default="data/test.jsonl")

    args = parser.parse_args()
    with open(args.data_path,encoding='UTF-8') as f:
        examples = json.load(f)

    with open(args.save_path, 'w') as f:
        for example in tqdm(examples, desc="formatting.."):
            #f.write(json.dumps(format_example(example),indent=4, ensure_ascii=False) + '\n')
            f.write(json.dumps(format_example(example)) + '\n')


if __name__ == "__main__":
    main()
