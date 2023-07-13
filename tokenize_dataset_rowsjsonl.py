import argparse
import json
from tqdm import tqdm

import datasets
import transformers

from ringrwkv.rwkv_tokenizer import TRIE_TOKENIZER

#将jsonl转换为数据集文件夹


def preprocess(tokenizer, example):
    #print(example["context"])
    #print(example["target"])
    prompt = example["context"]
    #print(prompt)
    target = example["target"]
    #print(target)
    prompt_ids = tokenizer.encode(prompt)
    target_ids = tokenizer.encode(target)
    input_ids = prompt_ids + target_ids
    #print(input_ids)
    print(tokenizer.decode(prompt_ids))
    #print(tokenizer.decode(target_ids))
    #print(tokenizer.decode(input_ids))
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}


def read_jsonl(path, max_seq_length, skip_overlength=False):
    tokenizer = TRIE_TOKENIZER('./ringrwkv/rwkv_vocab_v20230424.txt')
    with open(path, "r") as f:
        for line in tqdm(f.readlines()):
            example = json.loads(line)
            feature = preprocess(tokenizer, example)
            if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                continue
            feature["input_ids"] = feature["input_ids"][:max_seq_length]
            feature["seq_len"] =  feature["seq_len"]
            yield feature
    return feature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, default="data/test.jsonl")
    parser.add_argument("--save_path", type=str, default="data/test")
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--skip_overlength", type=bool, default=False)
    args = parser.parse_args()

    dataset = datasets.Dataset.from_generator(
        lambda: read_jsonl(args.jsonl_path, args.max_seq_length, args.skip_overlength)
    )
    #print(dataset)
    dataset.save_to_disk(args.save_path)


if __name__ == "__main__":
    main()
