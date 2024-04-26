import argparse
import base64
import csv
import os
from io import BytesIO

import requests
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .utils import PROMPTS, VideoTextDataset, read_file
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(args):
    # ======================================================
    # 1. read video list
    # ======================================================
    dataframe = read_file(args.input)
    print("read data from {}".format(args.input))

    output_file = os.path.splitext(args.input)[0] + "_llama3.csv"
    f = open(output_file, "w")
    writer = csv.writer(f)

    columns = dataframe.columns + [args.key]
    writer.writerow(columns)

    # add a new key named summary, write in csv file
    print("the processed data saved to {}".format(output_file))

    # ======================================================
    # 2. process using llama3 and prompt
    # ======================================================

    print("Using model with the id {}".format(args.model_id))
    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("======== Process data using LLAMA3 ========")

    def extract(text, prompt):
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        # The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
        # Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
        terminators = [tokenizer.eos_token_id,
                       tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                       ]

        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)

        outputs = model.generate(
            input_ids,
            max_new_tokens=512,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        response = tokenizer.decode(response, skip_special_tokens=True)
        return response

    print("Processing starting...")
    if args.prompt == "":
        prompt = ("You are a AI assistant to extract user's text into keywords. "
                  "For example: user: 'a woman on a news station talking about traffic.', you just need a list of keywords separate by , and covered by '[' and ']': '[woman, news station, traffic]' ")
    else:
        prompt = args.prompt

    print("Prompt: {}".format(prompt))
    key = args.key
    # prompt to process text, first input prompt to AI assistant, second input is the text to process
    # batch process, row['text'] to key extraction, for example: a woman on a news station talking about traffic. -> woman, news station, traffic

    length = len(dataframe)
    for index, row in dataframe.iterrows():
        row_text = row['text']

        # process text
        keywords = extract(row_text, prompt)

        # process keywords
        keywords_start = keywords.find("[")
        keywords_end = keywords.find("]")
        keywords = keywords[keywords_start+1:keywords_end]

        if index % 100 == 0:
            print("{}/{}".format(index, length))
            print(f"text: {row_text} "
                  f"keywords: {keywords}")

        # add a new key named summary, write in csv file
        writer.writerow([*row, keywords])

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="/home/data/models/meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("input", type=str, help="Path to the input CSV file")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument('--key', default='summary', type=str)
    args = parser.parse_args()

    main(args)
