import argparse
import csv
import os
import warnings
from datetime import timedelta

import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import read_file

os.system(f"cp {__file__} ~/backup/")  # optionally backup the script
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from torch.distributed.elastic.multiprocessing.errors import record


class CSVTextDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        # assert text is in the columns
        assert "text" in self.df.columns, "text column not found in the csv file"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.df):
            raise IndexError
        return self.df.iloc[idx]

    def set_rank_and_world_size(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.data_per_gpu = len(self) // world_size
        self.start_index = rank * self.data_per_gpu
        self.end_index = (rank + 1) * self.data_per_gpu if rank != world_size - 1 else len(self)
        self.df = self.df.iloc[self.start_index : self.end_index]

    def write_to_csv(self, output_file, data, new_key):
        """write the part of the df to a csv file corresponding to the rank and write self.data_list as a new column"""
        writer = csv.writer(open(output_file, "w"))
        columns = self.df.columns + [new_key]
        writer.writerow(columns)
        for index, row in self.df.iterrows():
            if index < self.start_index or index >= self.end_index:
                continue
            writer.writerow([*row, data[index - self.start_index]])
        writer.close()


def pad_left(sequences, padding_value=0):
    # Determine the maximum length of the sequences
    max_len = max([s.size(0) for s in sequences])
    # Create a list to hold the padded sequences
    padded_sequences = []
    for sequence in sequences:
        # Calculate the number of padding elements needed for this sequence
        num_padding = max_len - sequence.size(0)
        # Create a tensor of padding values
        padding = torch.full((num_padding,), padding_value, dtype=sequence.dtype).to(sequence.device)
        # Concatenate the padding and the sequence to pad on the left
        padded_sequence = torch.cat([padding, sequence], dim=0)
        padded_sequences.append(padded_sequence)
    # Stack the padded sequences into a batch
    batch = torch.stack(padded_sequences)
    return batch


@record
def main(args):
    # ======================================================
    # 1. init environment
    # ======================================================
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

    # ======================================================
    # 2. Prep rank-wise dataloader
    # ======================================================
    dataframe = read_file(args.input)
    print("read data from {}".format(args.input))
    dataset = CSVTextDataset(args.input)
    dataset.set_rank_and_world_size(dist.get_rank(), dist.get_world_size())

    import os

    if os.getenv("DEBUG_ADDRESS") != None and dist.get_rank() == 2:
        import ptvsd

        print("waiting for debugger attachment")
        ptvsd.enable_attach(address=("localhost", int(os.getenv("DEBUG_ADDRESS"))), redirect_output=True)
        ptvsd.wait_for_attach()

    output_file = args.output_prefix + f"_rank{dist.get_rank()}" + f"_{args.key}.csv"
    output_file_handle = open(output_file, "w")
    writer = csv.writer(output_file_handle)
    columns = list(dataframe.columns) + [args.key]

    writer.writerow(columns)

    # add a new key named summary, write in csv file
    print("the processed data saved on this rank will be saved to {}".format(output_file))

    def collate_fn(batch):
        return batch

    dataloader = torch.utils.data.DataLoader(
        dataset,
        # num_workers=2,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )

    # ======================================================
    # 2. process using llama3 and prompt
    # ======================================================

    print("Using model with the id {}".format(args.model_id))
    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=dist.get_rank() % torch.cuda.device_count(),
    )
    # .to(dist.get_rank() % torch.cuda.device_count())
    dist.barrier()
    print("======== Process data using LLAMA3 ========")

    def extract_batch(texts, prompt):
        input_ids_list = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": prompt}, {"role": "user", "content": text}],
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)[0]
            for text in texts
        ]

        attention_mask_list = [
            torch.ones(input_ids.shape, dtype=torch.long, device=model.device) for input_ids in input_ids_list
        ]

        # input_ids_batch = pad_left(
        #     input_ids_list, padding_value=tokenizer.eos_token_id
        # )

        input_ids_batch = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=tokenizer.eos_token_id
        )

        attention_mask_batch = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

        # attention_mask_batch = pad_left(
        #     attention_mask_list, padding_value=0
        # )

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        outputs = model.generate(
            input_ids_batch,
            max_new_tokens=512,
            attention_mask=attention_mask_batch,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=terminators,
            # do_sample=True,
            # temperature=0.6,
            # top_p=0.9,
        )

        responses = []
        for i in range(len(texts)):
            response = outputs[i][input_ids_list[i].shape[-1] :]
            response = tokenizer.decode(response, skip_special_tokens=True)
            responses.append(response)

        return responses

    print("Processing starting...")
    if args.prompt == "" and args.key == "objects":
        prompt = (
            "You are a AI assistant to extract objects from user's text. "
            "For example: user: 'In this video a dog is running around. In addition, a person is laughing at the dog.', you produce a list of objects separated by ',' and wrapped by '[' and ']': '[dog, person]' "
        )
    elif args.prompt == "" and args.key == "actions":
        prompt = (
            "You are a AI assistant to extract actions from user's text. "
            "For example: user: 'In this video a dog is running around. In addition, a person is laughing at the dog.', you produce a list of actions separated by ',' and wrapped by '[' and ']': '[run, laugh]' "
        )
    else:
        prompt = args.prompt

    print("Prompt: {}".format(prompt))

    args.batch_size
    # for i in tqdm(range(0, len(dataframe), batch_size)):
    for _, batch in enumerate(tqdm(dataloader)):
        # get the text column from the batch
        texts = [batch[i]["text"] for i in range(len(batch))]
        list_keywords = extract_batch(texts, prompt)

        for idx, keywords in enumerate(list_keywords):
            try:
                keywords_start = keywords.find("[")
                keywords_end = keywords.find("]")
                keywords = keywords[keywords_start + 1 : keywords_end]
                if (
                    "\n" in keywords or len(keywords.strip()) == 0
                ):  # we empirically observe that it produces newlines when no keywords are found
                    keywords = "NONE_FOUND"
            except:
                keywords = "NONE_FOUND"
            row = batch[idx]
            writer.writerow([*row, keywords])

    output_file_handle.close()
    dist.barrier()

    if dist.get_rank() == 0:
        collated_file = args.output_prefix + f"_{args.key}.csv"
        print("All ranks are finished. Collating the processed data to {}".format(collated_file))
        import pandas as pd

        csv_files = [args.output_prefix + f"_rank{i}" + f"_{args.key}.csv" for i in range(dist.get_world_size())]
        # List to hold DataFrames
        dataframes = []
        # Read each CSV into a DataFrame and append to list
        for file in csv_files:
            df = pd.read_csv(file)
            # scan each line in the df, if the ``key`` column is NaN, replace it with "NONE_FOUND"
            df[args.key] = df[args.key].fillna("NONE_FOUND")
            dataframes.append(df)
        # Concatenate all DataFrames
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Save the combined DataFrame to a new CSV file
        combined_df.to_csv(collated_file, index=False)
        print("Collated data saved to {}".format(collated_file))
    # terminate distributed env
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("input", type=str, help="Path to the input CSV file")
    parser.add_argument("--output_prefix", type=str, help="Path to the output CSV file")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--key", type=str)
    args = parser.parse_args()

    main(args)
