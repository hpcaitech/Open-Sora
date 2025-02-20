import argparse

import pandas as pd

parser = argparse.ArgumentParser(description="Convert CSV file to txt file")
parser.add_argument("csv_file", type=str, help="CSV file to convert")
parser.add_argument("txt_file", type=str, help="TXT file to save")
args = parser.parse_args()

data = pd.read_csv(args.csv_file)
text = data["text"].to_list()
text = "\n".join(text)
with open(args.txt_file, "w") as f:
    f.write(text)
