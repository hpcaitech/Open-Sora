import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_path", type=str)
    parser.add_argument("--ref_path", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    return args


args = parse_args()

with open(args.text_path, "r", encoding="utf-8") as text_f, open(args.ref_path, "r", encoding="utf-8") as ref_f, open(
    args.output, "w", encoding="utf-8"
) as f:
    texts = text_f.readlines()
    refs = ref_f.readlines()
    assert len(texts[:25]) == len(refs[:25])

    for text, ref in zip(texts, refs):
        text = text.strip()
        ref = ref.strip()
        formated_prompt = text + '{"reference_path": "' + ref + '"}'
        f.write(formated_prompt + "\n")
