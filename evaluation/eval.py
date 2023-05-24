import argparse

import hkkang_utils.file as file_utils

from evaluation.utils import hover_evaluate_doc, hover_evaluate_sp

PRED_FILE_PATH = "/root/Baleen/experiments/default/hover_inference/2023-05/17/13.40.41/output.json"
DEV_FILE_PATH = "/root/Baleen/data/hover/dev/qas.json"

def main(data_path: str, pred_path: str, eval_type: str):
    print(f"Reading data from {data_path}...")
    dev_data = file_utils.read_jsonl_file(data_path)
    print(f"Reading predictions from {pred_path}...")
    pred_data = file_utils.read_json_file(pred_path)

    if eval_type == "sp":
        result = hover_evaluate_sp(dev_data, pred_data)
    elif eval_type == "doc":
        result = hover_evaluate_doc(dev_data, pred_data)
    else:
        raise ValueError(f"Invalid eval_type: {eval_type}")
    print(result)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_file", type=str, default=DEV_FILE_PATH)
    parser.add_argument("--pred_file", type=str, default=PRED_FILE_PATH)
    parser.add_argument("--eval_type", type=str, default="doc", choices=["sp", "doc"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.dev_file, args.pred_file, args.eval_type)