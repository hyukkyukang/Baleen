import argparse
import copy

import hkkang_utils.file as file_utils

from evaluation.utils import hover_evaluate_doc, hover_evaluate_sp

PRED_FILE_PATH = "/root/Baleen/experiments/default/hover_inference/2023-05/17/13.40.41/output.json"
DEV_FILE_PATH = "/root/Baleen/data/hover/dev/qas.json"

def get_two_doc_ids(preds):
    # at most (or rather exactly) two PIDs
    pred_docs = dict()
    for key, pred in preds.items():
        sp_facts = pred[0]
        unique_doc_ids = []
        for sp_fact in sp_facts:
            doc_id, sent_id = sp_fact
            if doc_id not in unique_doc_ids:
                unique_doc_ids.append(doc_id)
        # If less than 2 documents, add documents from sp facts rank higher than 2 
        aux_doc_ids = copy.deepcopy(pred[1])
        while len(unique_doc_ids) < 2:
            unique_doc_ids.append(aux_doc_ids.pop(0))
        pred_docs[key] = unique_doc_ids[:2]
    return pred_docs
        
    if len(set([pid for pid, _ in preds])) > 2:
        first_two_pids = f7([pid for pid, _ in preds])[:2]
        preds = [(pid, sid) for pid, sid in preds if pid in first_two_pids]
    return preds

def main(data_path: str, pred_path: str, eval_type: str):
    print(f"Reading data from {data_path}...")
    dev_data = file_utils.read_jsonl_file(data_path)
    print(f"Reading predictions from {pred_path}...")
    pred_data = file_utils.read_json_file(pred_path)
    # Extract only the first two PIDs
    pred_data = get_two_doc_ids(pred_data)
    

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