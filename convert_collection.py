import tqdm
import hkkang_utils.file as file_utils

src_path = "/root/Baleen/ckpts/baleen/enwiki-20171001-one_passage/collection.jsonl"
dst_path = "/root/Baleen/ckpts/baleen/enwiki-20171001-one_passage/collection2.jsonl"

src = file_utils.read_jsonl_file(src_path)

for datum in tqdm.tqdm(src):
    datum["text"] = [" ".join(datum["text"])]
    
file_utils.write_jsonl_file(src, dst_path)