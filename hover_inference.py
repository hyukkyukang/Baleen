import os
import tqdm
import ujson
import argparse
import hkkang_utils.slack as slack_utils

from colbert.data import Queries
from colbert.infra import Run, RunConfig

from baleen.condenser.condense import Condenser
from baleen.hop_searcher import HopSearcher
from baleen.engine import Baleen

from colbert.utils.utils import print_message


def main(args):
    print_message("#> Starting...")

    collectionX_path = os.path.join(args.datadir, 'wiki.abstracts.2017/collection.json')
    queries_path = os.path.join(args.datadir, 'hover/dev/questions.tsv')

    checkpointL1 = os.path.join(args.datadir, 'hover.checkpoints-v1.0/condenserL1-v1.0.dnn')
    checkpointL2 = os.path.join(args.datadir, 'hover.checkpoints-v1.0/condenserL2-v1.0.dnn')

    with Run().context(RunConfig(root=args.root)):
        searcher = HopSearcher(index=args.index)
        condenser = Condenser(checkpointL1=checkpointL1, checkpointL2=checkpointL2,
                              collectionX_path=collectionX_path, deviceL1='cuda:0', deviceL2='cuda:0')

        baleen = Baleen(collectionX_path, searcher, condenser)
        baleen.searcher.configure(nprobe=2, ncandidates=8192)

    queries = Queries(path=queries_path)
    outputs = {}

    for qid, query in tqdm.tqdm(list(queries.items())):
        facts, pids_bag, _ = baleen.search(query, num_hops=4)
        outputs[qid] = (facts, pids_bag)

    # Convert set to list
    outputs = {key: (values[0], list(values[1])) for key, values in outputs.items()}

    with Run().open('hover_output.json', 'w') as f:
        f.write(ujson.dumps(outputs) + '\n')

    slack_utils.send_message(channel="question-answering", text=f"Baleen's Hover inference is Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--index", type=str, required=True)

    args = parser.parse_args()
    main(args)
