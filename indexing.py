import argparse
import logging
import os

import hkkang_utils.slack as slack_utils

from colbert import Indexer
from colbert.infra import ColBERTConfig, Run, RunConfig
from config import config

DEFAULT_NBITS = 2
DEFAULT_ROOT = os.path.join(config.base_dir, config.baleen.experiment_dir)
HOTPOTQA_CHECKPOINT_PATH = os.path.join(
    config.base_dir, config.baleen.checkpoint.hotpotqa.flipr
)
# Wiki Collection
WIKI_2017_COLLECTION = os.path.join(
    config.base_dir,
    config.colbert.wiki2017.base_dir,
    config.colbert.wiki2017.collection,
)
WIKI_2020_COLLECTION = os.path.join(
    config.base_dir,
    config.colbert.wiki2020.base_dir,
    config.colbert.wiki2020.collection,
)
# Index
WIKI_2017_INDEX = os.path.join(
    config.base_dir, config.baleen.wiki2017.base_dir, config.baleen.wiki2017.index
)
WIKI_2020_INDEX = os.path.join(
    config.base_dir, config.baleen.wiki2020.base_dir, config.baleen.wiki2020.index
)

logger = logging.getLogger("BaleenIndexer")


def index(collection_path: str, index_name: str, root: str, nbits: int = 2):
    with Run().context(RunConfig(root=root)):
        config = ColBERTConfig(doc_maxlen=256, nbits=nbits)
        indexer = Indexer(HOTPOTQA_CHECKPOINT_PATH, config=config)
        indexer.index(name=index_name, collection=collection_path)


def main(args):
    # Determine which collection to use
    if args.wiki_version == "2017":
        collection_path = WIKI_2017_COLLECTION
        index_name = WIKI_2017_INDEX
    elif args.wiki_version == "2020":
        collection_path = WIKI_2020_COLLECTION
        index_name = WIKI_2020_INDEX
    else:
        raise ValueError(f"Invalid wiki version: {args.wiki_version}")
    logger.info(f"Indexing {collection_path} to {index_name}")
    index(
        collection_path=collection_path,
        index_name=index_name,
        root=DEFAULT_ROOT,
        nbits=DEFAULT_NBITS,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wiki-version", type=str, required=True, choices=["2017", "2020"]
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )

    args = parse_args()

    with slack_utils.notification(
        channel="question-answering",
        success_msg=f"Baleen indexing {args.wiki_version} wiki collection is done!",
        error_msg=f"Baleen indexing {args.wiki_version} wiki collection is failed!",
    ):
        main(args)

    logger.info("Done!")
