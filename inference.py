import argparse
import logging
import os
import pickle
from typing import *

import hkkang_utils.pattern as pattern_utils
import hkkang_utils.time as time_utils
import redis
import torch
import tqdm

from api.data_types import DocumentWithScore, RetrievalResult
from baleen.condenser.condense import Condenser
from baleen.engine import Baleen
from baleen.hop_searcher import HopSearcher
from colbert.infra import Run, RunConfig
from config import config

DEFAULT_ROOT = os.path.join(config.base_dir, config.baleen.experiment_dir)
# Checkpoints
HOTPOTQA_FLIPR_CHECKPOINT_PATH = os.path.join(
    config.base_dir, config.baleen.checkpoint.hotpotqa.flipr
)
HOTPOTQA_L1_CHECKPOINT_PATH = os.path.join(
    config.base_dir, config.baleen.checkpoint.hotpotqa.L1
)
HOTPOTQA_L2_CHECKPOINT_PATH = os.path.join(
    config.base_dir, config.baleen.checkpoint.hotpotqa.L2
)
# Index
WIKI_2017_INDEX = os.path.join(
    config.base_dir, config.baleen.wiki2017.base_dir, config.baleen.wiki2017.index
)
WIKI_2020_INDEX = os.path.join(
    config.base_dir, config.baleen.wiki2017.base_dir, config.baleen.wiki2020.index
)
# Wiki Collection
WIKI_2017_COLLECTION = os.path.join(
    config.base_dir, config.baleen.wiki2017.base_dir, config.baleen.wiki2017.collection
)
WIKI_2020_COLLECTION = os.path.join(
    config.base_dir, config.baleen.wiki2017.base_dir, config.baleen.wiki2020.collection
)

logger = logging.getLogger("BaleenRetriever")


rd = redis.StrictRedis(
    host=config.redis.host, port=config.redis.port, db=config.redis.db_id
)

class Retriever(metaclass=pattern_utils.SingletonABCMetaWithArgs):
    def __init__(self, wiki_version: str):
        torch.set_default_device("cuda:0")
        assert wiki_version in ["2017", "2020"], f"Invalid wiki version: {wiki_version}"
        self.root = DEFAULT_ROOT
        self.wiki_version=wiki_version
        self.index = WIKI_2017_INDEX if wiki_version == "2017" else WIKI_2020_INDEX
        self.collection_path = (
            WIKI_2017_COLLECTION if wiki_version == "2017" else WIKI_2020_COLLECTION
        )
        self.checkpoint_flipr = HOTPOTQA_FLIPR_CHECKPOINT_PATH
        self.checkpoint_l1 = HOTPOTQA_L1_CHECKPOINT_PATH
        self.checkpoint_l2 = HOTPOTQA_L2_CHECKPOINT_PATH
        self.baleen = None
        self.return_num_keys = [10, 100, 1000, 2000]
        self.rd=rd
        self.__post_init__()

    def __post_init__(self):
        logger.info("Initializing BaleenRetriever...")
        with Run().context(RunConfig(root=self.root)):
            searcher = HopSearcher(index=self.index, checkpoint=self.checkpoint_flipr)
            condenser = Condenser(
                checkpointL1=self.checkpoint_l1,
                checkpointL2=self.checkpoint_l2,
                collectionX_path=self.collection_path,
                deviceL1="cuda:0",
                deviceL2="cuda:0",
            )

            self.baleen = Baleen(self.collection_path, searcher, condenser)
            self.baleen.searcher.configure(nprobe=2, ncandidates=8192)

    @time_utils.measure_time
    def __call__(
        self, query_or_queries: Union[List[str], str], return_num: int
    ) -> Union[RetrievalResult, List[RetrievalResult]]:
        # Navigate to proper function
        if type(query_or_queries) == str:
            return self.search(query=query_or_queries, return_num=return_num)
        elif type(query_or_queries) == list:
            assert type(query_or_queries[0]) == str, "Questions must be list of str"
            return self.search_batch_with_cache(queries=query_or_queries, return_num=return_num)
        raise ValueError(
            f"input must be str or list of str, but got {type(query_or_queries)}"
        )

    def _create_cache_key(self, query: str, return_num: int) -> str:
        return_num_key = self.return_num_keys[-1]
        for key in self.return_num_keys:
            if return_num <= key:
                return_num_key = key
                break
        return f"{self.wiki_version}_{query}_{return_num_key}"

    def _to_doc(
        self, doc_id: int, sent_id: int, score: Optional[int]
    ) -> DocumentWithScore:
        doc_info = self.baleen.collectionX[(doc_id, sent_id)]
        split_idx = doc_info.index(" | ")
        title = doc_info[:split_idx]
        text = doc_info[split_idx + 3 :]
        return DocumentWithScore(title=title, text=text, score=score)

    def search(self, query: str, return_num: int) -> RetrievalResult:
        return self.search_batch_with_cache(queries=[query], return_num=return_num)[0]

    def search_batch_with_cache(self, queries: List[str], return_num: int) -> List[RetrievalResult]:
        # Create cache keys
        cache_keys = [self._create_cache_key(query, return_num) for query in queries]
        results = {idx: None for idx in range(len(queries))}
        for idx, cache_key in enumerate(cache_keys):
            if self.rd.exists(cache_key):
                results[idx] = pickle.loads(self.rd.get(cache_key))
        target_indices = []
        target_queries = []
        for idx, result in results.items():
            if result is None:
                target_indices.append(idx)
                target_queries.append(queries[idx])
        # Search
        target_results = self.search_batch(queries=target_queries, return_num=return_num)
        # Update cache
        for target_idx, result in zip(target_indices, target_results):
            self.rd.set(cache_keys[idx], pickle.dumps(result.dic))
            results[target_idx] = result.dic
        # Get results
        retrieval_results = [RetrievalResult.from_dict(data=result) for result in results.values()]
        # Filter return num
        for retrieval_result in retrieval_results:
            retrieval_result.documents_with_score = retrieval_result.documents_with_score[:return_num]
        # Return
        return retrieval_results

    def search_batch(
        self, queries: List[str], return_num: int
    ) -> List[RetrievalResult]:
        all_results = []
        for query in tqdm.tqdm(queries, desc="Retrieving"):
            facts, pids_bag, more_facts = self.baleen.search(query, num_hops=2)
            # facts to document
            docs = []
            all_facts = facts + more_facts[2:]
            docs = [
                self._to_doc(
                    doc_id=all_facts[rank_idx][0],
                    sent_id=all_facts[rank_idx][1],
                    score=rank_idx,
                )
                for rank_idx in range(min(len(all_facts), return_num))
            ]
            all_results.append(RetrievalResult(query=query, documents_with_score=docs))

        return all_results


def main(wiki_version: str, query: str):
    retriever = Retriever(wiki_version=wiki_version)
    result = retriever(query, return_num=10)
    for doc in result.documents_with_score:
        logger.info(f"Title: {doc.title}")
        logger.info(f"Text: {doc.text}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wiki-version", type=str, default="2017", choices=["2017", "2020"]
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Which genus contains more species, Ortegocactus or Eschscholzia?",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )

    args = parse_args()

    main(wiki_version=args.wiki_version, query=args.query)

    logger.info("Done!")
