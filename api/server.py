import logging
import pickle
from typing import *
from typing import List, Union

from flask import Flask, request
from flask_cors import CORS
from waitress import serve

from api.data_types import RetrievalResult
from config import config
from inference import Retriever

ES_WIKI_INDEX_NAMES = [
    config.baleen.wiki2017.api_index_name,
    config.baleen.wiki2020.api_index_name,
]

app = Flask(__name__)
cors = CORS(app)
# Set logging
logger = logging.getLogger("FlaskServer")
logger.setLevel(logging.INFO)

@app.route("/")
def hello_world():
    return "<p>Hello, World! from Baleen API</p>"


@app.route("/retrieval", methods=["GET", "POST", "OPTIONS"])
def model_inference():
    logger.info(f"Received retrieval request from {request.remote_addr}")
    # Handle GET and POST requests
    if request.method == "POST":
        parameter_dict: Dict = request.json
    else:
        parameter_dict: Dict = request.args.to_dict()

    # Check parameters are valid
    params_to_check = ["modelName", "wikiVersion", "returnNum", "query_or_queries"]
    if len(parameter_dict) == 0:
        logger.warning("/retrieval: No parameter")
        return {}
    for param in params_to_check:
        if param not in parameter_dict:
            logger.warning(f"/retrieval: No {param} parameter")
            return {}

    # Get params
    model_name = parameter_dict["modelName"]
    query_or_queries = parameter_dict["query_or_queries"]
    wiki_version = parameter_dict["wikiVersion"]
    returnNum = int(parameter_dict["returnNum"])

    # Only accept modelName as IRRR
    if model_name.lower() != "baleen":
        logger.warning(f"Bad model name: {model_name}")
        return {}
    # Check wiki version
    if wiki_version == config.baleen.wiki2017.api_index_name:
        retriever_wiki_version = "2017"
    elif wiki_version == config.baleen.wiki2020.api_index_name:
        retriever_wiki_version = "2020"
    else:
        logger.warning(f"bad wiki version: {wiki_version}")
        return {}
    
    retriever = Retriever(wiki_version=retriever_wiki_version)
    result_or_results: Union[RetrievalResult, List[RetrievalResult]] = retriever(
        query_or_queries=query_or_queries, return_num=returnNum
    )
    
    # Result to dic
    if type(result_or_results) == list:
        result_or_results = [result.dic for result in result_or_results]
    else:
        result_or_results = result_or_results.dic
        
    if type(result_or_results) == list:
        logger.info(f"Mutliple questions asked: {len(result_or_results)}")
        result = result_or_results[0]
    logger.info(f"Question: {result['query']}")
    logger.info(f"Retrieved Docs length: {len(result['documents_with_score'])}")
    logger.info(f"Top 1 doc title: {result['documents_with_score'][0]['title']}")
    logger.info(f"Wiki: {retriever_wiki_version}")
    logger.info(f"MaxreturnNum: {returnNum}")

    # Format to response dict
    return result_or_results


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s|%(levelname)s| %(message)s",
        level=logging.INFO,
        datefmt="%m-%d-%H:%M:%S",
    )
    serve(app, host=config.api_server.host, port=config.api_server.port)
