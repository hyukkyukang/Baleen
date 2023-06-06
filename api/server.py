import logging
from flask import Flask, request
from flask_cors import CORS
from waitress import serve


import hkkang_utils.file as file_utils


app = Flask(__name__)
cors = CORS(app)
# Set logging
logger = logging.getLogger("FlaskServer")
logger.setLevel(logging.INFO)

# Load model to memory
logger.info("Loading model...")
inference_model = Inferencer()
logger.info("Loading model done!")


@app.route("/")
def hello_world():
    return "<p>Hello, World! from AISO API</p>"


@app.route("/inference", methods=["GET", "POST", "OPTIONS"])
def model_inference():
    # GET variables
    parameter_dict = request.args.to_dict()

    # Check parameters are valid
    params_to_check = ["modelName", "question", "wikiVersion", "stepSize"]
    if len(parameter_dict) == 0:
        logger.warning("/inference: No parameter")
        return {}
    for param in params_to_check:
        if param not in parameter_dict:
            logger.warning(f"/inference: No {param} parameter")
            return {}

    # Get params
    question = parameter_dict["question"]
    model_name = parameter_dict["modelName"]
    wiki_version = parameter_dict["wikiVersion"]
    step_size = parameter_dict["stepSize"]

    # Only accept modelName as IRRR
    if model_name.lower() != "irrr":
        logger.warning(f"Bad model name: {model_name}")
        return {}
    # Check wiki version
    if wiki_version in ES_WIKI_INDEX_NAMES:
        es_index = wiki_version
    else:
        logger.warning(f"bad wiki version: {wiki_version}")
        return {}
    # Parse max step to integer
    try:
        step_size = int(step_size)
    except:
        logger.warning(f"Max step is not an integer: {step_size}")
        return {}

    result = inference_model(question, index_name=es_index, max_step=step_size)
    logger.info(f"Question: {result.question}")
    logger.info(f"Wiki: {wiki_version}")
    logger.info(f"Step: {step_size}")
    logger.info(f"Answer: {result.answer}\n")
    return result.to_response_dict()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s|%(levelname)s| %(message)s",
        level=logging.INFO,
        datefmt="%m-%d-%H:%M:%S",
    )
    serve(app, host=config.api_server.host, port=config.api_server.port)
