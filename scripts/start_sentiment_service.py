import logging
import argparse
import sys
import json

from deepcubes.models.sentiment import Sentiment
from deepcubes_services.services import SentimentService


parser = argparse.ArgumentParser(description='Sentiment Service starter')
parser.add_argument('-m', '--model', required=True,
                    help="Path sentiment model cube file.")
parser.add_argument('-n', '--neural', required=True,
                    help="Path to sentiment model torch state dict file.")
parser.add_argument('-p', '--port', type=int, default=3333,
                    help="Port at which service will be opened.")
parser.add_argument('-l', '--logs', default="scripts/logs/sentiment_service.log",
                    help="Path to log file.")
args = parser.parse_args()

with open(args.model, "r") as infile:
    model_params = json.load(infile)
model = Sentiment.load(model_params, args.neural)

try:
    with open(args.model, "r") as infile:
        model_params = json.load(infile)
    model = Sentiment.load(model_params, args.neural)
except Exception:
    _, exc_obj, exc_tb = sys.exc_info()
    print("Error while loading the model.")
    print("line {}, {}".format(exc_tb.tb_lineno, exc_obj))
    sys.exit()


logger = logging.getLogger("SentimentService")
logger.setLevel(logging.INFO)

# create the logging file handler
handler = logging.FileHandler(args.logs)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

service = SentimentService(logger, model)
service.run(args.port)
