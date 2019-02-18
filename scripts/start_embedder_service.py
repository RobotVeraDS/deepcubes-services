import configparser
import logging
import argparse

from deepcubes.services import EmbedderService


parser = argparse.ArgumentParser(description='Embedder service starter')
parser.add_argument('-c', '--config', required=True,
                    help="Path to config file.")
parser.add_argument('-p', '--port', type=int, default=3333,
                    help="Port at which service will be opened.")
parser.add_argument('-l', '--logs', default="scripts/logs/embedder_service.log",
                    help="Path to log file.")
args = parser.parse_args()

logger = logging.getLogger("EmbedderService")
logger.setLevel(logging.INFO)

handler = logging.FileHandler(args.logs)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

config_file_path = args.config
config_parser = configparser.ConfigParser()
config_parser.read(config_file_path)

service = EmbedderService(config_parser, logger)
service.run(args.port)
