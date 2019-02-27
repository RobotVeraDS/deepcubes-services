import argparse
import logging
import configparser

from deepcubes_services.services import MultistageClassifierService


parser = argparse.ArgumentParser(description='Multistage Classifier service')
parser.add_argument('-c', '--config', type=str, required=True,
                    help="Path to config file")
parser.add_argument('-p', '--port', type=int, default=3333,
                    help="Port at which service will be opened.")
parser.add_argument('-l', '--logs', default="scripts/logs/multistage_classifier_service.log",
                    help="Path to log file.")
args = parser.parse_args()


logger = logging.getLogger("MultistageClassifierService")
logger.setLevel(logging.INFO)

handler = logging.FileHandler(args.logs)

formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

config_file_path = args.config
config_parser = configparser.RawConfigParser()
config_parser.read(config_file_path)

service = MultistageClassifierService(config_parser, logger)
service.run(args.port)
