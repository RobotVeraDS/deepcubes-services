import argparse
import configparser
import os
import json

import pandas as pd

from deepcubes.models import IntentClassifier
from deepcubes.cubes import Tokenizer, Classifier
from deepcubes_services.services.embedders import NetworkEmbedder


def main(csv_path, lang, config_path, model_id):
    config_parser = configparser.ConfigParser()
    config_parser.read(config_path)

    MODEL_STORAGE = config_parser.get('classifier-service', 'MODEL_STORAGE')
    LANG_TO_EMB_MODE = dict(config_parser['embedder'])
    EMB_MODE_TO_EMB_PATH = dict(config_parser['embedders'])

    embedder_mode = LANG_TO_EMB_MODE[lang]
    EMBEDDER_URL = EMB_MODE_TO_EMB_PATH[embedder_mode]

    embedder = NetworkEmbedder(EMBEDDER_URL)

    questions, answers = [], []

    if csv_path:
        # parse from pandas data frame
        data = pd.read_csv(csv_path)
        for column in data.columns:
            values = data.loc[
                ~pd.isnull(data[column])
            ][column].values

            answer = values[0].strip()
            for question in values[1:]:
                questions.append(question)
                answers.append(answer)

    classifier = IntentClassifier(embedder, Classifier.Mode.LINEARSVC)
    classifier.train(questions, answers)

    clf_params = classifier.save()
    clf_path = os.path.join(MODEL_STORAGE, '{}.cube'.format(model_id))

    os.makedirs(MODEL_STORAGE, exist_ok=True)
    with open(clf_path, 'w') as out:
        out.write(json.dumps(clf_params))

    if model_id is not None:
        print('Created intent classifier model id: {}'.format(model_id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train new model of intent classifier'
    )

    parser.add_argument('-s', '--csv_path', required=True)
    parser.add_argument('-l', '--lang', required=True)
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-m', '--model_id', required=True)

    args = parser.parse_args()
    main(args.csv_path, args.lang, args.config, args.model_id)
