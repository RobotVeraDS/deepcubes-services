import argparse
import configparser
import os
import json

import pandas as pd

from deepcubes.models import LogisticIntentClassifier
from deepcubes.services.embedders import EmbedderFactory
from deepcubes.services.utils import get_new_model_id


def main(csv_path, lang, config_path):
    config_parser = configparser.ConfigParser()
    config_parser.read(config_path)

    MODEL_STORAGE = config_parser.get('classifier-service', 'MODEL_STORAGE')
    EMBEDDER_PATH = config_parser.get('classifier-service', 'EMBEDDER_PATH')

    embedder_factory = EmbedderFactory(EMBEDDER_PATH)

    LANG_TO_EMB_MODE = dict(config_parser['embedder'])
    LANG_TO_TOK_MODE = dict(config_parser['tokenizer'])

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

    embedder = embedder_factory.create(LANG_TO_EMB_MODE[lang])
    classifier = LogisticIntentClassifier(embedder)
    tokenizer_mode = LANG_TO_TOK_MODE[lang]
    classifier.train(answers, questions, tokenizer_mode)

    new_model_id = get_new_model_id(MODEL_STORAGE)

    clf_params = classifier.save()
    clf_path = os.path.join(MODEL_STORAGE, '{}.cube'.format(new_model_id))

    os.makedirs(MODEL_STORAGE, exist_ok=True)
    with open(clf_path, 'w') as out:
        out.write(json.dumps(clf_params))

    if new_model_id is not None:
        print('Created intent classifier model id: {}'.format(new_model_id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train new model of intent classifier'
    )

    parser.add_argument('-s', '--csv_path', required=True)
    parser.add_argument('-l', '--lang', required=True)
    parser.add_argument('-c', '--config', required=True)

    args = parser.parse_args()
    main(args.csv_path, args.lang, args.config)
