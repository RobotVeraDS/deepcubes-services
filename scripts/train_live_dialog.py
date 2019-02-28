import argparse
import configparser
import os
from pprint import pprint
import json

import pandas as pd

from deepcubes.models import VeraLiveDialog
from deepcubes_services.services.embedders import NetworkEmbedder


def main(csv_path, lang, config_path, model_id):

    config_parser = configparser.ConfigParser()
    config_parser.read(config_path)

    MODEL_STORAGE = config_parser.get('live-dialog-service', 'MODEL_STORAGE')
    EMBEDDER_URL = config_parser.get('live-dialog-service', 'EMBEDDER_PATH')
    LANG_TO_EMB_MODE = dict(config_parser['embedder'])

    embedder = NetworkEmbedder('{}/{}'.format(EMBEDDER_URL, LANG_TO_EMB_MODE[lang]))

    LANG_TO_EMB_MODE = dict(config_parser['embedder'])

    GENERIC_DATA_PATH = config_parser.get('live-dialog-service',
                                          'GENERIC_DATA_PATH')
    labels_settings = list()

    if csv_path:
        # parse from pandas data frame
        data = pd.read_csv(csv_path)
        for column in data.columns:
            values = data.loc[
                ~pd.isnull(data[column])
            ][column].values

            answer = values[0].strip()
            intent_phrases = list()
            for question in values[1:]:
                intent_phrases.append(question)
            labels_settings.append({
                'label': answer,
                'intent_phrases': intent_phrases,
            })

    for generic in ['yes', 'no', 'repeat']:
        labels_settings.append({
            'label': generic,
            'generics': [generic],
        })

    config = {
        'labels_settings': labels_settings,
        'not_understand_label': 'not_understand_label',
    }
    config['embedder_mode'] = LANG_TO_EMB_MODE[lang]
    pprint(config)

    live_dialog_model = VeraLiveDialog(embedder, GENERIC_DATA_PATH)
    live_dialog_model.train(config)

    clf_params = live_dialog_model.save()
    clf_path = os.path.join(MODEL_STORAGE, '{}.cube'.format(model_id))

    os.makedirs(MODEL_STORAGE, exist_ok=True)
    with open(clf_path, 'w') as out:
        out.write(json.dumps(clf_params))

    if model_id is not None:
        print('Created live dialog model with id {}'.format(model_id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train new model of live dialog'
    )

    parser.add_argument('-s', '--csv_path', required=True)
    parser.add_argument('-l', '--lang', required=True)
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-m', '--model_id', required=True)

    args = parser.parse_args()
    main(args.csv_path, args.lang, args.config, args.model_id)
