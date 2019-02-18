import unittest
import json
import os
import logging
import configparser

from deepcubes_services.services import VeraLiveDialogService


class VeraLiveDialogServiceTest(unittest.TestCase):

    def setUp(self):
        logger = logging.getLogger("VeraLiveDialogTestService")
        logger.setLevel(logging.INFO)

        config_parser = configparser.ConfigParser()
        config_parser.read(
            "tests/data/vera_live_dialog/vera_live_dialog.conf"
        )

        app = VeraLiveDialogService(config_parser, logger).create_flask_app()
        app.tesing = True

        self.service = app.test_client()

        self.models_storage = 'tests/models/live_dialog'
        os.makedirs(self.models_storage, exist_ok=True)

        self.test_config_path = 'tests/data/vera_live_dialog/test.config'

        with open(self.test_config_path, 'r') as conf_file:
            self.test_config = conf_file.read()

        self.request_data = {'config':  self.test_config}
        self.test_models_list = list()

    def test_get_requests(self):
        train_resp = self.service.post('/train', json=self.request_data)
        train_resp_data = json.loads(train_resp.data.decode("utf-8"))

        self.assertIn('model_id', train_resp_data)
        model_id = train_resp_data['model_id']
        self.test_models_list.append(model_id)

        predict_resp_data = self._get_predict_response(
            query='привет',
            model_id=model_id
        )
        labels, probs = self._dicts_to_values_list(predict_resp_data)

        self.assertEqual(
            'hello',
            predict_resp_data[0]['label']
        )

        self.assertEqual(11, len(labels))

        predict_resp_data = self._get_predict_response(
            query='график',
            model_id=model_id,
            labels=['1', '2', '3']
        )
        labels, probs = self._dicts_to_values_list(predict_resp_data)

        self.assertEqual(
            'not_understand',
            predict_resp_data[0]['label']
        )

        self.assertEqual(1, len(labels))

    def test_post_train_request(self):
        train_resp = self.service.post('/train', data=self.request_data)
        train_resp_data = json.loads(train_resp.data.decode("utf-8"))

        self.assertIn('model_id', train_resp_data)
        model_id = train_resp_data['model_id']
        self.test_models_list.append(model_id)

        predict_resp_data = self._get_predict_response(
            query='неет',
            model_id=model_id
        )
        labels, probs = self._dicts_to_values_list(predict_resp_data)

        self.assertEqual(
            'no',
            predict_resp_data[0]['label']
        )

        self.assertIn('not_understand', labels)

        predict_resp_data = self._get_predict_response(
            query='',
            model_id=model_id,
            labels=['yes', 'no']
        )
        labels, probs = self._dicts_to_values_list(predict_resp_data)

        self.assertEqual(
            'not_understand',
            predict_resp_data[0]['label']
        )

        self.assertEqual(3, len(labels))

        predict_resp_data = self._get_predict_response(
            query='НЕЕТ!',
            model_id=model_id,
            labels=['no', 'yes']
        )
        labels, probs = self._dicts_to_values_list(predict_resp_data)

        self.assertEqual(
            'no',
            predict_resp_data[0]['label']
        )

        self.assertEqual(3, len(labels))

    def _get_predict_response(self, query, model_id, labels=None):
        predict_resp = self.service.post(
            '/predict', json={
                'query': query,
                'model_id': model_id,
                'labels': labels
            }
        )

        predict_resp_data = json.loads(predict_resp.data.decode("utf-8"))
        return predict_resp_data

    def _dicts_to_values_list(self, data):
        labels = [_dict['label'] for _dict in data]
        probs = [_dict['proba'] for _dict in data]
        return labels, probs

    def tearDown(self):
        for model_id in self.test_models_list:
            os.remove(
                os.path.join(self.models_storage, '{}.cube'.format(model_id))
            )
