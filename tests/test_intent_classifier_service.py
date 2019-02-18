import json
import os
import unittest
import logging
import configparser

from deepcubes_services.services import IntentClassifierService
from deepcubes.embedders import LocalEmbedder
from deepcubes.models import LogisticIntentClassifier
from deepcubes.services.utils import get_new_model_id


class IntentClassifierServiceTest(unittest.TestCase):

    def setUp(self):
        logger = logging.getLogger("IntentClassifierTestService")
        logger.setLevel(logging.INFO)

        config_parser = configparser.ConfigParser()
        config_parser.read(
            "tests/data/intent_classifier_service/intent_classifier_service.conf"
        )

        app = IntentClassifierService(config_parser, logger).create_flask_app()
        app.tesing = True

        self.service = app.test_client()

        self.model_storage = 'tests/models/intents'
        os.makedirs(self.model_storage, exist_ok=True)

        self.embedder = LocalEmbedder('tests/data/test_embeds.kv')
        self.classifier = LogisticIntentClassifier(self.embedder)

        self.questions, self.answers = [], []

        with open("tests/data/test_dialog.json", "r") as handle:
            data = json.load(handle)

        for label, category in enumerate(data):
            answer = category["answers"][0]

            for question in category["questions"]:
                self.questions.append(question)
                self.answers.append(answer)

        self.classifier.train(self.answers, self.questions, 'lem')
        self.model_id = get_new_model_id(self.model_storage)
        self.clf_params = self.classifier.save()

        self.clf_path = os.path.join(
            self.model_storage, '{}.cube'.format(self.model_id)
        )

        with open(self.clf_path, 'w') as out:
            out.write(json.dumps(self.clf_params))

        self.output_keys = ["answer", "probability",
                            "threshold", "accuracy_score"]

    def test_get_requests(self):
        predict_resp_data = self._get_predict_response(
            query='название',
            model_id=self.model_id
        )

        self.assertEqual(2, len(predict_resp_data))
        for output in predict_resp_data:
            for key in self.output_keys:
                self.assertIn(key, output)

    def test_post_train_request(self):

        predict_resp_data = self._get_predict_response(
            query='чем занимается ваша фирма',
            model_id=self.model_id
        )

        self.assertEqual(2, len(predict_resp_data))
        for output in predict_resp_data:
            for key in self.output_keys:
                self.assertIn(key, output)

    def _get_predict_response(self, query, model_id):
        predict_resp = self.service.post(
            '/predict', json={
                'query': query,
                'model_id': model_id,
            }
        )

        predict_resp_data = json.loads(predict_resp.data.decode("utf-8"))
        return predict_resp_data

    def tearDown(self):
        os.remove(self.clf_path)
