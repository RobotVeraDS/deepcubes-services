import os
import json

from flask import Flask, request
from flask_json import FlaskJSON, as_json, JsonError

from deepcubes.models import LogisticIntentClassifier, MultistageIntentClassifier

from .embedders import EmbedderFactory


class MultistageClassifierService(object):

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.model_storage = config.get('multistage-classifier-service',
                                        'MODEL_STORAGE')
        embedder_path = config.get('multistage-classifier-service',
                                   'EMBEDDER_PATH')
        self.embedder_factory = EmbedderFactory(embedder_path)
        major_model_id = config.get('multistage-classifier-service',
                                    'MAJOR_MODEL_ID')
        minor_model_id = config.get('multistage-classifier-service',
                                    'MINOR_MODEL_ID')
        groups_data_path = config.get('multistage-classifier-service',
                                      'GROUPS_DATA_PATH')
        self.multistage_model = self.load_model(major_model_id, minor_model_id,
                                                groups_data_path)

    def create_flask_app(self):
        self.logger.info("Started Multistage Classifier Server...")

        self.logger.info("Prepare Flask app...")
        app = Flask(__name__)

        @app.route("/predict", methods=["POST"])
        @as_json
        def predict():
            data = request.form if request.form else request.json

            try:
                self.logger.info("Received {} request from {}".format(
                    request.method, request.remote_addr)
                )
                query = data["query"]

                model_answer = self.multistage_model([query])

                output = [{
                    "answer": label,
                    "probability": None,
                    "threshold": 0.3,
                    "accuracy_score": None
                } for label in model_answer]

                self.logger.info('Received query: {}'.format(query))
                self.logger.info("Top predicted label: {}".format(output[0]['answer']))
                self.logger.info("Max probability: {}".format(output[0]['probability']))

                return output

            except Exception as e:
                self.logger.error('error when handling HTTP request', exc_info=True)
                raise JsonError(description=str(e), type=str(type(e).__name__))

        FlaskJSON(app)
        return app

    def load_model(self, major_model_id, minor_model_id, groups_data_path):
        self.logger.info("Loading major model {} ...".format(major_model_id))

        major_model_path = os.path.join(
            self.model_storage, "{}.cube".format(major_model_id)
        )

        if not os.path.isfile(major_model_path):
            raise ValueError("Model {} not found".format(major_model_id))

        with open(major_model_path, 'r') as data:
            major_model_params = json.loads(data.read())

        major_model = LogisticIntentClassifier.load(major_model_params,
                                                    self.embedder_factory)
        self.logger.info("Loading minor model {} ...".format(minor_model_id))

        minor_model_path = os.path.join(
            self.model_storage, "{}.cube".format(minor_model_id)
        )

        if not os.path.isfile(minor_model_path):
            raise ValueError("Model {} not found".format(minor_model_id))

        with open(minor_model_path, 'r') as data:
            minor_model_params = json.loads(data.read())

        minor_model = LogisticIntentClassifier.load(minor_model_params,
                                                    self.embedder_factory)

        multistage_model = MultistageIntentClassifier(major_model, minor_model)
        multistage_model.train(groups_data_path)

        return multistage_model

    def run(self, port):
        app = self.create_flask_app()
        app.run(host="0.0.0.0", port=port, debug=False)
