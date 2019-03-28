from flask import Flask, request
from flask_json import FlaskJSON, as_json, JsonError
import os
import json

from deepcubes.models import IntentClassifier
from deepcubes.cubes import Tokenizer
from deepcubes.utils.functions import sorted_labels
from deepcubes.embedders import (LocalEmbedder,
                                 EmbedderFactory as EmbedderFactoryABC)

from .embedders import FactoryType, NetworkEmbedder, is_url

class EmbedderFactory(EmbedderFactoryABC):
    # EmbedderFactory for creation of Network embedder
    # Use third-party embedder service
    # https://git.stafory.tech/services/embedder
    def __init__(self, path):
        if is_url(path):
            self.factory_type = FactoryType.NETWORK

        else:
            self.factory_type = FactoryType.LOCAL

        self.path = path

    def _get_full_url(self):
        return "{}".format(self.path)

    def _get_full_path(self, mode):
        return os.path.join(self.path, "{}.kv".format(mode))

    def create(self, embedder_mode, tokenizer_mode=Tokenizer.Mode.TOKEN):
        if self.factory_type == FactoryType.NETWORK:
            return NetworkEmbedder(self._get_full_url())
        else:
            return LocalEmbedder(self._get_full_path(embedder_mode),
                                 Tokenizer(tokenizer_mode))



class IntentClassifierService(object):

    def __init__(self, config, logger, models_ids=[]):
        self.config = config
        self.logger = logger
        self.models_ids = models_ids

        self.model_storage = config.get('classifier-service', 'MODEL_STORAGE')
        embedder_path = config.get('classifier-service', 'EMBEDDER_PATH')
        self.embedder_factory = EmbedderFactory(embedder_path)

        self.models = dict()
        for model_id in models_ids:
            self.models[model_id] = self.load_model(model_id)

    def create_flask_app(self):
        self.logger.info("Started Intent Classifier Server...")

        self.logger.info("Prepare Flask app...")
        app = Flask(__name__)

        @app.route("/predict", methods=["POST"])
        @as_json
        def predict():
            data = request.form if request.form else request.json

            try:
                model_id = int(data["model_id"])
                query = data["query"]

                self.logger.info("Received {} request from {}".format(
                    request.method, request.remote_addr))
                self.logger.info("Received model id: {}".format(model_id))

                if model_id in self.models:
                    model = self.models[model_id]
                else:
                    model = self.load_model(model_id)
                    self.models[model_id] = model

                model_answer = sorted_labels(model([query]))[0]
                output = [{
                    "answer": label,
                    "probability": probability,
                    "threshold": 0.3,
                    "accuracy_score": None
                } for label, probability in model_answer]

                self.logger.info('Received query: {}'.format(query))
                self.logger.info("Top predicted label: {}".format(output[0]['answer']))
                self.logger.info("Max probability: {}".format(output[0]['probability']))

                return output

            except Exception as e:
                self.logger.error('error when handling HTTP request', exc_info=True)
                raise JsonError(description=str(e), type=str(type(e).__name__))

        FlaskJSON(app)
        return app

    def load_model(self, model_id):
        self.logger.info("Loading intent model {} ...".format(model_id))
        model_path = os.path.join(self.model_storage, "{}.cube".format(model_id))

        if not os.path.isfile(model_path):
            raise ValueError("Model {} not found".format(model_id))

        with open(model_path, 'r') as data:
            model_params = json.loads(data.read())

        model = IntentClassifier.load(model_params, self.embedder_factory)
        return model

    def run(self, port):
        app = self.create_flask_app()
        app.run(host="0.0.0.0", port=port, debug=False)
