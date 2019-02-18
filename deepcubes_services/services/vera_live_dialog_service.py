import json
import os
from flask import Flask, request
from flask_json import FlaskJSON, as_json, JsonError

from deepcubes.models import VeraLiveDialog

from .embedders import EmbedderFactory
from .utils import get_new_model_id


class VeraLiveDialogService(object):

    def __init__(self, config, logger, models_ids=[]):
        self.config = config
        self.logger = logger

        self.model_storage = config.get('live-dialog-service', 'MODEL_STORAGE')
        self.generic_data_path = config.get('live-dialog-service', 'GENERIC_DATA_PATH')
        embedder_path = config.get('live-dialog-service', 'EMBEDDER_PATH')
        self.embedder_factory = EmbedderFactory(embedder_path)

        self.lang_to_emb_mode = dict(config['embedder'])
        self.lang_to_tok_mode = dict(config['tokenizer'])

        self.models = dict()
        for model_id in models_ids:
            self.models[model_id] = self.load_model(model_id)

    def create_flask_app(self):
        self.logger.info("Started Live Dialog Server...")

        self.logger.info("Prepare Flask app...")
        app = Flask(__name__)

        @app.route("/predict", methods=["POST"])
        @as_json
        def predict():
            data = request.form if request.form else request.json
            try:
                self.logger.info("Received {} `predict` request from {}".format(
                    request.method, request.remote_addr
                ))

                model_id = int(data['model_id'])
                query = data["query"]
                labels = data.get("labels", None)

                self.logger.info("Received model id: {}".format(model_id))

                if model_id in self.models:
                    model = self.models[model_id]
                else:
                    model = self.load_model(model_id)
                    self.models[model_id] = model

                model_answer = model([query], labels)[0]
                output = [{
                    "label": label,
                    "proba": probability
                } for label, probability in model_answer]

                self.logger.info('Received query: {}'.format(query))
                self.logger.info('Received labels: {}'.format(labels))
                self.logger.info("Top predicted label: {}".format(output[0]['label']))
                self.logger.info("Max probability: {}".format(output[0]['proba']))

                return output

            except Exception as e:
                self.logger.error('error when handling HTTP request', exc_info=True)
                raise JsonError(description=str(e), type=str(type(e).__name__))

        @app.route("/train", methods=["POST"])
        @as_json
        def train():
            data = request.form if request.form else request.json
            try:
                self.logger.info("Received {} `train` request from {}".format(
                    request.method, request.remote_addr
                ))

                config = json.loads(data["config"])
                self.logger.info("Received `lang` key: {}".format(config['lang']))

                embedder_mode = self.lang_to_emb_mode[config['lang']]
                embedder = self.embedder_factory.create(embedder_mode)
                self.logger.info("Set embedder mode: {}".format(embedder_mode))

                tokenizer_mode = self.lang_to_tok_mode[config['lang']]
                config['tokenizer_mode'] = tokenizer_mode
                self.logger.info("`{}` tokenizer mode set".format(tokenizer_mode))

                live_dialog_model = VeraLiveDialog(embedder, self.generic_data_path)
                live_dialog_model.train(config)

                new_model_id = get_new_model_id(self.model_storage)

                clf_params = live_dialog_model.save()
                clf_path = os.path.join(self.model_storage, '{}.cube'.format(new_model_id))

                os.makedirs(self.model_storage, exist_ok=True)
                with open(clf_path, 'w') as out:
                    out.write(json.dumps(clf_params))

                self.models[new_model_id] = live_dialog_model

                self.logger.info('Saved model with model_id {}'.format(new_model_id))

                return {
                    "message": 'Created model with model_id {}'.format(new_model_id),
                    "model_id": new_model_id,
                }

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

        model = VeraLiveDialog.load(model_params, self.embedder_factory)
        return model

    def run(self, port):
        app = self.create_flask_app()
        app.run(host="0.0.0.0", port=port, debug=False)
