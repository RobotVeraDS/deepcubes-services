from flask import Flask, request
from flask_json import FlaskJSON, as_json, JsonError

from deepcubes.embedders import LocalEmbedder


class EmbedderService(object):

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def create_flask_app(self):
        self.logger.info("Started Embedder Server...")

        self.logger.info("Load embedders: {} and paths: {} ...".format(
            list(self.config['embedders'].keys()),
            list(self.config['embedders'].values())
        ))

        embedders = {name: LocalEmbedder(path)
                     for name, path in self.config['embedders'].items()}

        self.logger.info("Prepare Flask app...")
        app = Flask(__name__)

        @app.route("/<name>", methods=["POST"])
        @as_json
        def get_vectors(name):
            try:
                self.logger.info("Received {} `{}` request from {}".format(
                    request.method, name, request.remote_addr
                ))

                if name not in embedders:
                    self.logger.error("Attempt to use wrong embedder : {}".format(name))
                    raise ValueError("`{}` embedder doesn't exists".format(name))

                embedder = embedders[name]
                data = request.form if request.form else request.json

                tokens = data["tokens"]
                vectors = embedder(tokens)

                return {'vectors': vectors}

            except Exception as e:
                self.logger.error('error when handling HTTP request', exc_info=True)
                raise JsonError(description=str(e), type=str(type(e).__name__))

        FlaskJSON(app)
        return app

    def run(self, port):
        app = self.create_flask_app()
        app.run(host="0.0.0.0", port=port, debug=False)
