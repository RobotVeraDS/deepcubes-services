from flask import Flask, request
from flask_json import FlaskJSON, as_json, JsonError

from deepcubes.embedders import LocalEmbedder
from deepcubes.cubes import Tokenizer


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

        tokenizer = Tokenizer(Tokenizer.Mode.TOKEN)
        embedders = {name: LocalEmbedder(path, tokenizer)
                     for name, path in self.config['embedders'].items()}

        self.logger.info("Prepare Flask app...")
        app = Flask(__name__)

        @app.route("/<name>/get_tokenizer_mode", methods=["POST"])
        @as_json
        def get_tokenizer_mode(name):
            try:
                self.logger.info("Received {} `{}` get_tokenizer_mode request from {}".format(
                    request.method, name, request.remote_addr
                ))

                return {"tokenizer_mode": embedders[name].get_tokenizer_mode()}

            except Exception as e:
                self.logger.error('error when handling HTTP request', exc_info=True)
                raise JsonError(description=str(e), type=str(type(e).__name__))

        @app.route("/<name>/get_embedder_mode", methods=["POST"])
        @as_json
        def get_embedder_mode(name):
            try:
                self.logger.info("Received {} `{}` get_embedder_mode request from {}".format(
                    request.method, name, request.remote_addr
                ))

                return {"embedder_mode": embedders[name].get_embedder_mode()}

            except Exception as e:
                self.logger.error('error when handling HTTP request', exc_info=True)
                raise JsonError(description=str(e), type=str(type(e).__name__))

        @app.route("/<name>/encode_tokens", methods=["POST"])
        @as_json
        def encode_tokens(name):
            try:
                self.logger.info("Received {} `{}` encode_tokens request from {}".format(
                    request.method, name, request.remote_addr
                ))

                if name not in embedders:
                    self.logger.error("Attempt to use wrong embedder : {}".format(name))
                    raise ValueError("`{}` embedder doesn't exists".format(name))

                embedder = embedders[name]
                data = request.form if request.form else request.json

                tokens = data["tokens"]
                vectors = embedder.encode_tokens(tokens)

                return {'vectors': vectors}

            except Exception as e:
                self.logger.error('error when handling HTTP request', exc_info=True)
                raise JsonError(description=str(e), type=str(type(e).__name__))

        @app.route("/<name>/encode_queries", methods=["POST"])
        @as_json
        def encode_queries(name):
            try:
                self.logger.info("Received {} `{}` encode_queries request from {}".format(
                    request.method, name, request.remote_addr
                ))

                if name not in embedders:
                    self.logger.error("Attempt to use wrong embedder : {}".format(name))
                    raise ValueError("`{}` embedder doesn't exists".format(name))

                embedder = embedders[name]
                data = request.form if request.form else request.json

                queries = data["queries"]
                vectors = embedder.encode_queries(queries)

                return {'vectors': vectors}

            except Exception as e:
                self.logger.error('error when handling HTTP request', exc_info=True)
                raise JsonError(description=str(e), type=str(type(e).__name__))

        FlaskJSON(app)
        return app

    def run(self, port):
        app = self.create_flask_app()
        app.run(host="0.0.0.0", port=port, debug=False)
