from flask import Flask, request
from flask_json import FlaskJSON, as_json, JsonError


class SentimentService(object):

    def __init__(self, logger, model):
        self.logger = logger
        self.model = model

    def create_flask_app(self):
        self.logger.info("Started Sentiment Server...")

        self.logger.info("Prepare Flask app...")
        app = Flask(__name__)

        @app.route("/sentiment", methods=["GET", "POST"])
        @as_json
        def sentiment():
            try:
                if request.method == "POST":
                    data = request.form if request.form else request.json
                    query = data["query"]
                else:
                    query = request.args.get("query")

                self.logger.info("Received {} `sentiment` request from {}".format(
                    request.method, request.remote_addr
                ))

                positive_proba = float(self.model([query])[0])
                return {'positive_proba': positive_proba}

            except Exception as e:
                self.logger.error('error when handling HTTP request', exc_info=True)
                raise JsonError(description=str(e), type=str(type(e).__name__))

        FlaskJSON(app)
        return app

    def run(self, port):
        app = self.create_flask_app()
        app.run(host="0.0.0.0", port=port, debug=False)
