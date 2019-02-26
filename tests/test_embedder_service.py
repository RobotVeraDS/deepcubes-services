import unittest
import logging
import configparser

from deepcubes_services.services import EmbedderService
from deepcubes.cubes import Tokenizer


class EmbedderServiceTest(unittest.TestCase):

    def setUp(self):
        logger = logging.getLogger("EmbedderTestService")
        logger.setLevel(logging.INFO)

        config_parser = configparser.ConfigParser()
        config_parser.read("tests/data/embedder_service/embedder_service.conf")

        app = EmbedderService(config_parser, logger).create_flask_app()
        app.tesing = True

        self.service = app.test_client()

    def test_get_requests(self):
        tokenizer = Tokenizer(Tokenizer.Mode.TOKEN)

        queries = ["Робот Вера"]
        tokens = tokenizer(queries)

        response = self.service.post('/test/encode_tokens', json={"tokens": tokens})
        generated_vectors = response.get_json()["vectors"]
        self.assertAlmostEqual(
            sum(generated_vectors[0]),
            -1.5613,
            3
        )

        response = self.service.post('/test/encode_queries', json={"queries": queries})
        generated_vectors = response.get_json()["vectors"]
        self.assertAlmostEqual(
            sum(generated_vectors[0]),
            -1.5613,
            3
        )
