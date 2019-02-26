import json
import os
from enum import Enum

import requests

from deepcubes.embedders import (
    Embedder,
    EmbedderFactory as EmbedderFactoryABC,
    LocalEmbedder,
)

from deepcubes.cubes import Tokenizer


class FactoryType(Enum):
    LOCAL = 0
    NETWORK = 1


class EmbedderFactory(EmbedderFactoryABC):

    def __init__(self, path):
        if is_url(path):
            self.factory_type = FactoryType.NETWORK
        else:
            self.factory_type = FactoryType.LOCAL

        self.path = path

    def _get_full_url(self, mode):
        return "{}/{}".format(self.path, mode)

    def _get_full_path(self, mode):
        return os.path.join(self.path, "{}.kv".format(mode))

    def create(self, embedder_serialized):
        if self.factory_type == FactoryType.NETWORK:
            return NetworkEmbedder(self._get_full_url(embedder_serialized["mode"]))
        else:
            if "tokenizer" in embedder_serialized:
                tokenizer = Tokenizer.load(embedder_serialized["tokenizer"])
            else:
                tokenizer = Tokenizer(Tokenizer.Mode.TOKEN)

            return LocalEmbedder(self._get_full_path(embedder_serialized["mode"]), tokenizer)


class NetworkEmbedder(Embedder):
    """Network embedder"""

    EMPTY_STRING = ""

    def __init__(self, url, mode=None):
        if mode is None:
            mode = os.path.basename(url)

        super().__init__(mode)
        self.url = url

    def _get_vectors(self, url, data):
        response = requests.post(url, json=data)

        if response.status_code != 200:
            raise ValueError("Network embedder error. Status code: {}.".format(
                response.status_code))

        content = json.loads(response.text)
        if 'vectors' not in content:
            # TODO: think about
            raise ValueError("Network embedder error. No `vectors` in output.")
        else:
            return content['vectors']

    def encode_queries(self, queries):
        data = {"queries": queries}
        url = "{}/encode_queries".format(self.url)
        return self._get_vectors(url, data)

    def encode_tokens(self, tokens_batch):
        # TODO: fix this not idiomatic way to process empty tokens
        for tokens in tokens_batch:
            if not len(tokens):
                tokens = [self.EMPTY_STRING]

        data = {'tokens': tokens_batch}
        url = "{}/encode_tokens".format(self.url)
        return self._get_vectors(url, data)

    def save(self):
        cube_params = {
            "class": self.__class__.__name__,
            "mode": self.mode
        }

        return cube_params

    @classmethod
    def load(cls, cube_params, url):
        network_embedder = cls(url, cube_params["mode"])
        return network_embedder


def is_url(path):
    # TODO: need more sophisticated url checker
    return path.startswith("http")
