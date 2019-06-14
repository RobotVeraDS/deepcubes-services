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
from classifier.embedder import Embedder as AdaptedEmbedder


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

    def create(self, embedder_mode, tokenizer_mode=Tokenizer.Mode.TOKEN):
        if self.factory_type == FactoryType.NETWORK:
            return NetworkEmbedder(self._get_full_url(embedder_mode))
        else:
            return LocalEmbedder(self._get_full_path(embedder_mode),
                                 Tokenizer(tokenizer_mode))

class AdaptedEmbedderFactory(EmbedderFactoryABC):

    def __init__(self, mode_to_url):
        self.mode_to_url = mode_to_url

    def create(self, embedder_mode, tokenizer_mode=Tokenizer.Mode.TOKEN):
        return AdaptedEmbedder(self.mode_to_url[embedder_mode])

class NetworkEmbedder(Embedder):
    """Network embedder"""

    EMPTY_STRING = ""

    def __init__(self, url, mode=None):
        if mode is None:
            mode = os.path.basename(url)

        self.mode = mode
        self.url = url

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

    def get_tokenizer_mode(self):
        response = requests.post("{}/get_tokenizer_mode".format(self.url))

        if response.status_code != 200:
            raise ValueError("Network embedder error. Status code: {}.".format(
                response.status_code))

        content = json.loads(response.text)
        if 'tokenizer_mode' not in content:
            raise ValueError("Network embedder error. No `tokenizer_mode` in output.")
        else:
            return content['tokenizer_mode']

    def get_embedde_mode(self):
        return self.mode

    def _get_vectors(self, url, data):
        response = requests.post(url, json=data)

        if response.status_code != 200:
            raise ValueError("Network embedder error. Status code: {}.".format(
                response.status_code))

        content = json.loads(response.text)
        if 'vectors' not in content:
            raise ValueError("Network embedder error. No `vectors` in output.")
        else:
            return content['vectors']


def is_url(path):
    # TODO: need more sophisticated url checker
    return path.startswith("http")
