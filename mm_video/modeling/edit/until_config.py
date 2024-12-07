"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import logging
import tarfile
import tempfile
import shutil
import torch
from .file_utils import cached_path

from torch import distributed as dist

logger = logging.getLogger(__name__)


class PretrainedConfig(object):
    pretrained_model_archive_map = {}
    config_name = ""
    weights_name = ""

    @classmethod
    def get_config(cls, pretrained_model_name, cache_dir, type_vocab_size, state_dict):
        archive_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), pretrained_model_name)
        if os.path.exists(archive_file) is False:
            if pretrained_model_name in cls.pretrained_model_archive_map:
                archive_file = cls.pretrained_model_archive_map[pretrained_model_name]
            else:
                archive_file = pretrained_model_name

        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except FileNotFoundError:
            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.error(
                    "Model name '{}' was not found in model name list. "
                    "We assumed '{}' was a path or url but couldn't find any file "
                    "associated to this path or url.".format(
                        pretrained_model_name,
                        archive_file))
            return None
        if resolved_archive_file == archive_file:
            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.info("loading archive file {}".format(archive_file))
        else:
            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.info("loading archive file {} from cache at {}".format(
                    archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file):
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.info("extracting archive file {} to temp dir {}".format(
                    resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, cls.config_name)
        config = cls.from_json_file(config_file)
        config.type_vocab_size = type_vocab_size
        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info("Model config {}".format(config))

        if state_dict is None:
            weights_path = os.path.join(serialization_dir, cls.weights_name)
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location='cpu')
            else:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    logger.info("Weight doesn't exsits. {}".format(weights_path))

        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)

        return config, state_dict

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = cls(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"