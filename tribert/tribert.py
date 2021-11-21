import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

import ipdb

import numpy as np


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(
        self,
        vocab_size_or_config_json_file,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        v_feature_size=1024,
        v_target_size=21,
        v_hidden_size=768,
        v_num_hidden_layers=3,
        v_num_attention_heads=12,
        v_intermediate_size=3072,
        p_feature_size = 1024,
        p_target_size=21,
        p_hidden_size=768,
        p_num_hidden_layers=3,
        p_num_attention_heads=12,
        p_intermediate_size=3072,
        bi_hidden_size=1024,
        bi_num_attention_heads=16,
        v_attention_probs_dropout_prob=0.1,
        v_hidden_act="gelu",
        v_hidden_dropout_prob=0.1,
        v_initializer_range=0.2,
        v_biattention_id=[0, 1],
        p_biattention_id=[10, 11],
        a_biattention_id=[16, 17],
        predict_feature=False,
        fast_mode=False,
        fixed_v_layer=0,
        fixed_p_layer=0,
        fixed_a_layer = 0,
        in_batch_pairs=False,
        fusion_method="mul",
        intra_gate=False,
        with_coattention=True
    ):

        assert len(v_biattention_id) == len(p_biattention_id)
        assert max(v_biattention_id) < v_num_hidden_layers
        assert max(p_biattention_id) < num_hidden_layers

        if isinstance(vocab_size_or_config_json_file, str) or (
            sys.version_info[0] == 2
            and isinstance(vocab_size_or_config_json_file, unicode)
        ):
            with open(vocab_size_or_config_json_file, "r", encoding="utf-8") as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.v_feature_size = v_feature_size
            self.v_hidden_size = v_hidden_size
            self.v_num_hidden_layers = v_num_hidden_layers
            self.v_num_attention_heads = v_num_attention_heads
            self.v_intermediate_size = v_intermediate_size
            self.v_attention_probs_dropout_prob = v_attention_probs_dropout_prob
            self.v_hidden_act = v_hidden_act
            self.v_hidden_dropout_prob = v_hidden_dropout_prob
            self.v_initializer_range = v_initializer_range
            self.v_biattention_id = v_biattention_id
            self.p_biattention_id = p_biattention_id
            self.p_hidden_act = v_hidden_act
            self.p_hidden_dropout_prob = v_hidden_dropout_prob
            self.a_hidden_act = v_hidden_act
            self.a_hidden_dropout_prob = v_hidden_dropout_prob
            self.v_target_size = v_target_size
            self.bi_hidden_size = bi_hidden_size
            self.bi_num_attention_heads = bi_num_attention_heads
            self.predict_feature = predict_feature
            self.fast_mode = fast_mode
            self.fixed_v_layer = fixed_v_layer
            self.fixed_p_layer = fixed_p_layer
            self.fixed_a_layer = fixed_a_layer

            self.in_batch_pairs = in_batch_pairs
            self.fusion_method = fusion_method
            self.intra_gate = intra_gate
            self.with_coattention=with_coattention
        else:
            raise ValueError(
                "First argument must be either a vocabulary size (int)"
                "or the path to a pretrained model config file (str)"
            )



    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding="utf-8") as reader:
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


