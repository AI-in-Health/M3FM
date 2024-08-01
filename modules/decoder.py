import numpy as np
import torch
import torch.nn as nn
import pickle
from typing import Tuple
from transformers import GPT2LMHeadModel
from .att_models import AttModel
import pdb

class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

class DeCap(AttModel):

    def __init__(self, args, tokenizer):
        super(DeCap, self).__init__(args, tokenizer)

        # decoder: 4 layers transformer with 4 attention heads
        # the decoder is not pretrained
        with open('./decoder_config/decoder_config.pkl', 'rb') as f:
            config = pickle.load(f)
        # Change the parameters you need
        config.vocab_size = tokenizer.get_vocab_size()
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        self.decoder = GPT2LMHeadModel(config)
        self.embedding_size = self.decoder.transformer.wte.weight.shape[1]
        self.prefix_size = 512
        self.clip_project = MLP((self.prefix_size, self.embedding_size))

    def _forward(self, clip_features, gpt_tokens):

        embedding_text = self.decoder.transformer.wte(gpt_tokens)
        embedding_clip = self.clip_project(clip_features)
        embedding_clip = embedding_clip.reshape(-1, 1, self.embedding_size)
        embedding_cat = torch.cat([embedding_clip, embedding_text], dim=1)
        out = self.decoder(inputs_embeds=embedding_cat)
        return out

