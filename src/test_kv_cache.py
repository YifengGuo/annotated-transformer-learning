#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : guoyifeng
# @Time     : 2025-02-20
# @Version  : 0.0.1

import torch
import numpy as np
from numpy.ma.core import shape

from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

# text: "The quick brown fox jumps over the lazy"
tokens = [[464, 2068, 7586, 21831, 18045, 625, 262, 16931]]
input_n = torch.tensor(tokens)
# print(input_n)
output_n = model(input_ids=input_n, output_hidden_states=True)

# text: " dog"
tokens[0].append(3290)
input_n_plus_1 = torch.tensor(tokens)
output_n_plus_1 = model(input_ids=input_n_plus_1, output_hidden_states=True)

for i, (hidden_n, hidden_n_plus_1) in enumerate(zip(output_n.hidden_states, output_n_plus_1.hidden_states)):
    print(f"layer {i}, max difference {(hidden_n - hidden_n_plus_1[:, :-1, :]).abs().max().item()}")
    assert torch.allclose(hidden_n, hidden_n_plus_1[:, :-1, :], atol=1e-4)