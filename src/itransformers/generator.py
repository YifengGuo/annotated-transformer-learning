#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : guoyifeng
# @Time     : 2025-02-20
# @Version  : 0.0.1


import torch.nn as nn
from torch.nn.functional import log_softmax


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        """
            这一层是输出层，所以输出的shape是 feature_size * vocab_size, 最后通过log_softmax对vocab这个维度进行归一化，找到概率最大的词汇
            Examples:
            >>> m = nn.Linear(20, 30)
            >>> input = torch.randn(128, 20)
            >>> output = m(input)
            >>> print(output. size())
            torch.Size([128, 30])
        """
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        """
        这个函数首先计算softmax，然后将结果取自然对数（logarithm）。这样做的好处是可以增加数值稳定性，特别是在处理非常小的概率值时。
        log_softmax函数中的参数dim=-1指定了在哪个维度上进行softmax操作。在Python中，负数索引表示从数组的末尾开始计数，因此dim=-1指的是最后一个维度。
        例如，假设你有一个二维张量（tensor），其形状为(batch_size, num_classes)，其中batch_size是批次中样本的数量，
        num_classes是类别的数量。如果你在这个张量上调用log_softmax(dim=-1)，
        那么softmax操作将会沿着num_classes这个维度进行，即每个样本的输出向量都会进行softmax归一化。
        """
        return log_softmax(self.proj(x), dim=-1)
