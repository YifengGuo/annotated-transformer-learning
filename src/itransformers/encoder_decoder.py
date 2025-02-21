#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : guoyifeng
# @Time     : 2025-02-20
# @Version  : 0.0.1


import torch.nn as nn


class EncoderDecoder(nn.Module):
    """
      A standard Encoder-Decoder architecture. Base for this and many
      other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        # 确保EncoderDecoder类正确继承和初始化其父类nn.Module
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder  # 将传入的编码器实例保存为类属性
        self.decoder = decoder  # 将传入的解码器实例保存为类属性
        self.src_embed = src_embed  # 将传入的源嵌入实例保存为类属性
        self.tgt_embed = tgt_embed  # 将传入的目标嵌入实例保存为类属性
        self.generator = generator  # 将传入的生成器实例保存为类属性


    # 前向传播方法，接收源序列、目标序列和它们的掩码作为参数
    def forward(self, src, tgt, src_mask, tgt_mask):
        # 对源序列进行编码，并将编码结果与掩码传递给解码器进行解码
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    # 编码方法，接收源序列和掩码作为参数
    def encode(self, src, src_mask):
        # 将源序列进行嵌入，然后将嵌入后的序列和源序列掩码传给编码器
        return self.encoder(self.src_embed(src), src_mask)

    # 解码方法，接收编码器输出（memory）、源序列掩码、目标序列和目标序列掩码作为参数
    def decode(self, memory, src_mask, tgt, tgt_mask):
        # 将目标序列进行嵌入，然后将嵌入后的序列、编码器输出、源序列掩码和目标序列掩码传给解码器
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
