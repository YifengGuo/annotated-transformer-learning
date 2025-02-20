#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : guoyifeng
# @Time     : 2025-02-20
# @Version  : 0.0.1

import torch.nn as nn
import copy


class LayerUtil:

    @staticmethod
    def clones(module, N):
        """
        Produce N identical layers.
        The encoder is composed of a stack of N=6 identical layers.
        """
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
