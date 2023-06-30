#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = [
            'SPoSE',
            'l1_regularization',
            ]

import re
import torch
import torch.nn as nn
import torch.nn.functional as F

class SPoSE(nn.Module):
# class SPoSE(nn.Module):

#     def __init__(
#                 self,
#                 in_size:int,
#                 out_size:int,
#                 init_weights:bool=True,
#                 ):
#         super(SPoSE, self).__init__()
#         self.in_size = in_size
#         self.out_size = out_size
#         self.hidden_size = 618
#         self.fc1 = nn.Linear(self.in_size, self.hidden_size, bias=False)
#         self.fc2 = nn.Linear(self.hidden_size, self.out_size, bias=False)

#         if init_weights:
#             self._initialize_weights()

#     def forward(self, x:torch.Tensor) -> torch.Tensor:
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)
    def __init__(
                self,
                in_size:int,
                out_size:int,
                init_weights:bool=True,
                ):
        super(SPoSE, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        # self.hidden_size = 928 # in + out / 2
        # self.fc1 = nn.Linear(self.in_size, self.hidden_size, bias=False)
        # self.fc2 = nn.Linear(self.hidden_size, self.out_size, bias=False)
        self.fc1 = nn.Linear(self.in_size, self.out_size, bias=False)

        if init_weights:
            self._initialize_weights()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, 0.2)
        return self.fc1(x)

    def _initialize_weights(self) -> None:
        mean, std = .1, .01
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean, std)

        # nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.xavier_normal_(self.fc2.weight)


def l1_regularization(model) -> torch.Tensor:
    l1_reg = torch.tensor(0., requires_grad=True)
    for n, p in model.named_parameters():
        if re.search(r'weight', n):
            l1_reg = l1_reg + torch.norm(p, 1)
    return l1_reg
