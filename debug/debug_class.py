#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
title= "keras的debug调试"
author= "huangtw"
ctime = 2017/01/06
"""
from keras import backend as K
from keras.layers.core import *


def call_f(inp, method, input_data):
    f = K.function([inp], [method])
    return f([input_data])[0]


def print_out(layer, input_data, train=True):
    if hasattr(layer, 'previous'):
        print call_f(layer.previous.input,
                     layer.get_output(train=train), input_data)
    else:
        print call_f(layer.input, layer.get_output(train=train), input_data)


# print_out(Masking(mask_value=1), [[[1, 1, 0], [1, 1, 1]]])
# [[[ 1.  1.  0.], [ 0.  0.  0.]]]

# class CustomMasking(Masking):
#   def get_output_mask(self, train=False):
#     X = self.get_input(train)
#
#     return K.any(K.ones_like(X) * (1. -
#       K.equal(K.minimum(X, self.mask_value),
#         self.mask_value)), axis=-1)
#
#   def get_output(self, train=False):
#     X = self.get_input(train)
#
#     return X * K.any((1. - K.equal(
#       K.minimum(X, self.mask_value),
#         self.mask_value)), axis=-1, keepdims=True)

# print_out(CustomMasking(mask_value=5),
#   [[[3, 4, 5], [5, 6, 7], [5, 5, 5]]])
# # [[[ 3.  4.  5.], [ 0.  0.  0.], [ 0.  0.  0.]]]

print_out(Dropout(.3), [1, 2, 3, 4, 5])
# [0,0,0,5.71428585,7.14285755]
