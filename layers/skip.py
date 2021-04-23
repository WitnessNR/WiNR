# -*- coding: utf-8 -*-

def skip(model, layer, inputs):
    inputs[layer.name] = inputs[layer.input.name]
    return model, layer, inputs
