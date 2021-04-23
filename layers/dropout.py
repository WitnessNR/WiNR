# -*- coding: utf-8 -*-

def dropout(model, layer, inputs):
    inputs[layer.output.name] = inputs[layer.input.name]
    return inputs, model
