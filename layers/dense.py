# -*- coding: utf-8 -*-
import datetime
from gurobipy import *
from itertools import product


def dense(model, layer, inputs):

    start = datetime.datetime.now()
    current_input_shape = layer.input_shape[1]
    print('current_input_shape ', current_input_shape)
    weights = layer.get_weights()
    w = weights[0]
    b = None
    if layer.use_bias:
        b = weights[1]
    sym = inputs[layer.input.name]
    new_sym = model.addVars(layer.units, lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=1.0, vtype=GRB.CONTINUOUS, name=layer.name+"_vars")
    for j in range(layer.units):
        tmp = 0
        if layer.use_bias and b is not None:
            tmp = b[j]
        for i in range(current_input_shape):
            tmp += sym[i] * w[i,j]
        model.addLConstr(new_sym[j]==tmp, name=layer.name+"_constrs_"+str(j))
    inputs[layer.name] = new_sym
    end = datetime.datetime.now()
    print('all dense layer symbolic prapogation time: ', (end - start).seconds)
    return model, layer, inputs
