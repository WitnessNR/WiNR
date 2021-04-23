# -*- coding: utf-8 -*-
import datetime
from gurobipy import *
from itertools import product
from functools import reduce


def flatten(model, layer, inputs):

    sym = inputs[layer.input.name]
    input_shape = layer.input_shape[1:]
    ndim = len(input_shape)
    assert ndim == 3, "The dimension of flatten must be 3"
    h, w, c = input_shape
    num = reduce(lambda x, y: x * y, input_shape)
    i = 0
    start = datetime.datetime.now()
    new_sym = model.addVars(num, lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=1.0, vtype=GRB.CONTINUOUS,
                                    name=layer.name + "_vars")
    for idx in product(range(h), range(w), range(c)):
        model.addLConstr(new_sym[i] == sym[idx], name=layer.name + "_constrs_" + str(i))
        i += 1
    assert i == num, "The calculation of output_shape occurs error"
    end = datetime.datetime.now()
    print('flatten spend time:', (end - start).seconds)
    inputs[layer.output.name] = new_sym
    return inputs, model
