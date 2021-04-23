# -*- coding: utf-8 -*-
import datetime
from gurobipy import *
from itertools import product


def conv2d(model, layer, inputs):
    x0 = inputs['x0']
    pad_map = inputs[layer.input.name]
    weights = layer.get_weights()
    kernel = weights[0]
    bias = None
    if layer.use_bias:
        bias = weights[1]
    pd = 0 if layer.padding == 'VALID' else 1
    sh, sw = layer.strides
    inheight, inwidth, inchannels = tuple(layer.input_shape[1:4])
    outheight, outwidth, outchannels = tuple(layer.output_shape[1:4])
    kheight, kwidth, inchannels, outchannels = kernel.shape
    center_x, center_y = 0, 0
    padding_top, padding_left = 0, 0
    if pd == 1:
        padding_need_height = (outheight - 1) * sh + kheight - inheight
        padding_need_height = max(padding_need_height, 0)
        padding_top = padding_need_height // 2
        padding_bottom = padding_need_height - padding_top
        padding_need_width = (outwidth - 1) * sw + kwidth - inwidth
        padding_need_width = max(padding_need_width, 0)
        padding_left = padding_need_width // 2
        padding_right = padding_need_width - padding_left
        center_y = padding_top
        center_x = padding_left
        new_height = inheight + padding_need_height
        new_width = inwidth + padding_need_width
        new_pad_map = tupledict()
        for i, j, k in product(range(new_height), range(new_width), range(inchannels)):
            new_pad_map[i, j, k] = x0
        for i, j, k in product(range(inheight), range(inwidth), range(inchannels)):
            new_pad_map[i + center_y, j + center_x, k] = pad_map[i, j, k]
        pad_map = new_pad_map
    
    model.update()

    res_map = tupledict()

    start = datetime.datetime.now()
    for i, j, oc in product(range(outheight), range(outwidth), range(outchannels)):
        res_map[i, j, oc] = x0
        if layer.use_bias and bias is not None:
            res_map[i, j, oc] = LinExpr(bias[oc])
        if pd == 1:
            rol_s = i * sh + center_y - padding_top
            rol_e = rol_s + kheight
            col_s = j * sw + center_x - padding_left
            col_e = col_s + kwidth
        else:
            rol_s = i * sh + center_y
            rol_e = rol_s + kheight
            col_s = j * sw + center_x
            col_e = col_s + kwidth
        for r, c, ic in product(range(rol_s, rol_e), range(col_s, col_e), range(inchannels)):
            res_map[i, j, oc] += pad_map[r, c, ic] * kernel[r-rol_s, c-col_s, ic, oc]
    
    out_map = model.addVars(*(outheight, outwidth, outchannels), lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=1.0,
                            vtype=GRB.CONTINUOUS, name=layer.name + "_out_conv2d_vars")
    for idx in product(range(outheight), range(outwidth), range(outchannels)):
        nm = layer.name + '_out_conv2d_eq_constrs' + str(list(idx)).replace(' ', '')
        model.addLConstr(out_map[idx], GRB.EQUAL, res_map[idx], name=nm)
    end = datetime.datetime.now()
    print('conv2d spend time:', (end - start).seconds)
    model.update()
    inputs[layer.name] = out_map
    return model, layer, inputs
