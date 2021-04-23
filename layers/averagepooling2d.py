# -*- coding: utf-8 -*-
import datetime
from gurobipy import *
from itertools import product


def averagepooling2d(model, layer, inputs):
    fmap = inputs[layer.input.name]
    x0 = inputs['x0']
    pd = 0 if layer.padding == 'VALID' else 1
    sh, sw = layer.strides
    inheight, inwidth, inchannels = layer.input_shape[1:4]
    outheight, outwidth, outchannels = layer.output_shape[1:4]
    kheight, kwidth = layer.pool_size
    pad_map = fmap
    center_x, center_y = 0, 0
    padding_top, padding_left = 0, 0
    if inchannels != outchannels:
        raise Exception("inchannels can not match outchannels!")
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
        pad_map = tupledict()
        for i, j, k in product(range(new_height), range(new_width), range(inchannels)):
            pad_map[i, j, k] = x0
        for i, j, k in product(range(inheight), range(inwidth), range(inchannels)):
            pad_map[i + center_y, j + center_x, k] = fmap[i, j, k]

    model.update()

    res_map = tupledict()
    start = datetime.datetime.now()
    pool_sum = kheight * kwidth
    out_map = model.addVars(*(outheight, outwidth, outchannels), lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=1.0,
                            vtype=GRB.CONTINUOUS, name=layer.name + "_out_vars")
    for i, j, oc in product(range(outheight), range(outwidth), range(outchannels)):
        res_map[i, j, oc] = x0
        if pd == 1:
            rol_s = i * sh + center_y - padding_top
            rol_e = i * sh + center_y - padding_top + kheight
            col_s = j * sw + center_x - padding_left
            col_e = j * sw + center_x - padding_left + kwidth
        else:
            rol_s = i * sh + center_y
            rol_e = i * sh + center_y + kheight
            col_s = j * sw + center_x
            col_e = j * sw + center_x + kwidth
        for r, c in product(range(rol_s, rol_e), range(col_s, col_e)):
            res_map[i, j, oc] += pad_map[r, c, oc]
        res_map[i, j, oc] /= pool_sum
        nm = layer.name + '_avg_out_eq_constrs' + str([i, j, oc]).replace(' ', '')
        model.addLConstr(out_map[i, j, oc], GRB.EQUAL, res_map[i, j, oc], name=nm)
    end = datetime.datetime.now()
    print('average pooling construct constraints spend time:', (end - start).seconds)
    model.update()
    inputs[layer.output.name] = out_map
    return inputs, model
