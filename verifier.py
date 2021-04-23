# -*- coding: utf-8 -*-

import datetime
from gurobipy import *
from itertools import product
import numpy as np
import math
from layers import *

def validate_inputs(lp_model, inputs, layer):
    if lp_model is None:
        raise Exception("solver is empty")
    if layer is None:
        raise Exception("layer is empty")
    is_var = False
    if layer.input.name not in inputs:
        start = datetime.datetime.now()
        input_shape = layer.input_shape[1:]
        print('input_shape: ', input_shape)
        x = lp_model.addVars(*input_shape, obj=1.0, vtype=GRB.CONTINUOUS, name="input_vars")
        ub, lb = inputs['ub_img'], inputs['lb_img']
        if input_shape != ub.shape:
            raise Exception("input_shape is different with the dimension of image")
        ndim = ub.ndim
        print('ndim: ', ndim)
        if ndim == 1:
            h = input_shape[0]
            iter_item = range(h)
        elif ndim == 3:
            h, w, c = input_shape
            iter_item = product(range(h), range(w), range(c))
        else:
            raise Exception("not support dimenstion: ", ndim)
        
        # set upper and lower bound to variable
        for idx in iter_item:
            x[idx].setAttr(GRB.Attr.LB, lb[idx])
            x[idx].setAttr(GRB.Attr.UB, ub[idx])
        inputs['input_shape'] = input_shape
        inputs[layer.input.name] = x
        inputs['input_vars'] = x
        x0 = LinExpr(0.0)
        inputs['x0'] = x0
        is_var = True

        end = datetime.datetime.now()
        print('create variable time: ', (end-start).seconds)

    return inputs, lp_model, is_var

def set_objective(lp_model, constr):
    lp_model.setObjective(constr, GRB.MINIMIZE)
    lp_model.optimize()
    min_value = lp_model.objVal
    return min_value, lp_model

def get_solution_value(input_vars, input_shape):
    ndim = len(input_shape)
    if ndim == 1:
        h = input_shape[0]
        iter_item = range(h)
    elif ndim == 3:
        h, w, c = input_shape
        iter_item = product(range(h), range(w), range(c))
    else:
        raise Exception("input_shape dimension abnormal")
    values = np.zeros(input_shape)
    for idx in iter_item:
        values[idx] = input_vars[idx].x
    return values

def predict_label(nn, adv_image, net_type, dataset):
    adv_image_predict_label = None

    if net_type == 'fnn':
        # here gtsrb image is gray
        if dataset in ['mnist', 'fashion_mnist']:
            shape = 28 * 28
            a = adv_image.reshape(1,shape)
            adv_image_predict_label = np.argmax(nn.predict(a))
        elif dataset == 'gtsrb':
            shape = 32 * 32
            a = adv_image.reshape(1,shape)
            adv_image_predict_label = np.argmax(nn.predict(a))
        elif dataset == 'cifar10':
            shape = 32 * 32 * 3
            a = adv_image.reshape(1,shape)
            adv_image_predict_label = np.argmax(nn.predict(a))
    elif net_type == 'cnn':
        if dataset in ['mnist', 'fashion_mnist', 'gtsrb']:
            a = adv_image[np.newaxis, :, :, :]
            adv_image_predict_label = np.argmax(nn.predict(a))
        elif dataset == 'cifar10':
            a = adv_image[np.newaxis, :, :, :]
            adv_image_predict_label = np.argmax(nn.predict(a))
    return adv_image_predict_label

def solve(lp_model, nn, inputs, label, net_type, dataset):
    # obtain variable of input layer
    input_vars = inputs['input_vars']
    # obtain variable of last layer
    last_layer_output_name = nn.layers[-1].name
    if last_layer_output_name not in inputs:
        raise Exception("The calculation of model has occurred error")
    classified_output = inputs[last_layer_output_name]
    length = len(inputs[last_layer_output_name])
    sort_adv_labels = []
    
    robust_flag = True
    adv_image = None
    adv_label = None
    false_positive = False
    
    for i in range(length):
        if i == label:
            continue
        else:
            constr = classified_output[label] - classified_output[i]
            min_value, m = set_objective(lp_model, constr)
            if m.status==GRB.Status.OPTIMAL and min_value <= 0:
                robust_flag = False
                adv_image = get_solution_value(input_vars, inputs['input_shape'])
                adv_label = predict_label(nn, adv_image, net_type, dataset)
                if adv_label != label:
                    false_positive = False
                    return robust_flag, adv_image, adv_label, false_positive
                else:
                    false_positive = True
            
    return robust_flag, adv_image, adv_label, false_positive

def new_model():
    env = Env()
    env.setParam(GRB.Param.LogToConsole, 0)
    env.setParam(GRB.Param.OptimalityTol, 1.0e-4)
    env.setParam(GRB.Param.FeasibilityTol, 1.0e-4)
    env.setParam(GRB.Param.MIPGapAbs, 1.0e-4)
    env.setParam(GRB.Param.Method, 3)
    env.setParam(GRB.Param.Presolve, 2)
    model = Model("lp_model", env)
    return model

def verify_network_with_solver(ub_img, lb_img, label, nn, net_type, dataset):
    lp_model = new_model()
    inputs = dict()
    inputs['ub_img'] = ub_img
    inputs['lb_img'] = lb_img
    for index, layer in enumerate(nn.layers):
        print(index, len(nn.layers)-1, layer.name.lower())
        layer_name = type(layer).__name__

        if index == 0:
            inputs, lp_model, _ = validate_inputs(lp_model, inputs, layer)

        if 'Conv2D' in layer_name:
            inputs, lp_model = activation(*conv2d(lp_model, layer, inputs))
        elif 'Dense' in layer_name:
            inputs, lp_model = activation(*dense(lp_model, layer, inputs))
        elif 'Activation' in layer_name:
            inputs, lp_model = activation(*skip(lp_model, layer, inputs))
        elif 'Flatten' in layer_name:
            inputs, lp_model = flatten(lp_model, layer, inputs)
        elif 'ReLU' in layer_name:
            #TODO
            continue
        elif 'ZeroPadding2D' in layer_name:
            #TODO
            continue
        elif 'BatchNormalization' in layer_name:
            #TODO
            continue
        elif 'MaxPooling2D' in layer_name:
            #TODO
            continue
        elif 'AveragePooling2D' in layer_name:
            inputs, lp_model = averagepooling2d(lp_model, layer, inputs)
        elif 'GlobalMaxPooling2D' in layer_name:
            #TODO
            continue
        elif 'GlobalAveragePooling2D' in layer_name:
            #TODO
            continue
        elif 'Dropout' in layer_name:
            inputs, lp_model = dropout(lp_model, layer, inputs)
        elif 'InputLayer' in layer_name:
            continue
        elif 'Add' in layer_name:
            #TODO
            continue
        elif 'Concatenate' in layer_name:
            #TODO
            continue
        else:
            print('not support layer', layer_name)
            raise Exception("not support layer")

    robust_flag, adv_image, adv_label, false_positive = solve(lp_model, nn, inputs, label, net_type, dataset)

    return robust_flag, adv_image, adv_label, false_positive
