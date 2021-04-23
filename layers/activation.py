# -*- coding: utf-8 -*-
import datetime
from gurobipy import *
from itertools import product
from .utils import *
import math

def get_neuron_upper_bound(lp_model, neuron):
    lp_model.setObjective(neuron, GRB.MAXIMIZE)
    lp_model.optimize()
    ub = lp_model.objVal
    return ub

def get_neuron_lower_bound(lp_model, neuron):
    lp_model.setObjective(neuron, GRB.MINIMIZE)
    lp_model.optimize()
    lb = lp_model.objVal
    return lb

def activation(lp_model, layer, inputs):

    if layer.activation is None:
        inputs[layer.output.name] = inputs[layer.name]
        return inputs, lp_model

    start = datetime.datetime.now()
    length = len(layer.output_shape)
    if length == 2:
        h = layer.output_shape[1]
        iter_item = product(range(h))
    elif length == 4:
        h, w, c = layer.output_shape[1:]
        iter_item = product(range(h), range(w), range(c))
    else:
        raise Exception("not support dimenstion: ", length)

    act_name = layer.activation.__name__
    if act_name == 'swish':
        sym = inputs[layer.name]
        t = -1.2784645
        m1, m2 = -2.399357, 2.399357

        print("Computing the upper and lower bound of the neuron before going through swish ...")
        new_sym = lp_model.addVars(*(tuple(layer.output_shape[1:])), lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=1.0, vtype=GRB.CONTINUOUS, name=layer.name+act_name+"_vars")
        for idx in iter_item:
            idx = idx[0] if len(idx)==1 else idx
            ub = get_neuron_upper_bound(lp_model, sym[idx])
            lb = get_neuron_lower_bound(lp_model, sym[idx])
            # print('ub: ', ub, 'lb: ', lb)

            if (ub < m1) or (m2 < lb):
                mid = (lb + ub)/2
                px = p(mid, sym[idx])
                rx_l = r(mid, lb)
                rx_u = r(mid, ub)
                rx = max(abs(rx_l), abs(rx_u))
                err_var = lp_model.addVar(lb=-rx, ub=0, obj=1.0, vtype=GRB.CONTINUOUS, name=layer.name+act_name+str(idx)+"_err_var")
                lp_model.addLConstr(new_sym[idx]==px+err_var, name=layer.name+act_name+str(idx)+"_constr")

            elif (lb < m1) and (ub < t):
                px = p(m1, sym[idx])
                rx_l = r(m1, lb)
                rx_u = r(m1, ub)
                err_var = lp_model.addVar(lb=rx_l, ub=rx_u, obj=1.0, vtype=GRB.CONTINUOUS, name=layer.name+act_name+str(idx)+"_err_var")
                lp_model.addLConstr(new_sym[idx]==px+err_var, name=layer.name+act_name+str(idx)+"_constr")

            elif (lb < m1) and (ub < 0):
                rx_l = swish(lb)
                rx_u = swish(ub)
                rx_t = swish(t)
                rx = min(abs(rx_l), abs(rx_u))
                err_var = lp_model.addVar(lb=rx_t, ub=-rx, obj=1.0, vtype=GRB.CONTINUOUS, name=layer.name+act_name+str(idx)+"_err_var")
                lp_model.addLConstr(new_sym[idx]==err_var, name=layer.name+act_name+str(idx)+"_constr")

            elif (lb < m1) and (ub < m2):
                k = (swish(ub) - swish(lb))/(ub - lb)
                upper = ub
                px = k*(sym[idx] - lb) + swish(lb)
                tmpx = find_tangent_point(lb, upper, k)
                if tmpx == -1:
                    raise Exception("not find_tangent_point is equal with k! ")
                A, B = k, -1 
                C = (-1)*k*lb + swish(lb)
                tmpx_y0 = swish(tmpx)
                rx = k*(tmpx -lb) + swish(lb) - tmpx_y0
                err_var = lp_model.addVar(lb=-rx, ub=0, obj=1.0, vtype=GRB.CONTINUOUS, name=layer.name+act_name+str(idx)+"_err_var")
                lp_model.addLConstr(new_sym[idx], GRB.EQUAL, px+err_var, name=layer.name+act_name+str(idx)+"_constr")

            elif (lb < m1) and (m2 < ub):
                mid = 0
                px = p(mid, sym[idx])
                rx_l = r(mid, lb)
                rx_u = r(mid, ub)
                rx = max(rx_l, rx_u)
                err_var = lp_model.addVar(lb=0, ub=rx, obj=1.0, vtype=GRB.CONTINUOUS, name=layer.name+act_name+str(idx)+"_err_var")
                lp_model.addLConstr(new_sym[idx]==px+err_var, name=layer.name+act_name+str(idx)+"_constr")     
                
            elif (m1 < lb) and (ub < m2) :
                mid = (lb + ub)/2
                px = p(mid, sym[idx])
                rx_l = r(mid, lb)
                rx_u = r(mid, ub)
                rx = max(rx_l, rx_u)
                err_var = lp_model.addVar(lb=0, ub=rx, obj=1.0, vtype=GRB.CONTINUOUS, name=layer.name+act_name+str(idx)+"_err_var")
                lp_model.addLConstr(new_sym[idx]==px+err_var, name=layer.name+act_name+str(idx)+"_constr")      

            elif (m1 < lb < m2) and (m2 < ub):
                mid = m2
                px = p(mid, sym[idx])
                rx_l = r(mid, lb)
                rx_u = r(mid, ub)
                err_var = lp_model.addVar(lb=rx_u, ub=rx_l, obj=1.0, vtype=GRB.CONTINUOUS, name=layer.name+act_name+str(idx)+"_err_var")
                lp_model.addLConstr(new_sym[idx]==px+err_var, name=layer.name+act_name+str(idx)+"_constr")    
                
        inputs[layer.output.name] = new_sym
        end = datetime.datetime.now()
        print('linearing activation function time: ', (end - start).seconds)
        return inputs, lp_model

    elif act_name == 'relu':
        sym = inputs[layer.name]
        print("Computing the upper and lower bound of the neuron before going through relu ...")
        new_sym = lp_model.addVars(*(tuple(layer.output_shape[1:])), lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=1.0, vtype=GRB.CONTINUOUS, name=layer.name+act_name+"_vars")
        for idx in iter_item:
            idx = idx[0] if len(idx)==1 else idx
            ub = get_neuron_upper_bound(lp_model, sym[idx])
            lb = get_neuron_lower_bound(lp_model, sym[idx])
            #print('ub: ', ub, 'lb: ', lb)
            if ub <= 0:
                lp_model.addLConstr(new_sym[idx]==0, name=layer.name+act_name+str(idx)+"_constr")

            elif lb >= 0:
                lp_model.addLConstr(new_sym[idx]==sym[idx], name=layer.name+act_name+str(idx)+"_constr")

            else:
                k = ub/(ub-lb)
                px = k*(sym[idx] - lb)
                rx = (-1)*k*lb
                err_var = lp_model.addVar(lb=-rx, ub=0, obj=1.0, vtype=GRB.CONTINUOUS, name=layer.name+act_name+str(idx)+"_err_var")
                lp_model.addLConstr(new_sym[idx]==px+err_var, name=layer.name+act_name+str(idx)+"_constr")   

        inputs[layer.output.name] = new_sym
        return inputs, lp_model

    elif act_name == 'sigmoid':
        sym = inputs[layer.name]
        print("Computing the upper and lower bound of the neuron before going through sigmoid ...")
        new_sym = lp_model.addVars(*(tuple(layer.output_shape[1:])), lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=1.0, vtype=GRB.CONTINUOUS, name=layer.name+act_name+"_vars")
        for idx in iter_item:
            idx = idx[0] if len(idx)==1 else idx
            ub = get_neuron_upper_bound(lp_model, sym[idx])
            lb = get_neuron_lower_bound(lp_model, sym[idx])
            act = sigmoid
            actd = sigmoidd
            actid = sigmoidid
            actut = sigmoidut
            actlt = sigmoidlt
            #print('ub: ', ub, 'lb: ', lb)
            px = None
            beta_l, beta_u = None, None
            
            if math.isclose(ub,lb):
                alpha_u = actd(ub)
                alpha_l = actd(lb)
                beta_u = act(ub)-actd(ub)*ub
                beta_l = act(lb)-actd(lb)*lb
                px = alpha_u * sym[idx] 
                
            elif ub <= 0:
                alpha = (act(ub)-act(lb))/(ub-lb)
                d = sigmoidlow(lb, ub, alpha)
                alpha_u = alpha
                alpha_l = actd(d)
                beta_u = act(lb)-alpha*lb
                beta_l = act(d)-actd(d)*d
                px = alpha * sym[idx] 

            elif lb >= 0:
                alpha = (act(ub)-act(lb))/(ub-lb)
                d = sigmoidup(lb, ub, alpha)
                alpha_u = actd(d)
                alpha_l = alpha
                beta_u = act(d)-actd(d)*d
                beta_l = act(lb)-alpha*lb
                px = alpha * sym[idx]

            else:
                alpha = (act(ub)-act(lb))/(ub-lb)
                dU = actd(ub)
                dL = actd(lb)
                if act(ub)-dU*(ub-lb) < act(lb) and act(lb)+dL*(ub-lb) < act(ub):
                    alpha_u = alpha
                    beta_u = act(lb)-alpha*lb
                    d = sigmoidloww(lb, ub, alpha)
                    alpha_l = actd(d)
                    beta_l = act(d)-actd(d)*d
                    px = alpha * sym[idx]
                elif act(ub)-dU*(ub-lb) > act(lb) and act(lb)+dL*(ub-lb) > act(ub):
                    alpha_l = alpha
                    beta_l = act(lb)-alpha*lb
                    d = sigmoidupp(lb, ub, alpha)
                    alpha_u = actd(d)
                    beta_u = act(d)-actd(d)*d
                    px = alpha * sym[idx]

                else:
                    du = actut(lb, ub)
                    dus = (act(du)-act(lb))/(du-lb)
                    dut = actd(du)
                    alpha_u = min(dut, dus)

                    dl = actlt(lb, ub)
                    dls = (act(dl)-act(ub))/(dl-ub)
                    dlt = actd(dl)
                    alpha_l = min(dlt, dls)
                        
                    alpha = min(alpha_u, alpha_l)
                    beta_u = act(du)-alpha*du
                    beta_l = act(dl)-alpha*dl
                    px = alpha * sym[idx]  
                    
            if math.isnan(beta_l):
                beta_l = 0.0
            if math.isnan(beta_u):
                beta_u = 0.0

            err_var = lp_model.addVar(lb=beta_l, ub=beta_u, obj=1.0, vtype=GRB.CONTINUOUS, name=layer.name+act_name+str(idx)+"_err_var")
            lp_model.addLConstr(new_sym[idx]==px+err_var, name=layer.name+act_name+str(idx)+"_constr") 
            
        inputs[layer.output.name] = new_sym
        return inputs, lp_model

    elif act_name in ['softmax','linear']:
        inputs[layer.output.name] = inputs[layer.name]
        return inputs, lp_model
    else:
        raise Exception("not support activation: ", act_name)
