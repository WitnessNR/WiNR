from numba import njit
import numpy as np
import matplotlib.pyplot as plt

from solve import *

# from tensorflow.contrib.keras.api.keras.models import Sequential
# from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, Lambda
# from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, InputLayer, BatchNormalization, Reshape
# from tensorflow.contrib.keras.api.keras.models import load_model
# from tensorflow.contrib.keras.api.keras import backend as K
# from tensorflow.contrib.keras.api.keras.datasets import mnist, cifar10

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, InputLayer, BatchNormalization, Reshape
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist, cifar10
import tensorflow as tf


from utils import generate_data_myself
import time
from activations import sigmoid_linear_bounds
from pgd_attack import *
linear_bounds = None

import random

def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)
class CNNModel:
    def __init__(self, model, inp_shape = (28,28,1)):
        print('-----------', inp_shape, '---------')
        temp_weights = [layer.get_weights() for layer in model.layers]

        self.weights = []
        self.biases = []
        self.shapes = []
        self.pads = []
        self.strides = []
        self.model = model
        
        cur_shape = inp_shape
        self.shapes.append(cur_shape)
        for layer in model.layers:
            print(cur_shape)
            weights = layer.get_weights()
            if type(layer) == Conv2D:
                print('conv')
                if len(weights) == 1:
                    W = weights[0].astype(np.float32)
                    b = np.zeros(W.shape[-1], dtype=np.float32)
                else:
                    W, b = weights
                    W = W.astype(np.float32)
                    b = b.astype(np.float32)
                padding = layer.get_config()['padding']
                stride = layer.get_config()['strides']
                pad = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                if padding == 'same':
                    desired_h = int(np.ceil(cur_shape[0]/stride[0]))
                    desired_w = int(np.ceil(cur_shape[0]/stride[1]))
                    total_padding_h = stride[0]*(desired_h-1)+W.shape[0]-cur_shape[0]
                    total_padding_w = stride[1]*(desired_w-1)+W.shape[1]-cur_shape[1]
                    pad = (int(np.floor(total_padding_h/2)),int(np.ceil(total_padding_h/2)),int(np.floor(total_padding_w/2)),int(np.ceil(total_padding_w/2)))
                cur_shape = (int((cur_shape[0]+pad[0]+pad[1]-W.shape[0])/stride[0])+1, int((cur_shape[1]+pad[2]+pad[3]-W.shape[1])/stride[1])+1, W.shape[-1])
                self.strides.append(stride)
                self.pads.append(pad)
                self.shapes.append(cur_shape)
                self.weights.append(W)
                self.biases.append(b)
            elif type(layer) == GlobalAveragePooling2D:
                print('global avg pool')
                b = np.zeros(cur_shape[-1], dtype=np.float32)
                W = np.zeros((cur_shape[0],cur_shape[1],cur_shape[2],cur_shape[2]), dtype=np.float32)
                for f in range(W.shape[2]):
                    W[:,:,f,f] = 1/(cur_shape[0]*cur_shape[1])
                pad = (0,0,0,0)
                stride = ((1,1))
                cur_shape = (1,1,cur_shape[2])
                self.strides.append(stride)
                self.pads.append(pad)
                self.shapes.append(cur_shape)
                self.weights.append(W)
                self.biases.append(b)
            elif type(layer) == AveragePooling2D:
                print('avg pool')
                b = np.zeros(cur_shape[-1], dtype=np.float32)
                pool_size = layer.get_config()['pool_size']
                stride = layer.get_config()['strides']
                W = np.zeros((pool_size[0],pool_size[1],cur_shape[2],cur_shape[2]), dtype=np.float32)
                for f in range(W.shape[2]):
                    W[:,:,f,f] = 1/(pool_size[0]*pool_size[1])
                pad = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                if padding == 'same':
                    desired_h = int(np.ceil(cur_shape[0]/stride[0]))
                    desired_w = int(np.ceil(cur_shape[0]/stride[1]))
                    total_padding_h = stride[0]*(desired_h-1)+pool_size[0]-cur_shape[0]
                    total_padding_w = stride[1]*(desired_w-1)+pool_size[1]-cur_shape[1]
                    pad = (int(np.floor(total_padding_h/2)),int(np.ceil(total_padding_h/2)),int(np.floor(total_padding_w/2)),int(np.ceil(total_padding_w/2)))
                cur_shape = (int((cur_shape[0]+pad[0]+pad[1]-pool_size[0])/stride[0])+1, int((cur_shape[1]+pad[2]+pad[3]-pool_size[1])/stride[1])+1, cur_shape[2])
                self.strides.append(stride)
                self.pads.append(pad)
                self.shapes.append(cur_shape)
                self.weights.append(W)
                self.biases.append(b)
            elif type(layer) == Activation:
                print('activation')
            elif type(layer) == Lambda:
	            print('lambda')
            elif type(layer) == InputLayer:
                print('input')
            elif type(layer) == BatchNormalization:
                print('batch normalization')
                gamma, beta, mean, std = weights
                std = np.sqrt(std+0.001) #Avoids zero division
                a = gamma/std
                b = -gamma*mean/std+beta
                self.weights[-1] = a*self.weights[-1]
                self.biases[-1] = a*self.biases[-1]+b
            elif type(layer) == Dense:
                print('FC')
                W, b = weights
                b = b.astype(np.float32)
                W = W.reshape(list(cur_shape)+[W.shape[-1]]).astype(np.float32)
                cur_shape = (1,1,W.shape[-1])
                self.strides.append((1,1))
                self.pads.append((0,0,0,0))
                self.shapes.append(cur_shape)
                self.weights.append(W)
                self.biases.append(b)
            elif type(layer) == Dropout:
                print('dropout')
            elif type(layer) == MaxPooling2D:
                print('pool')
                pool_size = layer.get_config()['pool_size']
                stride = layer.get_config()['strides']
                pad = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                if padding == 'same':
                    desired_h = int(np.ceil(cur_shape[0]/stride[0]))
                    desired_w = int(np.ceil(cur_shape[0]/stride[1]))
                    total_padding_h = stride[0]*(desired_h-1)+pool_size[0]-cur_shape[0]
                    total_padding_w = stride[1]*(desired_w-1)+pool_size[1]-cur_shape[1]
                    pad = (int(np.floor(total_padding_h/2)),int(np.ceil(total_padding_h/2)),int(np.floor(total_padding_w/2)),int(np.ceil(total_padding_w/2)))
                cur_shape = (int((cur_shape[0]+pad[0]+pad[1]-pool_size[0])/stride[0])+1, int((cur_shape[1]+pad[2]+pad[3]-pool_size[1])/stride[1])+1, cur_shape[2])
                self.strides.append(stride)
                self.pads.append(pad)
                self.shapes.append(cur_shape)
                self.weights.append(np.full(pool_size+(1,1),np.nan,dtype=np.float32))
                self.biases.append(np.full(1,np.nan,dtype=np.float32))
            elif type(layer) == Flatten:
                print('flatten')
            elif type(layer) == Reshape:
                print('reshape')
            else:
                print(str(type(layer)))
                raise ValueError('Invalid Layer Type')
        print(cur_shape)

        for i in range(len(self.weights)):
            self.weights[i] = np.ascontiguousarray(self.weights[i].transpose((3,0,1,2)).astype(np.float32))
            self.biases[i] = np.ascontiguousarray(self.biases[i].astype(np.float32))
    def predict(self, data):
        return self.model(data)


@njit
def conv(W, x, pad, stride):
    p_hl, p_hr, p_wl, p_wr = pad
    s_h, s_w = stride
    y = np.zeros((int((x.shape[0]-W.shape[1]+p_hl+p_hr)/s_h)+1, int((x.shape[1]-W.shape[2]+p_wl+p_wr)/s_w)+1, W.shape[0]), dtype=np.float32)
    for a in range(y.shape[0]):
        for b in range(y.shape[1]):
            for c in range(y.shape[2]):
                for i in range(W.shape[1]):
                    for j in range(W.shape[2]):
                        for k in range(W.shape[3]):
                            if 0<=s_h*a+i-p_hl<x.shape[0] and 0<=s_w*b+j-p_wl<x.shape[1]:
                                y[a,b,c] += W[c,i,j,k]*x[s_h*a+i-p_hl,s_w*b+j-p_wl,k]
    return y

@njit
def pool(pool_size, x0, pad, stride):
    p_hl, p_hr, p_wl, p_wr = pad
    s_h, s_w = stride
    y0 = np.zeros((int((x0.shape[0]+p_hl+p_hr-pool_size[0])/s_h)+1, int((x0.shape[1]+p_wl+p_wr-pool_size[1])/s_w)+1, x0.shape[2]), dtype=np.float32)
    for x in range(y0.shape[0]):
        for y in range(y0.shape[1]):
            for r in range(y0.shape[2]):
                cropped = LB[s_h*x-p_hl:pool_size[0]+s_h*x-p_hl, s_w*y-p_wl:pool_size[1]+s_w*y-p_wl,r]
                y0[x,y,r] = cropped.max()
    return y0

@njit
def conv_bound(W, b, pad, stride, x0, eps, p_n):
    y0 = conv(W, x0, pad, stride)
    UB = np.zeros(y0.shape, dtype=np.float32)
    LB = np.zeros(y0.shape, dtype=np.float32)
    for k in range(W.shape[0]):
        if p_n == 105: # p == "i", q = 1
            dualnorm = np.sum(np.abs(W[k,:,:,:]))
        elif p_n == 1: # p = 1, q = i
            dualnorm = np.max(np.abs(W[k,:,:,:]))
        elif p_n == 2: # p = 2, q = 2
            dualnorm = np.sqrt(np.sum(W[k,:,:,:]**2))
        mid = y0[:,:,k]+b[k]
        UB[:,:,k] = mid+eps*dualnorm
        LB[:,:,k] = mid-eps*dualnorm
    return LB, UB

@njit
def conv_full(A, x, pad, stride):
    p_hl, p_hr, p_wl, p_wr = pad
    s_h, s_w = stride
    y = np.zeros((A.shape[0], A.shape[1], A.shape[2]), dtype=np.float32)
    for a in range(y.shape[0]):
        for b in range(y.shape[1]):
            for c in range(y.shape[2]):
                for i in range(A.shape[3]):
                    for j in range(A.shape[4]):
                        for k in range(A.shape[5]):
                            if 0<=s_h*a+i-p_hl<x.shape[0] and 0<=s_w*b+j-p_wl<x.shape[1]:
                                y[a,b,c] += A[a,b,c,i,j,k]*x[s_h*a+i-p_hl,s_w*b+j-p_wl,k]
    return y

@njit
def conv_bound_full(A, B, pad, stride, x0, eps, p_n):
    y0 = conv_full(A, x0, pad, stride)
    UB = np.zeros(y0.shape, dtype=np.float32)
    LB = np.zeros(y0.shape, dtype=np.float32)
    for a in range(y0.shape[0]):
        for b in range(y0.shape[1]):
            for c in range(y0.shape[2]):
                if p_n == 105: # p == "i", q = 1
                    dualnorm = np.sum(np.abs(A[a,b,c,:,:,:]))
                elif p_n == 1: # p = 1, q = i
                    dualnorm = np.max(np.abs(A[a,b,c,:,:,:]))
                elif p_n == 2: # p = 2, q = 2
                    dualnorm = np.sqrt(np.sum(A[a,b,c,:,:,:]**2))
                mid = y0[a,b,c]+B[a,b,c]
                UB[a,b,c] = mid+eps*dualnorm
                LB[a,b,c] = mid-eps*dualnorm
    return LB, UB

@njit
def upper_bound_conv(A, B, pad, stride, W, b, inner_pad, inner_stride, inner_shape, LB, UB):
    A_new = np.zeros((A.shape[0], A.shape[1], A.shape[2], inner_stride[0]*(A.shape[3]-1)+W.shape[1], inner_stride[1]*(A.shape[4]-1)+W.shape[2], W.shape[3]), dtype=np.float32)
    B_new = np.zeros(B.shape, dtype=np.float32)
    A_plus = np.maximum(A, 0)
    A_minus = np.minimum(A, 0)
    alpha_u, alpha_l, beta_u, beta_l = linear_bounds(LB, UB)
    assert A.shape[5] == W.shape[0]

    for x in range(A_new.shape[0]):
        for y in range(A_new.shape[1]):
            for t in range(A_new.shape[3]):
                for u in range(A_new.shape[4]):
                    if 0<=t+stride[0]*inner_stride[0]*x-inner_stride[0]*pad[0]-inner_pad[0]<inner_shape[0] and 0<=u+stride[1]*inner_stride[1]*y-inner_stride[1]*pad[2]-inner_pad[2]<inner_shape[1]:
                        for p in range(A.shape[3]):
                            for q in range(A.shape[4]):
                                if 0<=t-inner_stride[0]*p<W.shape[1] and 0<=u-inner_stride[1]*q<W.shape[2] and 0<=p+stride[0]*x-pad[0]<alpha_u.shape[0] and 0<=q+stride[1]*y-pad[2]<alpha_u.shape[1]:
                                    for z in range(A_new.shape[2]):
                                        for v in range(A_new.shape[5]):
                                            for r in range(W.shape[0]):
                                                A_new[x,y,z,t,u,v] += W[r,t-inner_stride[0]*p,u-inner_stride[1]*q,v]*alpha_u[p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],r]*A_plus[x,y,z,p,q,r]
                                                A_new[x,y,z,t,u,v] += W[r,t-inner_stride[0]*p,u-inner_stride[1]*q,v]*alpha_l[p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],r]*A_minus[x,y,z,p,q,r]
                                                
    B_new = conv_full(A_plus,alpha_u*b+beta_u,pad,stride) + conv_full(A_minus,alpha_l*b+beta_l,pad,stride)+B
    return A_new, B_new

@njit
def lower_bound_conv(A, B, pad, stride, W, b, inner_pad, inner_stride, inner_shape, LB, UB):
    A_new = np.zeros((A.shape[0], A.shape[1], A.shape[2], inner_stride[0]*(A.shape[3]-1)+W.shape[1], inner_stride[1]*(A.shape[4]-1)+W.shape[2], W.shape[3]), dtype=np.float32)
    B_new = np.zeros(B.shape, dtype=np.float32)
    A_plus = np.maximum(A, 0)
    A_minus = np.minimum(A, 0)
    alpha_u, alpha_l, beta_u, beta_l = linear_bounds(LB, UB)
    assert A.shape[5] == W.shape[0]
    for x in range(A_new.shape[0]):
        for y in range(A_new.shape[1]):
            for t in range(A_new.shape[3]):
                for u in range(A_new.shape[4]):
                    if 0<=t+stride[0]*inner_stride[0]*x-inner_stride[0]*pad[0]-inner_pad[0]<inner_shape[0] and 0<=u+stride[1]*inner_stride[1]*y-inner_stride[1]*pad[2]-inner_pad[2]<inner_shape[1]:
                        for p in range(A.shape[3]):
                            for q in range(A.shape[4]):
                                if 0<=t-inner_stride[0]*p<W.shape[1] and 0<=u-inner_stride[1]*q<W.shape[2] and 0<=p+stride[0]*x-pad[0]<alpha_u.shape[0] and 0<=q+stride[1]*y-pad[2]<alpha_u.shape[1]:
                                    for z in range(A_new.shape[2]):
                                        for v in range(A_new.shape[5]):
                                            for r in range(W.shape[0]):
                                                A_new[x,y,z,t,u,v] += W[r,t-inner_stride[0]*p,u-inner_stride[1]*q,v]*alpha_l[p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],r]*A_plus[x,y,z,p,q,r]
                                                A_new[x,y,z,t,u,v] += W[r,t-inner_stride[0]*p,u-inner_stride[1]*q,v]*alpha_u[p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],r]*A_minus[x,y,z,p,q,r]
    B_new = conv_full(A_plus,alpha_l*b+beta_l,pad,stride) + conv_full(A_minus,alpha_u*b+beta_u,pad,stride)+B
    return A_new, B_new

@njit
def pool_linear_bounds(LB, UB, pad, stride, pool_size):
    p_hl, p_hr, p_wl, p_wr = pad
    s_h, s_w = stride
    alpha_u = np.zeros((pool_size[0], pool_size[1], int((UB.shape[0]+p_hl+p_hr-pool_size[0])/s_h)+1, int((UB.shape[1]+p_wl+p_wr-pool_size[1])/s_w)+1, UB.shape[2]), dtype=np.float32)
    beta_u = np.zeros((int((UB.shape[0]+p_hl+p_hr-pool_size[0])/s_h)+1, int((UB.shape[1]+p_wl+p_wr-pool_size[1])/s_w)+1, UB.shape[2]), dtype=np.float32)
    alpha_l = np.zeros((pool_size[0], pool_size[1], int((LB.shape[0]+p_hl+p_hr-pool_size[0])/s_h)+1, int((LB.shape[1]+p_wl+p_wr-pool_size[1])/s_w)+1, LB.shape[2]), dtype=np.float32)
    beta_l = np.zeros((int((LB.shape[0]+p_hl+p_hr-pool_size[0])/s_h)+1, int((LB.shape[1]+p_wl+p_wr-pool_size[1])/s_w)+1, LB.shape[2]), dtype=np.float32)

    for x in range(alpha_u.shape[2]):
        for y in range(alpha_u.shape[3]):
            for r in range(alpha_u.shape[4]):
                cropped_LB = LB[s_h*x-p_hl:pool_size[0]+s_h*x-p_hl, s_w*y-p_wl:pool_size[1]+s_w*y-p_wl,r]
                cropped_UB = UB[s_h*x-p_hl:pool_size[0]+s_h*x-p_hl, s_w*y-p_wl:pool_size[1]+s_w*y-p_wl,r]

                max_LB = cropped_LB.max()
                idx = np.where(cropped_UB>=max_LB)
                u_s = np.zeros(len(idx[0]), dtype=np.float32)
                l_s = np.zeros(len(idx[0]), dtype=np.float32)
                gamma = np.inf
                for i in range(len(idx[0])):
                    l_s[i] = cropped_LB[idx[0][i],idx[1][i]]
                    u_s[i] = cropped_UB[idx[0][i],idx[1][i]]
                    if l_s[i] == u_s[i]:
                        gamma = l_s[i]

                if gamma == np.inf:
                    gamma = (np.sum(u_s/(u_s-l_s))-1)/np.sum(1/(u_s-l_s))
                    if gamma < np.max(l_s):
                        gamma = np.max(l_s)
                    elif gamma > np.min(u_s):
                        gamma = np.min(u_s)
                    weights = ((u_s-gamma)/(u_s-l_s)).astype(np.float32)
                else:
                    weights = np.zeros(len(idx[0]), dtype=np.float32)
                    w_partial_sum = 0
                    num_equal = 0
                    for i in range(len(idx[0])):
                        if l_s[i] != u_s[i]:
                            weights[i] = (u_s[i]-gamma)/(u_s[i]-l_s[i])
                            w_partial_sum += weights[i]
                        else:
                            num_equal += 1
                    gap = (1-w_partial_sum)/num_equal
                    if gap < 0.0:
                        gap = 0.0
                    elif gap > 1.0:
                        gap = 1.0
                    for i in range(len(idx[0])):
                        if l_s[i] == u_s[i]:
                            weights[i] = gap

                for i in range(len(idx[0])):
                    t = idx[0][i]
                    u = idx[1][i]
                    alpha_u[t,u,x,y,r] = weights[i]
                    alpha_l[t,u,x,y,r] = weights[i]
                beta_u[x,y,r] = gamma-np.dot(weights, l_s)
                growth_rate = np.sum(weights)
                if growth_rate <= 1:
                    beta_l[x,y,r] = np.min(l_s)*(1-growth_rate)
                else:
                    beta_l[x,y,r] = np.max(u_s)*(1-growth_rate)
    return alpha_u, alpha_l, beta_u, beta_l

@njit
def upper_bound_pool(A, B, pad, stride, pool_size, inner_pad, inner_stride, inner_shape, LB, UB):
    A_new = np.zeros((A.shape[0], A.shape[1], A.shape[2], inner_stride[0]*(A.shape[3]-1)+pool_size[0], inner_stride[1]*(A.shape[4]-1)+pool_size[1], A.shape[5]), dtype=np.float32)
    B_new = np.zeros(B.shape, dtype=np.float32)
    A_plus = np.maximum(A, 0)
    A_minus = np.minimum(A, 0)
    alpha_u, alpha_l, beta_u, beta_l = pool_linear_bounds(LB, UB, inner_pad, inner_stride, pool_size)

    for x in range(A_new.shape[0]):
        for y in range(A_new.shape[1]):
            for t in range(A_new.shape[3]):
                for u in range(A_new.shape[4]):
                    inner_index_x = t+stride[0]*inner_stride[0]*x-inner_stride[0]*pad[0]-inner_pad[0]
                    inner_index_y = u+stride[1]*inner_stride[1]*y-inner_stride[1]*pad[2]-inner_pad[2]
                    if 0<=inner_index_x<inner_shape[0] and 0<=inner_index_y<inner_shape[1]:
                        for p in range(A.shape[3]):
                            for q in range(A.shape[4]):
                                if 0<=t-inner_stride[0]*p<alpha_u.shape[0] and 0<=u-inner_stride[1]*q<alpha_u.shape[1] and 0<=p+stride[0]*x-pad[0]<alpha_u.shape[2] and 0<=q+stride[1]*y-pad[2]<alpha_u.shape[3]:
                                    A_new[x,y,:,t,u,:] += A_plus[x,y,:,p,q,:]*alpha_u[t-inner_stride[0]*p,u-inner_stride[1]*q,p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],:]
                                    A_new[x,y,:,t,u,:] += A_minus[x,y,:,p,q,:]*alpha_l[t-inner_stride[0]*p,u-inner_stride[1]*q,p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],:]
    B_new = conv_full(A_plus,beta_u,pad,stride) + conv_full(A_minus,beta_l,pad,stride)+B
    return A_new, B_new

@njit
def lower_bound_pool(A, B, pad, stride, pool_size, inner_pad, inner_stride, inner_shape, LB, UB):
    A_new = np.zeros((A.shape[0], A.shape[1], A.shape[2], inner_stride[0]*(A.shape[3]-1)+pool_size[0], inner_stride[1]*(A.shape[4]-1)+pool_size[1], A.shape[5]), dtype=np.float32)
    B_new = np.zeros(B.shape, dtype=np.float32)
    A_plus = np.maximum(A, 0)
    A_minus = np.minimum(A, 0)
    alpha_u, alpha_l, beta_u, beta_l = pool_linear_bounds(LB, UB, inner_pad, inner_stride, pool_size)

    for x in range(A_new.shape[0]):
        for y in range(A_new.shape[1]):
            for t in range(A_new.shape[3]):
                for u in range(A_new.shape[4]):
                    inner_index_x = t+stride[0]*inner_stride[0]*x-inner_stride[0]*pad[0]-inner_pad[0]
                    inner_index_y = u+stride[1]*inner_stride[1]*y-inner_stride[1]*pad[2]-inner_pad[2]
                    if 0<=inner_index_x<inner_shape[0] and 0<=inner_index_y<inner_shape[1]:
                        for p in range(A.shape[3]):
                            for q in range(A.shape[4]):
                                if 0<=t-inner_stride[0]*p<alpha_u.shape[0] and 0<=u-inner_stride[1]*q<alpha_u.shape[1] and 0<=p+stride[0]*x-pad[0]<alpha_u.shape[2] and 0<=q+stride[1]*y-pad[2]<alpha_u.shape[3]:
                                    A_new[x,y,:,t,u,:] += A_plus[x,y,:,p,q,:]*alpha_l[t-inner_stride[0]*p,u-inner_stride[1]*q,p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],:]
                                    A_new[x,y,:,t,u,:] += A_minus[x,y,:,p,q,:]*alpha_u[t-inner_stride[0]*p,u-inner_stride[1]*q,p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],:]
    B_new = conv_full(A_plus,beta_l,pad,stride) + conv_full(A_minus,beta_u,pad,stride)+B
    return A_new, B_new

@njit
def compute_bounds(weights, biases, out_shape, nlayer, x0, eps, p_n, strides, pads, LBs, UBs):
    pad = (0,0,0,0)
    stride = (1,1)
    modified_LBs = LBs + (np.ones(out_shape, dtype=np.float32),)
    modified_UBs = UBs + (np.ones(out_shape, dtype=np.float32),)
    for i in range(nlayer-1, -1, -1):
        if not np.isnan(weights[i]).any(): #Conv
            if i == nlayer-1:
                A_u = weights[i].reshape((1, 1, weights[i].shape[0], weights[i].shape[1], weights[i].shape[2], weights[i].shape[3]))*np.ones((out_shape[0], out_shape[1], weights[i].shape[0], weights[i].shape[1], weights[i].shape[2], weights[i].shape[3]), dtype=np.float32)
                B_u = biases[i]*np.ones((out_shape[0], out_shape[1], out_shape[2]), dtype=np.float32)
                A_l = A_u.copy()
                B_l = B_u.copy()
            else:
                A_u, B_u = upper_bound_conv(A_u, B_u, pad, stride, weights[i], biases[i], pads[i], strides[i], modified_UBs[i].shape, modified_LBs[i+1], modified_UBs[i+1])
                A_l, B_l = lower_bound_conv(A_l, B_l, pad, stride, weights[i], biases[i], pads[i], strides[i], modified_LBs[i].shape, modified_LBs[i+1], modified_UBs[i+1])
        else: #Pool
            if i == nlayer-1:
                A_u = np.eye(out_shape[2]).astype(np.float32).reshape((1,1,out_shape[2],1,1,out_shape[2]))*np.ones((out_shape[0], out_shape[1], out_shape[2], 1,1,out_shape[2]), dtype=np.float32)
                B_u = np.zeros(out_shape, dtype=np.float32)
                A_l = A_u.copy()
                B_l = B_u.copy()
            A_u, B_u = upper_bound_pool(A_u, B_u, pad, stride, weights[i].shape[1:], pads[i], strides[i], modified_UBs[i].shape, np.maximum(modified_LBs[i],0), np.maximum(modified_UBs[i],0))
            A_l, B_l = lower_bound_pool(A_l, B_l, pad, stride, weights[i].shape[1:], pads[i], strides[i], modified_LBs[i].shape, np.maximum(modified_LBs[i],0), np.maximum(modified_UBs[i],0))
        pad = (strides[i][0]*pad[0]+pads[i][0], strides[i][0]*pad[1]+pads[i][1], strides[i][1]*pad[2]+pads[i][2], strides[i][1]*pad[3]+pads[i][3])
        stride = (strides[i][0]*stride[0], strides[i][1]*stride[1])
    LUB, UUB = conv_bound_full(A_u, B_u, pad, stride, x0, eps, p_n)
    LLB, ULB = conv_bound_full(A_l, B_l, pad, stride, x0, eps, p_n)
    return LLB, ULB, LUB, UUB, A_u, A_l, B_u, B_l, pad, stride

def find_output_bounds(weights, biases, shapes, pads, strides, x0, eps, p_n):
    LB, UB = conv_bound(weights[0], biases[0], pads[0], strides[0], x0, eps, p_n)
    LBs = [x0-eps, LB]
    UBs = [x0+eps, UB]
    for i in range(2,len(weights)+1):
        LB, _, _, UB, A_u, A_l, B_u, B_l, pad, stride = compute_bounds(tuple(weights), tuple(biases), shapes[i], i, x0, eps, p_n, tuple(strides), tuple(pads), tuple(LBs), tuple(UBs))
        UBs.append(UB)
        LBs.append(LB)
    return LBs[-1], UBs[-1], A_u, A_l, B_u, B_l, pad, stride

ts = time.time()
timestr = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')
#Prints to log file
def printlog(s):
    print(s, file=open("logs/cnn_bounds_full_core_with_LP"+timestr+".txt", "a"))

def run(file_name, n_samples, eps_0, p_n, q_n, activation = 'sigmoid', cifar=False, fashion_mnist=False, gtsrb=False):
    np.random.seed(1215)
    #tf.set_random_seed(1215)
    random.seed(1215)
    keras_model = load_model(file_name, custom_objects={'fn':fn, 'tf':tf})

    if cifar:
        model = CNNModel(keras_model, inp_shape = (32,32,3))
    elif gtsrb:
        print('gtsrb')
        model = CNNModel(keras_model, inp_shape = (48,48,3))
    else:
        model = CNNModel(keras_model)
    print('--------abstracted model-----------')
    
    global linear_bounds
    linear_bounds = sigmoid_linear_bounds
    
    upper_bound_conv.recompile()
    lower_bound_conv.recompile()
    compute_bounds.recompile()

    dataset = ''
    
    if cifar:
        dataset = 'cifar10'
        inputs, targets, true_labels, true_ids = generate_data_myself('cifar10', model.model, samples=n_samples, start=0)
    elif gtsrb:
        dataset = 'gtsrb'
        inputs, targets, true_labels, true_ids = generate_data_myself('gtsrb', model.model, samples=n_samples, start=0)
    elif fashion_mnist:
        dataset = 'fashion_mnist'
        inputs, targets, true_labels, true_ids = generate_data_myself('fashion_mnist', model.model, samples=n_samples, start=0)
    else:
        dataset = 'mnist'
        inputs, targets, true_labels, true_ids = generate_data_myself('mnist', model.model, samples=n_samples, start=0)
        
    print('----------generated data---------')

    
    #eps_0 = 0.020
   
    printlog('===========================================')
    printlog("model name = {}".format(file_name))
    printlog("eps = {:.5f}".format(eps_0))
    
    time_limit = 2000
    
    DeepCert_robust_number = 0
    PGD_falsified_number = 0
    PGD_DeepCert_unknown_number = 0
    DeepCert_robust_img_id = []
    PGD_time = 0
    DeepCert_time = 0
    total_images = 0
    
    '''
    printlog("----------------PGD+DeepCert----------------")
    
    for i in range(len(inputs)):
        total_images += 1
        printlog("----------------image id = {}----------------".format(i))
        predict_label = np.argmax(true_labels[i])
        printlog("image predict label = {}".format(predict_label))
        
        printlog("----------------PGD----------------")
        PGD_start_time = time.time()
        # generate adversarial example using PGD
        PGD_flag = False
        predict_label_for_attack = predict_label.astype("float32")
        image = tf.constant(inputs[i])
        image = tf.expand_dims(image, axis=0)
        attack_kwargs = {"eps": eps_0, "alpha": eps_0/1000, "num_iter": 48, "restarts": 48}
        attack = PgdRandomRestart(model=keras_model, **attack_kwargs)
        attack_inputs = (image, tf.constant(predict_label_for_attack))
        adv_example = attack(*attack_inputs, time_limit=20, predict_label=predict_label)
        
        # judge whether the adv_example is true adversarial example
        adv_example_label = keras_model.predict(adv_example)
        adv_example_label = np.argmax(adv_example_label)
        if adv_example_label != predict_label:
            original_image = image.numpy()
            adv_example = adv_example.numpy()
            norm_fn = lambda x: np.max(np.abs(x),axis=(1,2,3))
            norm_diff = norm_fn(adv_example-original_image)
            printlog("PGD norm_diff(adv_example-original_example) = {}".format(norm_diff))
            PGD_flag = True
            PGD_falsified_number += 1
            #falsified_number += 1
            printlog("PGD adv_example_label = {}".format(adv_example_label))
            printlog("PGD attack succeed!")
        else:
            printlog("PGD attack failed!")
        PGD_time += (time.time() - PGD_start_time)
        
        if PGD_flag:
            continue
        
        printlog('----------------DeepCert----------------')
        DeepCert_start_time = time.time()
        DeepCert_flag = True
        for j in range(i*9,i*9+9):
            target_label = targets[j]
            printlog("target label = {}".format(target_label))
            
            weights = model.weights[:-1]
            biases = model.biases[:-1]
            shapes = model.shapes[:-1]
            W, b, s = model.weights[-1], model.biases[-1], model.shapes[-1]
            last_weight = (W[predict_label,:,:,:]-W[target_label,:,:,:]).reshape([1]+list(W.shape[1:]))
            weights.append(last_weight)
            biases.append(np.asarray([b[predict_label]-b[target_label]]))
            shapes.append((1,1,1))
                                                      
            LB, UB, A_u, A_l, B_u, B_l, pad, stride = find_output_bounds(weights, biases, shapes, model.pads, model.strides, inputs[i].astype(np.float32), eps_0, p_n)
            
            printlog("DeepCert:  {:.6s} <= f_c - f_t <= {:.6s}".format(str(np.squeeze(LB)),str(np.squeeze(UB))))
    
            if LB <= 0:
                DeepCert_flag = False 
                break

        if DeepCert_flag:
            DeepCert_robust_number += 1
            DeepCert_robust_img_id.append(i)
            printlog("DeepCert: robust")
        elif PGD_flag:
            pass
        else:
            PGD_DeepCert_unknown_number += 1
            printlog("DeepCert: unknown")
        
        DeepCert_time += (time.time()-DeepCert_start_time)
        
        printlog("PGD      - falsified: {}".format(PGD_flag))
        printlog("DeepCert - robust: {}, unknown: {}".format((DeepCert_flag and not(PGD_flag)), not(DeepCert_flag)))
        
        if (PGD_time+DeepCert_time)>=time_limit:
            printlog("[L1] PGD_DeepCert_total_time = {}, reach time limit!".format(PGD_time+DeepCert_time))
            break

    
    PGD_DeepCert_total_time = (PGD_time+DeepCert_time)
    PGD_aver_time = PGD_time / total_images
    DeepCert_aver_time = DeepCert_time / total_images
    PGD_DeepCert_aver_time = PGD_DeepCert_total_time / total_images
    printlog("[L0] method = PGD, average runtime = {:.3f}".format(PGD_aver_time))
    printlog("[L0] method = DeepCert, average runtime = {:.3f}".format(DeepCert_aver_time))
    printlog("[L0] method = PGD+DeepCert, eps = {}, total images = {}, robust = {}, falsified = {}, unknown = {}, average runtime = {:.3f}".format(eps_0, total_images, DeepCert_robust_number, PGD_falsified_number, PGD_DeepCert_unknown_number, PGD_DeepCert_aver_time))
    '''
    
    printlog("----------------WiNR----------------")
    WiNR_start_time = time.time()
    WiNR_robust_number = 0
    WiNR_falsified_number = 0
    WiNR_unknown_number = 0 
    verified_number = 0
    WiNR_robust_img_id = []
    WiNR_falsified_img_id = []
    total_images = 0
    
    for i in range(len(inputs)):
        total_images += 1
        
        printlog("----------------image id = {}----------------".format(i))
        predict_label = np.argmax(true_labels[i])
        printlog("image predict label = {}".format(predict_label))
        
        adv_false = []
        has_adv_false = False
        WiNR_robust_flag = True
        WiNR_falsified_flag = False
        
        for j in range(i*9,i*9+9):
            target_label = targets[j]
            printlog("target label = {}".format(target_label))
            
            weights = model.weights[:-1]
            biases = model.biases[:-1]
            shapes = model.shapes[:-1]
            W, b, s = model.weights[-1], model.biases[-1], model.shapes[-1]
            last_weight = (W[predict_label,:,:,:]-W[target_label,:,:,:]).reshape([1]+list(W.shape[1:]))
            weights.append(last_weight)
            biases.append(np.asarray([b[predict_label]-b[target_label]]))
            shapes.append((1,1,1))
                        
            LB, UB, A_u, A_l, B_u, B_l, pad, stride = find_output_bounds(weights, biases, shapes, model.pads, model.strides, inputs[i].astype(np.float32), eps_0, p_n)
            
            # solving
            lp_model = new_model()
            lp_model, x = creat_var(lp_model, inputs[i], eps_0)
            shape = inputs[i].shape
            adv_image, min_val = get_solution_value(lp_model, x, shape, A_u, A_l, B_u, B_l, pad, stride, p_n, eps_0)
            printlog("WiNR min_val={:.5f}".format(min_val))
            
            if min_val > 0:
                continue
            
            WiNR_robust_flag = False
            # label of potential counterexample
            a = adv_image[np.newaxis,:,:,:]
            aa = a.astype(np.float32)
            adv_label = np.argmax(np.squeeze(keras_model.predict(aa)))
            
            if adv_label == predict_label:
                adv_false.append((adv_image, target_label))
                has_adv_false = True
                printlog('this adv_example is false!')
                continue
            
            WiNR_diff = (adv_image-inputs[i]).reshape(-1)
            WiNR_diff = np.absolute(WiNR_diff)
            WiNR_diff = np.max(WiNR_diff)
            printlog("WiNR diff(adv_example-original_example) = {}".format(WiNR_diff))
            WiNR_falsified_flag = True
            break
        
        if WiNR_robust_flag:
            WiNR_robust_number += 1
            WiNR_robust_img_id.append(i)
            printlog("WiNR: robust")
        elif WiNR_falsified_flag:
            printlog("WiNR: falsified")
            WiNR_falsified_number += 1
            WiNR_falsified_img_id.append(i)
        else:
            printlog("WiNR: unknown")
            WiNR_unknown_number += 1
            
        printlog("WiNR - robust: {}, falsified: {}, unknown: {}".format(WiNR_robust_flag, WiNR_falsified_flag, (not(WiNR_robust_flag) and not(WiNR_falsified_flag))))
        end_time = (time.time()-WiNR_start_time)
        verified_number += 1
        if end_time >= time_limit:
            printlog("verifying time : {} sec, reach time limit {} sec.".format(end_time, time_limit))
            break
        
    WiNR_total_time = (time.time()-WiNR_start_time)
    WiNR_aver_time = WiNR_total_time / total_images
    printlog("[L0] method = WiNR, eps = {}, total images = {}, verified number = {}, robust = {}, falsified = {}, unknown = {}, average runtime = {:.3f}".format(eps_0, total_images, verified_number, WiNR_robust_number, WiNR_falsified_number, WiNR_unknown_number, WiNR_aver_time))
    printlog("[L0] DeepCert robust images id: {}".format(DeepCert_robust_img_id))
    printlog("[L0] WiNR robust images id: {}".format(WiNR_robust_img_id))
    
    printlog("----------------PGD+WiNR----------------")

    PGD_before_WiNR_falsified_number = 0
    WiNR_after_PGD_robust_number = 0
    WiNR_after_PGD_falsified_number = 0
    WiNR_after_PGD_unknown_number = 0 
    PGD_before_WiNR_time = 0
    WiNR_after_PGD_time = 0
    PGD_before_WiNR_falsified_img_id = []
    WiNR_after_PGD_falsified_img_id = []
    total_images = 0
    
    for i in range(len(inputs)):
        total_images += 1
        printlog("----------------image id = {}----------------".format(i))
        predict_label = np.argmax(true_labels[i])
        printlog("image predict label = {}".format(predict_label))
        
        printlog("----------------PGD(+WiNR)----------------")
        PGD_before_WiNR_start_time = time.time()
        # generate adversarial example using PGD
        PGD_before_WiNR_flag = False
        predict_label_for_attack = predict_label.astype("float32")
        image = tf.constant(inputs[i])
        image = tf.expand_dims(image, axis=0)
        attack_kwargs = {"eps": eps_0, "alpha": eps_0/1000, "num_iter": 48, "restarts": 48}
        attack = PgdRandomRestart(model=keras_model, **attack_kwargs)
        attack_inputs = (image, tf.constant(predict_label_for_attack))
        adv_example = attack(*attack_inputs, time_limit=20, predict_label=predict_label)
        
        # judge whether the adv_example is true adversarial example
        adv_example_label = keras_model.predict(adv_example)
        adv_example_label = np.argmax(adv_example_label)
        if adv_example_label != predict_label:
            original_image = image.numpy()
            adv_example = adv_example.numpy()
            norm_fn = lambda x: np.max(np.abs(x),axis=(1,2,3))
            norm_diff = norm_fn(adv_example-original_image)
            printlog("PGD(+WiNR) norm_diff(adv_example-original_example) = {}".format(norm_diff))
            PGD_before_WiNR_flag = True
            PGD_before_WiNR_falsified_number += 1
            PGD_before_WiNR_falsified_img_id.append(i)
            printlog("PGD(+WiNR) adv_example_label = {}".format(adv_example_label))
            printlog("PGD(+WiNR) attack succeed!")
        else:
            printlog("PGD(+WiNR) attack failed!")
        PGD_before_WiNR_time += (time.time() - PGD_before_WiNR_start_time)
        
        if PGD_before_WiNR_flag:
            continue
        
        printlog('----------------WiNR(+PGD)----------------')
        WiNR_after_PGD_start_time = time.time()
        adv_false = []
        has_adv_false = False
        WiNR_after_PGD_robust_flag = True
        WiNR_after_PGD_falsified_flag = False
        
        for j in range(i*9,i*9+9):
            target_label = targets[j]
            printlog("target label = {}".format(target_label))
            
            weights = model.weights[:-1]
            biases = model.biases[:-1]
            shapes = model.shapes[:-1]
            W, b, s = model.weights[-1], model.biases[-1], model.shapes[-1]
            last_weight = (W[predict_label,:,:,:]-W[target_label,:,:,:]).reshape([1]+list(W.shape[1:]))
            weights.append(last_weight)
            biases.append(np.asarray([b[predict_label]-b[target_label]]))
            shapes.append((1,1,1))
              
            LB, UB, A_u, A_l, B_u, B_l, pad, stride = find_output_bounds(weights, biases, shapes, model.pads, model.strides, inputs[i].astype(np.float32), eps_0, p_n)   
            
            # solving
            lp_model = new_model()
            lp_model, x = creat_var(lp_model, inputs[i], eps_0)
            shape = inputs[i].shape
            adv_image, min_val = get_solution_value(lp_model, x, shape, A_u, A_l, B_u, B_l, pad, stride, p_n, eps_0)
            printlog("WiNR(+PGD) min_val={:.5f}".format(min_val))
            
            if min_val > 0:
                continue
            
            WiNR_after_PGD_robust_flag = False
            # label of potential counterexample
            a = adv_image[np.newaxis,:,:,:]
            aa = a.astype(np.float32)
            adv_label = np.argmax(np.squeeze(keras_model.predict(aa)))
            
            if adv_label == predict_label:
                adv_false.append((adv_image, target_label))
                has_adv_false = True
                print('this adv_example is false!')
                continue
            
            WiNR_diff = (adv_image-inputs[i]).reshape(-1)
            WiNR_diff = np.absolute(WiNR_diff)
            WiNR_diff = np.max(WiNR_diff)
            printlog("WiNR(+PGD) diff(adv_example-original_example) = {}".format(WiNR_diff))
            WiNR_after_PGD_falsified_flag = True
            break
        
        if WiNR_after_PGD_robust_flag:
            WiNR_after_PGD_robust_number += 1
            printlog("WiNR(+PGD): robust")
        elif WiNR_after_PGD_falsified_flag:
            WiNR_after_PGD_falsified_number += 1
            WiNR_after_PGD_falsified_img_id.append(i)
            printlog("WiNR(+PGD): falsified")
        else:
            printlog("WiNR(+PGD): unknown")
            WiNR_after_PGD_unknown_number += 1
        
        WiNR_after_PGD_time += (time.time()-WiNR_after_PGD_start_time)
        printlog("PGD(+WiNR) - falsified: {}".format(PGD_before_WiNR_flag))
        printlog("WiNR(+PGD) - robust: {}, falsified: {}, unknown: {}".format(WiNR_after_PGD_robust_flag, WiNR_after_PGD_falsified_flag, (not(WiNR_after_PGD_robust_flag) and not(WiNR_after_PGD_falsified_flag))))
        
        if (PGD_before_WiNR_time + WiNR_after_PGD_time) >= time_limit:
            printlog("PGD + WiNR total time : {} sec, reach time limit!".format(PGD_before_WiNR_time + WiNR_after_PGD_time))
            break
    
    PGD_before_WiNR_aver_time = PGD_before_WiNR_time / total_images
    WiNR_after_PGD_aver_time = WiNR_after_PGD_time / total_images
    PGD_WiNR_total_time = PGD_before_WiNR_time + WiNR_after_PGD_time
    PGD_WiNR_total_aver_time = PGD_WiNR_total_time / total_images
    printlog("[L0] method = PGD(+WiNR), average runtime = {:.3f}".format(PGD_before_WiNR_aver_time))
    printlog("[L0] method = WiNR(+PGD), average runtime = {:.3f}".format(WiNR_after_PGD_aver_time))
    printlog("[L0] method = PGD+WiNR, eps = {}, total images = {}, robust = {}, falsified = {}, unknown = {}, average runtime = {:.3f}".format(eps_0, total_images, WiNR_after_PGD_robust_number, (PGD_before_WiNR_falsified_number+WiNR_after_PGD_falsified_number), WiNR_after_PGD_unknown_number, PGD_WiNR_total_aver_time))
    printlog("[L0] PGD(+WiNR) falsified images id: {}".format(len(PGD_before_WiNR_falsified_img_id)))
    printlog("[L0] WiNR(+PGD) falsified images: {}".format(len(WiNR_after_PGD_falsified_img_id)))
    printlog("[L0] WiNR falsified images: {}".format(len(WiNR_falsified_img_id)))
    printlog("[L0] PGD(+WiNR) falsified images id: {}".format(PGD_before_WiNR_falsified_img_id))
    printlog("[L0] WiNR(+PGD) falsified images id: {}".format(WiNR_after_PGD_falsified_img_id))
    printlog("[L0] WiNR falsified images id: {}".format(WiNR_falsified_img_id))
       
    print('------------------')
    print('------------------')
    #return eps_0, len(inputs), DeepCert_robust_number, PGD_falsified_number, PGD_DeepCert_unknown_number, PGD_aver_time, DeepCert_aver_time, PGD_DeepCert_aver_time, WiNR_robust_number, WiNR_falsified_number, WiNR_unknown_number, WiNR_aver_time, WiNR_after_PGD_robust_number, (PGD_before_WiNR_falsified_number+WiNR_after_PGD_falsified_number), WiNR_after_PGD_unknown_number, PGD_before_WiNR_aver_time, WiNR_after_PGD_aver_time, PGD_WiNR_total_aver_time

    return eps_0, len(inputs), WiNR_robust_number, WiNR_falsified_number, WiNR_unknown_number, WiNR_aver_time, WiNR_after_PGD_robust_number, (PGD_before_WiNR_falsified_number+WiNR_after_PGD_falsified_number), WiNR_after_PGD_unknown_number, PGD_before_WiNR_aver_time, WiNR_after_PGD_aver_time, PGD_WiNR_total_aver_time


    # for i in range(len(inputs)):
    #     print('image: ', i, file=f)
    #     print('image: ', i)
    #     predict_label = np.argmax(true_labels[i])
    #     print('predict_label:', predict_label)
    #     print('predict_label:', predict_label, file=f)
    #     adv_false = []
    #     has_adv_false = False
    #     flag = True
    #     for j in range(i*9,i*9+9):
    #         target_label = targets[j]
    #         print('target_label:', target_label)
    #         print('target_label:', target_label, file=f)
    #         weights = model.weights[:-1]
    #         biases = model.biases[:-1]
    #         shapes = model.shapes[:-1]
    #         W, b, s = model.weights[-1], model.biases[-1], model.shapes[-1]
    #         last_weight = (W[predict_label,:,:,:]-W[target_label,:,:,:]).reshape([1]+list(W.shape[1:]))
    #         weights.append(last_weight)
    #         biases.append(np.asarray([b[predict_label]-b[target_label]]))
    #         shapes.append((1,1,1))

    #         LB, UB, A_u, A_l, B_u, B_l, pad, stride = find_output_bounds(weights, biases, shapes, model.pads, model.strides, inputs[i].astype(np.float32), eps_0, p_n)
            
    #         # solving
    #         lp_model = new_model()
    #         lp_model, x = creat_var(lp_model, inputs[i], eps_0)
    #         shape = inputs[i].shape
    #         adv_image, min_val = get_solution_value(lp_model, x, shape, A_u, A_l, B_u, B_l, pad, stride, p_n, eps_0)
            
    #         if min_val > 0:
    #             continue
            
    #         # label of potential counterexample
    #         a = adv_image[np.newaxis,:,:,:]
    #         print(a.dtype)
    #         aa = a.astype(np.float32)
    #         adv_label = np.argmax(np.squeeze(keras_model.predict(aa)))
    #         print('adv_label: ', adv_label)
    #         print('adv_label: ', adv_label, file=f)
            
    #         if adv_label == predict_label:
    #             adv_false.append((adv_image, target_label))
    #             has_adv_false = True
    #             print('this adv_example is false!', file=f)
    #             continue
    #         flag = False
            
    #         fashion_mnist_labels_names = ['T-shirt or top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            
    #         cifar10_labels_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            
    #         # save adv images
    #         print('adv_image.shape:', adv_image.shape)
    #         print(adv_image)
    #         print(adv_image, file=f)
    #         save_adv_image = np.clip(adv_image * 255, 0, 255)
    #         if cifar:
    #             save_adv_image = save_adv_image.astype(np.int32)
    #             plt.imshow(save_adv_image)
    #             adv_label_str = cifar10_labels_names[adv_label]
    #         elif gtsrb:
    #             save_adv_image = save_adv_image.astype(np.int32)
    #             plt.imshow(save_adv_image)
    #             adv_label_str = str(adv_label)
    #         elif fashion_mnist:
    #             plt.imshow(save_adv_image, cmap='gray')
    #             adv_label_str = fashion_mnist_labels_names[adv_label]
    #         else:
    #             plt.imshow(save_adv_image, cmap='gray')
    #             adv_label_str = str(adv_label)
    #         print(save_adv_image)
    #         print(save_adv_image, file=f)
    #         print('adv_label_str.shape:', type(adv_label_str))
    #         save_path = 'adv_examples/'+ dataset + '_'+str(eps_0)+'_adv_image_'+str(i)+'_adv_label_'+adv_label_str +'.png'
    #         plt.savefig(save_path)
            
    #         print(inputs[i].astype(np.float32))
    #         original_image = np.clip(inputs[i].astype(np.float32)*255,0,255)
    #         if cifar:
    #             original_image = original_image.astype(np.int32)
    #             plt.imshow(original_image)
    #             predict_label_str = cifar10_labels_names[predict_label]
    #         elif gtsrb:
    #             original_image = original_image.astype(np.int32)
    #             plt.imshow(original_image)
    #             predict_label_str = str(predict_label)
    #         elif fashion_mnist:
    #             plt.imshow(original_image, cmap='gray')
    #             predict_label_str = fashion_mnist_labels_names[predict_label]
    #         else:
    #             plt.imshow(original_image, cmap='gray')
    #             predict_label_str = str(predict_label)
    #         print(original_image, file=f)
    #         save_path = 'adv_examples/'+ dataset +'_'+str(eps_0)+'_original_image_'+str(i)+'_predict_label_'+predict_label_str+'.png'
    #         plt.savefig(save_path)
            
    #         break
            
    #     if not flag:
    #         unrobust_number += 1
    #         print('this figure is not robust in eps_0', file=f)
    #         print("[L1] method = WiNR-{}, model = {}, image no = {}, true_label = {}, target_label = {}, adv_label = {}, robustness = {:.5f}".format(activation, file_name, i+1, predict_label, target_label, adv_label,eps_0), file=f)
    #     else:
    #         if has_adv_false:
    #             has_adv_false_number += 1
    #         else:
    #             robust_number += 1
    #             print("figure {} is robust in {}.".format(i, eps_0), file=f)
                
    #     print('---------------------------------', file=f)
    #     print("robust: {}, unrobust: {}, has_adv_false: {}".format((flag and (not has_adv_false)), (not flag), has_adv_false), file=f)
    #     time_sum = time.time() - start_time
    #     if time_sum >= limit_time:
    #         print('time_sum:',time_sum, file=f)
    #         break

    # first_sort_time = (time.time()-start_time)
    # print("[L0] method = WiNR-{}, model = {}, eps = {}, total images = {}, robust = {}, unrobust = {}, has_adv_false = {}, total runtime = {:.2f}".format(activation,file_name,eps_0, len(inputs), robust_number, unrobust_number, has_adv_false_number, first_sort_time), file=f)
    # results.append((eps_0, robust_number, unrobust_number, has_adv_false_number, first_sort_time))
    
    # print('eps_0  robust_number  unrobust_number  has_adv_false_number total_runtime', file=f)
    # for i in range(len(results)):
    #     print(results[i][0], '\t', results[i][1], '\t\t', results[i][2], '\t\t', results[i][3], '\t\t', results[i][4], file=f)
    
    # f.close()     
    
    # print('------------------')
    # print('------------------')
    # return results
