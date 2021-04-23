# -*- coding: utf-8 -*-
import numpy as np
import math

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoidd(x):
    return np.exp(-x)/(1.0+np.exp(-x))**2

def sigmoidid(x):
    return 2.0*np.arccosh(1.0/(2.0*np.sqrt(x)))

def sigmoidut(l, u):
    act = sigmoid
    actd = sigmoidd
    actid = sigmoidid
    upper = u
    lower = 0
    al = act(l)
    for i in range(20):
        guess = (upper + lower)/2
        guesst = actd(guess)
        guesss = (act(guess)-al)/(guess-l)
        if guesss >= guesst:
            upper = guess
        else:
            lower = guess
    return upper

def sigmoidlt(l, u):
    act = sigmoid
    actd = sigmoidd
    actid = sigmoidid
    upper = 0
    lower = l
    au = act(u)
    for i in range(20):
        guess = (upper + lower)/2
        guesst = actd(guess)
        guesss = (au-act(guess))/(u-guess)
        if guesss >= guesst:
            lower = guess
        else:
            upper = guess
    return lower

def sigmoidup(l, u, k):
    act = sigmoid
    actd = sigmoidd
    upper = u
    lower = l
    for i in range(20):
        guess = (upper + lower)/2
        guesst = actd(guess)
        if k > guesst:
            upper = guess
        elif k < guesst:
            lower = guess
        else:
            upper = guess
            break
    return upper

def sigmoidlow(l, u, k):
    act = sigmoid
    actd = sigmoidd
    upper = u
    lower = l
    for i in range(20):
        guess = (upper + lower)/2
        guesst = actd(guess)
        if k > guesst:
            lower = guess
        elif k < guesst:
            upper = guess
        else:
            lower = guess
            break
    return lower
    

def sigmoidupp(l, u, k):
    act = sigmoid
    actd = sigmoidd
    upper = u
    lower = 0
    for i in range(20):
        guess = (upper + lower)/2
        guesst = actd(guess)
        if k > guesst:
            upper = guess
        elif k < guesst:
            lower = guess
        else:
            upper = guess
            break
    return upper

def sigmoidloww(l, u, k):
    act = sigmoid
    actd = sigmoidd
    upper = 0
    lower = l
    for i in range(20):
        guess = (upper + lower)/2
        guesst = actd(guess)
        if k > guesst:
            lower = guess
        elif k < guesst:
            upper = guess
        else:
            lower = guess
            break
    return lower

def swish(x):
    s = x * sigmoid(x)
    return s

def dswish(x):
    s = swish(x) + sigmoid(x) * (1 - swish(x))
    return s

def p(mid, x):
    s = swish(mid) + dswish(mid) * (x - mid)
    return s

def r(mid, x):
    s = swish(x) - p(mid, x)
    return s

def swishut(lb, ub):
    lower = 2.399357
    upper = ub
    al = sigmoid(lb)
    while lower < upper:
        mid = (upper + lower)/2     
        guesst = sigmoidd(mid)
        guesss = (sigmoid(mid)-al)/(mid-lb)
        if math.isclose(mid,lower,rel_tol=1e-6):
            return mid
        if guesss >= guesst:
            upper = mid
        else:
            lower = mid
    return upper

def find_tangent_point(lb, ub, k):
    lower = lb
    upper = ub
    while lower < upper:
        mid = (upper + lower) / 2
        tmpk = dswish(mid)
        if math.isclose(tmpk,k,rel_tol=1e-6):
            return mid
        if tmpk > k:
            upper = mid
        else:
            lower = mid
    return -1

def distance(A, B, C, x0, y0):
    s = abs((A*x0 + B*y0 + C) / math.sqrt(A**2 + B**2))
    return s
