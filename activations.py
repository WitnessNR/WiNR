
from numba import njit
import numpy as np

@njit
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

@njit
def sigmoidd(x):
    return np.exp(-x)/(1.0+np.exp(-x))**2

@njit
def sigmoidid(x):
    return 2.0*np.arccosh(1.0/(2.0*np.sqrt(x)))

@njit 
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
    
@njit 
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

# added new functions
@njit 
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

@njit 
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

@njit 
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

@njit 
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

@njit
def sigmoid_linear_bounds(LB, UB):
    alpha_u = np.zeros(UB.shape, dtype=np.float32)
    beta_u = np.zeros(UB.shape, dtype=np.float32)
    alpha_l = np.zeros(LB.shape, dtype=np.float32)
    beta_l = np.zeros(LB.shape, dtype=np.float32)
    for i in range(LB.shape[0]):
        for j in range(LB.shape[1]):
            for k in range(LB.shape[2]):
                act = sigmoid
                actd = sigmoidd
                actut = sigmoidut
                actlt = sigmoidlt
                if UB[i,j,k] == LB[i,j,k]:
                    alpha_u[i,j,k] = actd(UB[i,j,k])
                    alpha_l[i,j,k] = actd(LB[i,j,k])
                    beta_u[i,j,k] = act(UB[i,j,k])-actd(UB[i,j,k])*UB[i,j,k]
                    beta_l[i,j,k] = act(LB[i,j,k])-actd(LB[i,j,k])*LB[i,j,k]
                
                elif LB[i,j,k] >= 0:
                    alpha = (act(UB[i,j,k])-act(LB[i,j,k]))/(UB[i,j,k]-LB[i,j,k])
                    d = sigmoidup(LB[i,j,k], UB[i,j,k], alpha)
                    alpha_u[i,j,k] = actd(d)
                    alpha_l[i,j,k] = alpha
                    beta_u[i,j,k] = act(d)-actd(d)*d
                    beta_l[i,j,k] = act(LB[i,j,k])-alpha*LB[i,j,k]
                elif UB[i,j,k] <= 0:
                    alpha = (act(UB[i,j,k])-act(LB[i,j,k]))/(UB[i,j,k]-LB[i,j,k])
                    d = sigmoidlow(LB[i,j,k], UB[i,j,k], alpha)
                    alpha_u[i,j,k] = alpha
                    alpha_l[i,j,k] = actd(d)
                    beta_u[i,j,k] = act(LB[i,j,k])-alpha*LB[i,j,k]
                    beta_l[i,j,k] = act(d)-actd(d)*d
                else:
                    alpha = (act(UB[i,j,k])-act(LB[i,j,k]))/(UB[i,j,k]-LB[i,j,k])
                    dU = actd(UB[i,j,k])
                    dL = actd(LB[i,j,k])
                    if act(UB[i,j,k])-dU*(UB[i,j,k]-LB[i,j,k]) < act(LB[i,j,k]) and act(LB[i,j,k])+dL*(UB[i,j,k]-LB[i,j,k]) < act(UB[i,j,k]):
                        alpha_u[i,j,k] = alpha
                        beta_u[i,j,k] = act(LB[i,j,k])-alpha*LB[i,j,k]
                        d = sigmoidloww(LB[i,j,k], UB[i,j,k], alpha)
                        alpha_l[i,j,k] = actd(d)
                        beta_l[i,j,k] = act(d)-actd(d)*d
                    elif act(UB[i,j,k])-dU*(UB[i,j,k]-LB[i,j,k]) > act(LB[i,j,k]) and act(LB[i,j,k])+dL*(UB[i,j,k]-LB[i,j,k]) > act(UB[i,j,k]):
                        alpha_l[i,j,k] = alpha
                        beta_l[i,j,k] = act(LB[i,j,k])-alpha*LB[i,j,k]
                        d = sigmoidupp(LB[i,j,k], UB[i,j,k], alpha)
                        alpha_u[i,j,k] = actd(d)
                        beta_u[i,j,k] = act(d)-actd(d)*d
                    else:
                        du = actut(LB[i,j,k], UB[i,j,k])
                        dus = (act(du)-act(LB[i,j,k]))/(du-LB[i,j,k])
                        dut = actd(du)
                        if dut < dus:
                            alpha_u[i,j,k] = dut
                            beta_u[i,j,k] = act(du)-dut*du
                        else:
                            alpha_u[i,j,k] = dus
                            beta_u[i,j,k] = act(LB[i,j,k])-LB[i,j,k]*dus
                        dl = actlt(LB[i,j,k], UB[i,j,k])
                        dls = (act(dl)-act(UB[i,j,k]))/(dl-UB[i,j,k])
                        dlt = actd(dl)
                        if dlt < dls:
                            alpha_l[i,j,k] = dlt
                            beta_l[i,j,k] = act(dl)-dlt*dl
                        else:
                            alpha_l[i,j,k] = dls
                            beta_l[i,j,k] = act(UB[i,j,k])-UB[i,j,k]*dls
    return alpha_u, alpha_l, beta_u, beta_l
