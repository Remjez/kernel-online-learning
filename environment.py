import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from tqdm import tqdm_notebook as tqdm
import copy
from forecaster import *
from time import time

def random_linear_stream(d,n,eps=0):
    u = np.random.rand(d)
    for i in range(n):
        x = np.random.rand(d)
        yield (x, np.dot(u,x)/d + np.random.rand()*eps)
        
def random_cos_stream(d,n,eps=0):
    u = np.random.rand(d)
    A = np.random.rand(d,d)
    for i in range(n):
        x = np.random.rand(d)
        yield (x, np.dot(u,np.cos(np.dot(A,(x-.5)*(np.sign(x-.5)+1))))/d + np.random.rand()*eps)

def run_on_stream(stream, loss, forecaster, timeout=-1):
    L = []
    x, y = stream[0]
    d = x.shape[0]
    X = np.zeros((0,d))
    Y = np.zeros((0,))
    pred = np.zeros((0,))
    s = 0
    start = time()
    T = []
    for t, (x, y) in tqdm(list(enumerate(stream))):
        pred = forecaster.predict(x)
        s += loss(pred,y)
        L.append(s)
        T.append(time()-start)
        forecaster.update(x, y)
        if timeout > 0 and time() - start > timeout:
            break
    for q in range(t+1,len(list(stream))):
        L.append(np.nan)
        T.append(np.nan)
    return (L, T)

def linear_regret(pred, X, Y, lbd=1.):
    H = np.dot(X,np.dot(np.linalg.pinv(np.dot(np.transpose(X),X) + lbd*np.eye(X.shape[1])), np.transpose(X)))
    res = np.linalg.norm(pred - Y)**2 \
           - np.dot(np.transpose(Y), np.dot(np.eye(H.shape[0]) - H, Y))
    return res

def regret_against_oracle(pred, pred_oracle, X, Y):
    return np.linalg.norm(pred - Y)**2 - np.linalg.norm(pred_oracle - Y)**2

def run_with_adversary(d, n, loss, forecaster, oracle_kernel=lambda x0,x1: np.dot(x0,x1), timeout=-1):
    R_hist = []
    X = np.zeros((0,d))
    Y = np.zeros((0,))
    pred_l = np.zeros((0,))
    lbd = 1e-0
    oracle = Kernel_ridge_reg(d, oracle_kernel)
    start = time()
    for t in tqdm(range(n)):
        def R(xy):
           x = xy[:-1]
           y = xy[-1]
           oracle_uptodate = copy.deepcopy(oracle)
           oracle_uptodate.update(x, y)
           Xt = np.concatenate((X,x[None,:]), axis=0)
           Yt = np.concatenate((Y,np.array(y)[None]), axis=0)
           return -regret_against_oracle(np.concatenate((pred_l,np.array(forecaster.predict(x))[None]), axis=0),
                                         np.array([oracle_uptodate.predict(x) for x in Xt]),
                                         Xt, Yt) 
        o = opt.minimize(R, np.random.rand(d+1), bounds=np.ones((d+1,2))*[-1,1])#, options={'maxiter':1})
        R_t = -o['fun']
        xy = o['x']
        #R_max = 0
        #best_xy = None
        #for _ in range(10):
        #    xy = np.random.rand(d+1)
        #    R_t = -R(xy)
        #    if R_t > R_max:
        #        R_max = R_t
        #        best_xy = xy
        #xy = best_xy
        x, y = xy[:-1], xy[-1]
        X = np.concatenate((X,x[None,:]), axis=0)
        Y = np.concatenate((Y,np.array(y)[None]), axis=0)
        pred = forecaster.predict(x)
        pred_l = np.concatenate((pred_l, np.array(pred)[None]), axis=0)
        R_hist.append(R_t)
        forecaster.update(x, y)
        oracle.update(x, y)
        if timeout > 0 and time() - start > timeout:
            break
    for q in range(t+1,n):
        R_hist.append(np.nan)
    return R_hist
