import numpy as np
from math import factorial, sqrt
import operator
from functools import reduce
from collections import Counter
import scipy.linalg.blas as sclb
from scipy.linalg import solve_triangular
from time import time

def npermutations(l):
    num = factorial(len(l))
    mults = Counter(l).values()
    den = reduce(operator.mul, (factorial(v) for v in mults), 1)
    return num / den

def list_oc(l):
    mults = Counter(l).values()
    return mults

def compute_taylor_features(x, r, sig):
    res = np.array([1.])
    monome = np.array([1.])
    for k in range(1,r+1):
        monome = (np.outer(monome, x)).flatten()
        res = np.concatenate((res,monome/(sig**k*sqrt(factorial(k)))))
    return np.array(res)*np.exp(-np.linalg.norm(x)**2/(2*sig**2))

def update_inv(A_inv,x):
    B = x[:-1][:,None]
    C = np.transpose(B)
    D = np.array(x[-1])[None,None]
    if A_inv.size == 0:
        return 1./D
    compl = 1./(D - np.dot(np.dot(C,A_inv),B))
    R0 = A_inv + np.dot(np.dot(np.dot(np.dot(A_inv,B),compl),C),A_inv)
    R1 = - np.dot(np.dot(A_inv,B),compl)
    R2 = - np.dot(np.dot(compl,C),A_inv)
    R3 = np.array(compl)
    return np.block([[R0, R1], [R2, R3]])

class Ridge_reg:
    def __init__(self, d, lbd=1., phi=lambda x:x):
        self.w = np.zeros(d)
        self.A_inv = 1/lbd*np.eye(d)
        self.phi = phi

    def predict(self, x):
        x = self.phi(x)
        return np.dot(self.w,x)

    def update(self, x, y):
        x = self.phi(x)
        A_inv_dot_x = sclb.dsymv(1, self.A_inv.T, x)
        #A_inv_dot_x = np.dot(self.A_inv,x)
        sclb.dsyr(-1/(1 + np.dot(x,A_inv_dot_x)), A_inv_dot_x,  a=self.A_inv.T, overwrite_a=True)
        #self.A_inv -= np.outer(A_inv_dot_x, A_inv_dot_x) 
        sclb.dsymv(-(np.dot(self.w,x) - y), self.A_inv.T, x, beta=1, y=self.w, overwrite_y=True)
        #self.w -= np.dot(self.A_inv,(np.dot(self.w,x) - y)*x)

class Vovk_reg:
    def __init__(self, d, lbd=1., phi=lambda x:x):
        self.b = np.zeros(d)
        self.A_inv = 1/lbd*np.eye(d)
        self.phi = phi

    def predict(self, x):
        x = self.phi(x)
        A_inv_dot_x = np.dot(self.A_inv,x)
        A_inv_t = self.A_inv - np.outer(A_inv_dot_x, A_inv_dot_x) / (1 + np.dot(x,A_inv_dot_x))
        w_hat_t = np.dot(A_inv_t,self.b)
        return np.dot(w_hat_t, x)

    def update(self, x, y):
        x = self.phi(x)
        A_inv_dot_x = np.dot(self.A_inv,x)
        self.A_inv -= np.outer(A_inv_dot_x, A_inv_dot_x) / (1 + np.dot(x,A_inv_dot_x))
        self.b += y*x
        
class Kernel_ridge_reg:
    def __init__(self, d, k, lbd=1.):
        self.c = np.zeros(0)
        self.X = np.zeros((0,d))
        self.Y = np.zeros(0)
        self.k = k
        self.lbd = lbd
        self.K_inv = np.eye(0)

    def predict(self, x):
        n = self.X.shape[0]
        Kn = np.array([self.k(self.X[i,:],x) for i in range(n)])
        return np.dot(Kn, self.c)

    def update(self, x, y):
        self.X = np.concatenate((self.X,x[None,:]), axis=0)
        self.Y = np.concatenate((self.Y,np.array(y)[None]), axis=0)
        n = self.X.shape[0]
        en = np.zeros(n)
        en[-1] = 1
        Kn = np.array([self.k(self.X[i,:], x) for i in range(n)]) + self.lbd*en
        self.K_inv = update_inv(self.K_inv, Kn)
        self.c = np.dot(self.K_inv,self.Y)

class Kernel_vovk_reg:
    def __init__(self, d, k, lbd=1.):
        self.X = np.zeros((0,d))
        self.Y = np.zeros(0)
        self.k = k
        self.lbd = lbd
        self.K_inv = np.eye(0)

    def predict(self, x):
        X = np.concatenate((self.X,x[None,:]), axis=0)
        Y = np.concatenate((self.Y,np.array([0])), axis=0)
        n = X.shape[0]
        en = np.zeros(n)
        en[-1] = 1
        Kn = np.array([self.k(X[i,:], x) for i in range(n)])
        K_inv = update_inv(self.K_inv,Kn + self.lbd*en)
        c = np.dot(K_inv,Y)
        return np.dot(Kn, c)

    def update(self, x, y):
        self.X = np.concatenate((self.X,x[None,:]), axis=0)
        self.Y = np.concatenate((self.Y,np.array(y)[None]), axis=0)
        n = self.X.shape[0]
        en = np.zeros(n)
        en[-1] = 1
        Kn = np.array([self.k(self.X[i,:], x) for i in range(n)])
        self.K_inv = update_inv(self.K_inv,Kn + self.lbd*en)

def KORS(x, KMM, S, SKS_inv, lbd=1., eps=0.5, beta=1e0):
    kS = KMM[:,-1]*S
    SKS = np.array(S)[None,:] * KMM * np.array(S)[:,None]
    en = np.eye(len(kS))[:,-1]
    SKS_inv_tmp = update_inv(SKS_inv, kS + lbd*en)
    tau = (1+eps)/lbd*(KMM[-1,-1] - np.dot(kS, np.dot(SKS_inv_tmp, kS)))
    p = max(min(beta*tau,1),0)
    z = np.random.binomial(1,p)
    S = S[:-1]
    if z:
        S.append(1/p)
        SKS_inv = update_inv(SKS_inv, 1/p*KMM[:,-1]*S + lbd*en)
    return z, S, SKS_inv

class Fourier_online_GD_reg:
    def __init__(self, d, D, sig, eta):
        self.d = d
        self.D = D
        self.eta = eta
        self.sig = sig
        self.w = np.zeros(2*D)
        self.u = np.random.randn(D,d)/self.sig

    def predict(self, x):
        z = np.concatenate((np.sin(np.dot(self.u,x)),np.cos(np.dot(self.u,x))))
        return np.dot(self.w, z)/self.D

    def update(self, x, y):
        z = np.concatenate((np.sin(np.dot(self.u,x)),np.cos(np.dot(self.u,x))))
        pred = np.dot(self.w, z)/self.D
        self.w -= self.eta*2*z*(pred - y)

class Pros_n_kons:
    def __init__(self, d, C, k, alpha=1., beta=1.):
        np.random.seed(0)
        self.C = C
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.j = 0
        self.w = np.zeros(0)
        self.g = np.zeros(0)
        self.A_inv = 1./alpha*np.eye(0)
        self.K_j = np.eye(0)
        self.U_j = np.eye(0)
        self.Sig_inv_j = np.eye(0)
        self.S = np.zeros(0)
        self.X = np.zeros((0,d))
        self.SKS_inv = np.eye(0)

    def predict(self, x):
        j = self.X.shape[0]
        Kt = np.array([self.k(self.X[i,:],x) for i in range(j)])
        phi_tilde = np.dot(np.dot(self.Sig_inv_j, np.transpose(self.U_j)), Kt)
        pred = np.clip(np.dot(self.w, phi_tilde),-self.C,self.C)
        return pred

    def update(self, x, y):
        j = self.X.shape[0]
        Kt = np.array([self.k(self.X[i,:],x) for i in range(j)])
        K_kors = np.block([[self.K_j, Kt[:,None]], [Kt[None,:], self.k(x, x)]])
        z, self.S, self.SKS_inv = KORS(x, K_kors, list(self.S)+[1], self.SKS_inv, lbd=self.alpha, beta=self.beta)
        if z:
            self.j += 1
            self.X = np.concatenate((self.X,x[None,:]), axis=0)
            self.K_j = K_kors
            self.U_j, s, _ = np.linalg.svd(K_kors)
            self.Sig_inv_j = np.diag(1/np.sqrt(s))
            self.A_inv = 1./self.alpha*np.eye(self.j)
            self.w = np.zeros(self.j)
            phi_tilde = np.zeros(self.j)
        else:
            phi_tilde = np.dot(np.dot(self.Sig_inv_j, np.transpose(self.U_j)), Kt)
            self.v = self.w - np.dot(self.A_inv, self.g)
            d = np.dot(np.dot(phi_tilde, self.A_inv), phi_tilde)
            h = lambda z: np.sign(z)*max(np.abs(z)-self.C, 0)
            self.w = self.v - h(np.dot(phi_tilde, self.v))/d*np.dot(self.A_inv, phi_tilde)
        pred = np.dot(self.w, phi_tilde)
        self.g = 2*(pred - y)*phi_tilde
        u = self.g/4
        A_inv_dot_u = np.dot(self.A_inv,u)
        self.A_inv -= np.outer(A_inv_dot_u, A_inv_dot_u) / (1 + np.dot(u,A_inv_dot_u))

def cholup_python(R,x,sign):
    p = np.size(x)
    x = np.copy(x.T)
    for k in range(p):
        if sign == '+':
            r = np.sqrt(R[k,k]**2 + x[k]**2)
        elif sign == '-':
            r = np.sqrt(R[k,k]**2 - x[k]**2)
        c = r/R[k,k]
        s = x[k]/R[k,k]
        R[k,k] = r
        if sign == '+':
            R[k,k+1:p] = (R[k,k+1:p] + s*x[k+1:p])/c
        elif sign == '-':
            R[k,k+1:p] = (R[k,k+1:p] - s*x[k+1:p])/c
        x[k+1:p]= c*x[k+1:p] - s*R[k, k+1:p]
    return R

from choldate import cholupdate, choldowndate

def cholup(R,x,sgn):
    u = x.copy()
    if sgn == '+':
        cholupdate(R,u)
    elif sgn == '-':
        choldowndate(R,u)
    return R
            
class Nystrom_chol_kernel_ridge_reg:
    def __init__(self, d, k, lbd=1., beta=1.):
        self.c = np.zeros(0)
        self.X = np.zeros((0,d))
        self.Y = np.zeros(0)
        self.k = k
        self.lbd = lbd
        self.beta = beta
        self.R = np.eye(0)
        self.chosen_idx = []
        self.KnM = np.eye(0)
        self.S = []
        self.SKS_inv = np.eye(0)

    def predict(self, x):
        Kn = np.array([self.k(self.X[i,:],x) for i in self.chosen_idx])
        return np.dot(Kn, self.c)

    def update(self, x, y):
        self.X = np.concatenate((self.X,x[None,:]), axis=0)
        self.Y = np.concatenate((self.Y,np.array(y)[None]), axis=0)
        n = self.X.shape[0]
        Kn = np.array([self.k(self.X[i,:],x) for i in self.chosen_idx])
        self.KnM = np.concatenate((self.KnM,Kn[None,:]), axis=0)
        K_kors = np.concatenate((self.KnM[self.chosen_idx+[n-1],:], 
                                 np.concatenate((Kn, [self.k(x, x)]))[:,None]), axis=1)
        z, self.S, self.SKS_inv = KORS(x, K_kors, list(self.S)+[1], \
                                       self.SKS_inv, lbd=self.lbd, beta = self.beta)
        self.R = cholup(self.R, self.KnM[-1,:], '+')
        if z:
            self.chosen_idx.append(n-1)
            M = len(self.chosen_idx)
            KM = np.array([self.k(self.X[i,:],x) for i in range(n)])
            self.KnM = np.concatenate((self.KnM,KM[:,None]), axis=1)
            a = self.KnM[:,-1].T
            d = np.dot(a, a) + self.lbd*self.KnM[-1,-1]
            if M == 1:
                self.R = np.array([[np.sqrt(d)]])
            else:
                b = self.KnM[self.chosen_idx[:-1],-1]
                c = np.dot(self.KnM[:,:-1].T, a) + self.lbd*b
                g = np.sqrt(1 + d)
                u = np.concatenate((c/(1+g), [g]))
                v = np.concatenate((c/(1+g), [-1]))
                self.R = np.block([[self.R, np.zeros((M-1,1))], [np.zeros((1,M-1)), 0]])
                self.R = cholup(self.R, u, '+')
                self.R = cholup(self.R, v, '-')
        if len(self.R) > 0:
            self.c = solve_triangular(self.R, solve_triangular(self.R.T, np.dot(self.KnM.T,self.Y), lower=True, check_finite=False))

class Nystrom_chol_kernel_vovk_reg:
    def __init__(self, d, k, lbd=1., beta=1.):
        self.c = np.zeros(0)
        self.X = np.zeros((0,d))
        self.Y = np.zeros(0)
        self.k = k
        self.lbd = lbd
        self.beta = beta
        self.R = np.eye(0)
        self.chosen_idx = []
        self.KnM = np.eye(0)
        self.S = []
        self.SKS_inv = np.eye(0)

    def predict(self, x):
        self.X = np.concatenate((self.X,x[None,:]), axis=0)
        n = self.X.shape[0]
        Kn = np.array([self.k(self.X[i,:],x) for i in self.chosen_idx])
        self.KnM = np.concatenate((self.KnM,Kn[None,:]), axis=0)
        K_kors = np.concatenate((self.KnM[self.chosen_idx+[n-1],:], 
                                 np.concatenate((Kn, [self.k(x, x)]))[:,None]), axis=1)
        z, self.S, self.SKS_inv = KORS(x, K_kors, list(self.S)+[1], \
                                       self.SKS_inv, lbd=self.lbd, beta=self.beta)
        self.R = cholup(self.R, self.KnM[-1,:], '+')
        if z:
            self.chosen_idx.append(n-1)
            M = len(self.chosen_idx)
            KM = np.array([self.k(self.X[i,:],x) for i in range(n)])
            self.KnM = np.concatenate((self.KnM,KM[:,None]), axis=1)
            a = self.KnM[:,-1].T
            d = np.dot(a, a) + self.lbd*self.KnM[-1,-1]
            if M == 1:
                self.R = np.array([[np.sqrt(d)]])
            else:
                b = self.KnM[self.chosen_idx[:-1],-1]
                c = np.dot(self.KnM[:,:-1].T, a) + self.lbd*b
                g = np.sqrt(1 + d)
                u = np.concatenate((c/(1+g), [g]))
                v = np.concatenate((c/(1+g), [-1]))
                self.R = np.block([[self.R, np.zeros((M-1,1))], [np.zeros((1,M-1)), 0]])
                self.R = cholup(self.R, u, '+')
                self.R = cholup(self.R, v, '-')
        Yp = np.concatenate((self.Y, [0]))
        if len(self.R) > 0:
            self.c = solve_triangular(self.R, solve_triangular(self.R.T, np.dot(self.KnM.T,Yp), lower=True))
        Kn = np.array([self.k(self.X[i,:],x) for i in self.chosen_idx])
        return np.dot(Kn, self.c)

    def update(self, x, y):
        self.Y = np.concatenate((self.Y,np.array(y)[None]), axis=0)

