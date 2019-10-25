
import numpy.linalg as la
import numpy as np
import matplotlib.pyplot as plt



##############################
## Case Study (section 9.3.2)
##############################
## kernel density estimator
def ourkde(x,h,u):
    # u is a output vector
    fu = []
    for u0 in u:
        t = (x - u0)/h
        K = h**(-1)*np.exp(-0.5*t**2)/np.sqrt(2*np.pi)
        fu.append(np.mean(K))
    fu = np.asarray(fu)
    return fu

from scipy import interpolate
hx = fu[:,2]
M = max(hx)
sample = []

iter = 0
N = 1000
interp = interpolate.interp1d(u, hx, kind='linear')
while len(sample)<N:
    iter = iter + 1
    X = (36-23)*np.random.random() + 23
    U = np.random.random()
    h_X = interp(X)
    if U < h_X/M:
        sample.append(X)

sample = np.asarray(sample)

u2 = np.linspace(23,38,100)
h = 1.06*np.std(sample)*N**(-0.2)
fx = ourkde(sample,h,u2)

plt.figure(figsize=(10,8))
plt.plot(u2,fx,'-')
plt.legend(['spd estimate'])
plt.savefig(r'D:\hot\book\chapter9\fig\c9-option2')



##############################
## Example 9.2 (section 9.3.3)
##############################

## MCMC MH algorithm
# generate random sample from  f ~ exp(-|x|)
chain = [0]
B = 1000000
for j in range(B):
    v0 = chain[j]
    e = np.random.randn()*0.5
    v1 = v0 + e
    ratio = np.exp(-np.abs(v1))/np.exp(-np.abs(v0))
    if np.random.rand()<ratio:
        chain.append(v1)
    else:
        chain.append(v0)

plt.hist(chain,60)
plt.savefig(r'D:\hot\book\chapter9\fig\mh92')




##############################
## Example 9.3 (section 9.3.3)
##############################

## MCMC Toy 2 / Gibbs sampling
# generate data
n = 100
x = np.random.randn(n,1)
err = np.random.randn(n,1)*0.7
y = -1 + 2*x  + err
#plt.plot(x,y,'.')

class mcmctoy:
    mcmc_len = []
    N = []
    alpha = []
    beta = []
    tau = []
    tau_alpha = []
    tau_beta = []
    a = []; b = []
    y = []; x = []

    def get_initial(self):
        self.alpha.append(np.random.randn()*3**0.5)
        self.beta.append(np.random.randn()*3**0.5)
        self.tau.append(np.random.gamma(0.1,10,))
        self.N = len(self.y)

    def update_alpha(self):
            inv_var = self.tau_alpha + self.tau[-1] * self.N
            mu = (self.tau[-1] * np.sum(self.y - self.beta[-1] * self.x))/inv_var
            self.alpha.append(np.random.normal(mu, 1 / np.sqrt(inv_var)))

    def update_beta(self):
            inv_var = self.tau_beta + self.tau[-1] * np.sum(self.x * self.x)
            mu = (self.tau[-1] * np.sum((self.y - self.alpha[-1]) * x))/inv_var
            self.beta.append(np.random.normal(mu, 1 / np.sqrt(inv_var)))
            
    def update_tau(self):
            a2 = self.a + self.N / 2
            res = self.y - self.alpha[-1] - self.beta[-1] * self.x
            b2 = self.b + np.sum(res * res) / 2
            tau_rnd = np.random.gamma(a2, 1 / b2)
            self.tau.append(tau_rnd)

    def mcmc_main(self):
        self.get_initial()
        for i in range(self.mcmc_len):
            self.update_alpha()
            self.update_beta()
            self.update_tau()

    def mcmc_plot(self):
        crk = 100
        plt.plot(self.alpha[-m1.mcmc_len+crk:])
        plt.plot(self.beta[-m1.mcmc_len+crk:])
        plt.plot(self.tau[-m1.mcmc_len+crk:])
        


m1 = mcmctoy()
m1.mcmc_len = 5000 
m1.x = x
m1.y = y
m1.a = 0.1
m1.b = 10
m1.tau_alpha = 1
m1.tau_beta = 1
m1.mcmc_main()
m1.mcmc_plot()

plt.plot(m1.alpha[-400:])
plt.plot(m1.beta[-400:])
plt.plot(m1.tau[-400:])
plt.legend(['alpha','beta','tau'],loc='center right')
plt.savefig(r'D:\hot\book\chapter9\fig\gibbs')





##############################
## Example 9.4 (section 9.3.4)
##############################
## Importance sampling
# prob(Z>4.5)

import scipy.stats as st
import numpy as np

N = 100000
X0 = np.random.randn(N,)
mu0 = np.sum(X0>4.5)/N
M = 20
X = np.random.rand(N,)*M + 4.5
mu = np.sum(st.norm.pdf(X,0,1))*M/N
print(mu)




##############################
## Example 9.5 (section 9.4.2)
##############################
## variational inference
## 参考TAPAS matlab 程序
import numpy as np
import numpy.linalg as la
import scipy.special as sp

def vblm(y,X,a_0,b_0,c_0,d_0):
    n,p = X.shape
    mu_n     = np.zeros((p,1))
    Lambda_n = np.eye(p)
    a_n      = a_0
    b_n      = b_0
    c_n      = c_0
    d_n      = d_0
    F        = -np.inf
    
    iter   = 0
    diff   = 100
    tol    = 0.0001
    max_it = 30
    while diff>tol and iter<max_it:       
        Lambda_n = a_n/b_n*np.eye(p) + c_n/d_n*X.T.dot(X)
        mu_n = c_n/d_n*la.inv(Lambda_n).dot(X.T).dot(y)        
        a_n = a_0 + p/2
        b_n = b_0 + 1/2 * (mu_n.T.dot(mu_n) + np.trace(la.inv(Lambda_n)))        
        c_n = c_0 + n/2
        d_n = d_0 + 1/2*(y.T.dot(y)-2*mu_n.T.dot(X.T).dot(y)\
              +np.trace(X.T.dot(X).dot(mu_n.dot(mu_n.T)+la.inv(Lambda_n))))
    
        F_old = F
        J = n/2*(sp.digamma(c_n)-np.log(d_n)) - n/2*np.log(2*np.pi) \
          - 1/2*c_n/d_n*y.T.dot(y)+c_n/d_n*mu_n.T.dot(X.T).dot(y) \
          - 1/2*c_n/d_n*np.trace(X.T.dot(X).dot(mu_n.dot(mu_n.T)+la.inv(Lambda_n))) \
          - p/2*np.log(2*np.pi)+n/2*(sp.digamma(a_n)-np.log(b_n)) \
          - 1/2*a_n/b_n*(mu_n.T.dot(mu_n)+np.trace(la.inv(Lambda_n))) \
          + a_0*np.log(b_0)-sp.gammaln(a_0)+(a_0-1)*(sp.digamma(a_n)-np.log(b_n))-b_0*a_n/b_n \
          + c_0*np.log(d_0)-sp.gammaln(c_0)+(c_0-1)*(sp.digamma(c_n)-np.log(d_n))-d_0*c_n/d_n        
        H = p/2*(1+np.log(2*np.pi))+1/2*np.log(la.det(la.inv(Lambda_n))) \
          + a_n-np.log(b_n)+sp.gammaln(a_n)+(1-a_n)*sp.digamma(a_n) \
          + c_n-np.log(d_n)+sp.gammaln(c_n)+(1-c_n)*sp.digamma(c_n)   
        F = J + H
        diff = F - F_old
    return mu_n,Lambda_n,a_n,b_n,c_n,d_n

X = np.random.rand(400,3)
beta = np.array([[1],[2],[3]])
y = X.dot(beta) + 0.3*np.random.randn(400,1)
mu_n,Lambda_n,a_n,b_n,c_n,d_n = vblm(y,X,10,1,10,1)