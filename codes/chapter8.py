
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


##############################
## Example 8.1 (section 8.3)
##############################
## bootstrap simulation  

## 1, nonparametric bootstrap
n = 100
x = np.random.rand(n,1)
err = np.random.standard_t(3,(n,1))*0.4
y = np.sin(2*np.pi*x) + err


B = 1000
xi1 = 0.33
xi2 = 0.67
c = np.ones((n,1))
u = np.linspace(0,1,100).reshape(100,1)
cu = np.ones((100,1))
xiu1 = (u - xi1)**3*(u - xi1>0)
xiu2 = (u - xi2)**3*(u - xi2>0)
U = np.hstack((cu,u,u**2,u**3,xiu1,xiu2))

estf = np.zeros((100,1000))
for j in range(B):
    #id = np.random.random_integers(0,99,(100,))
    id = np.random.randint(0,100,(100,))
    nx = x[id]
    ny = y[id]
    k1 = ((nx - xi1)*(nx - xi1>0))**3
    k2 = ((nx - xi2)*(nx - xi2>0))**3
    X = np.hstack((c,nx,nx**2,nx**3,k1,k2))
    beta = la.inv(X.T.dot(X)).dot(X.T).dot(ny)
    est = U.dot(beta)
    estf[:,j] = est.ravel()

qt975 = np.percentile(estf,97.5,axis = 1)
qt025 = np.percentile(estf,2.5,axis = 1)
qt050 = np.percentile(estf,50,axis = 1)


plt.plot(u,qt975,'r--')
plt.plot(u,qt025,'r--')
plt.plot(u,qt050,'k-')
plt.plot(x,y,'bo')
plt.savefig(r'D:\hot\book\chapter8\fig\c8-sim-boot-np')


## 2, parametric bootstrap
n = 100
x = np.random.rand(n,1)
err = np.random.standard_t(3,(n,1))*0.4
y = np.sin(2*np.pi*x) + err

## step 1, estimate model
xi1 = 0.33
xi2 = 0.67
c = np.ones((n,1))
k1 = ((x - xi1)*(x - xi1>0))**3
k2 = ((x - xi2)*(x - xi2>0))**3
X = np.hstack((c,x,x**2,x**3,k1,k2))
beta_ols = la.inv(X.T.dot(X)).dot(X.T).dot(y)

yhat = X.dot(beta_ols)
mse = np.mean((y - yhat)**2)


B = 1000
u = np.linspace(0,1,100)
estf = np.zeros((100,1000))
for j in range(B):
    nx = x
    ny = yhat + np.random.randn(100,1)*mse**(0.5)
    k1 = ((nx - xi1)*(nx - xi1>0))**3
    k2 = ((nx - xi2)*(nx - xi2>0))**3
    X = np.hstack((c,nx,nx**2,nx**3,k1,k2))
    beta = la.inv(X.T.dot(X)).dot(X.T).dot(ny)
    est = U.dot(beta)
    estf[:,j] = est.ravel()

qt975 = np.percentile(estf,97.5,axis = 1)
qt025 = np.percentile(estf,2.5,axis = 1)
qt050 = np.percentile(estf,50,axis = 1)

plt.plot(u,qt975,'r--')
plt.plot(u,qt025,'r--')
plt.plot(u,qt050,'k-')
plt.plot(x,y,'bo')
plt.savefig(r'D:\hot\book\chapter8\fig\c8-sim-boot-par')






##############################
## Example 8.2 (section 8.5.2)
##############################
## mixture model/ EM algorithm



#import numpy as np
import scipy.stats as st
n = 100
x1 = np.random.randn(n,1)*0.5 + 0.7
x2 = np.random.randn(n,1)*0.8 - 1.2
x = np.vstack((x1,x2)).ravel()
plt.hist(x,20)


def em_mixture1(x,K,mu,sigma2,p,tol=0.001,max_it = 200):
    n = len(x);
    f = np.zeros((n,K))
    r = np.zeros((n,K))
    iter, diff = 0, 1
    loglike = -np.inf
    while np.abs(diff)>tol and iter < max_it:
        loglike_old = loglike
        for k in range(K):
            f[:,k] = st.norm.pdf(x,mu[k],np.sqrt(sigma2[k]))
        for k in range(K):    
            r[:,k] = p[k]*f[:,k]/f.dot(p) 
            mu[k] = np.sum(x*r[:,k])/np.sum(r[:,k])        
            sigma2[k] = np.sum((x-mu[k])**2*r[:,k])/np.sum(r[:,k])
        p = np.mean(r,axis = 0)
        iter = iter + 1
        loglike = np.sum(np.log(f.dot(p)))
        diff = loglike - loglike_old
    return mu, sigma2,p,iter,loglike

K = 2
mu = np.array([1.0,-1.0])
sigma2 = np.array([1.0,1.0])
p = np.array([0.5,0.5])
x = x.ravel()
mu, sigma2,p,iter,loglike = em_mixture1(x,K,mu,sigma2,p)



##############################
## Example 8.3 (section 8.5.3)
##############################
## HMM model code simulation
  
import numpy  as np
import pandas as pd
from scipy.stats import norm
from sklearn.cluster import KMeans
from scipy import special,stats

def Forward_Backward_Algorithm(pi, PP, trans):
    T,S = PP.shape
    alpha = np.zeros((T,S))      # 向前概率
    beta  = np.zeros((T,S))      # 向后概率
    coef  = np.zeros((T,1))      # 正则化系数
    alpha[0,:] = pi.T.dot(np.diag(PP[0,:]))
    coef[0,0]  = 1/np.sum(alpha[0,:])
    alpha[0,:] = alpha[0,:]*coef[0,0]       # 估计正则化
    for t in np.arange(1, T):
        alpha[t,:] = alpha[[t-1],:].dot(trans).dot(np.diag(PP[t,:]))
        coef[t,0]  = 1/np.sum(alpha[t,:])
        alpha[t,:] = alpha[t,:]*coef[t,0]       
    beta[T-1,:] = coef[T-1,0]
    for t in reversed(np.arange(0, T-1)):
        beta[t,:] = (trans.dot(np.diag(PP[t+1,:])).dot(beta[[t+1],:].T)).T
        beta[t,:] = beta[t,:]*coef[t,0]     
    return alpha, beta, coef

def Baum_Welch_Algorithm(d, S):  
    # d: T*1  S: 隐状态个数
    T = len(d)   
    # K均值方法获取初始值
    kmeans_model = KMeans(S)
    model_result = kmeans_model.fit(d)
    mean_ini  = np.sort(model_result.cluster_centers_ ,0) 
    sigma_ini = 0.1*np.ones((S,1))
    pi_ini    = 0.5*np.ones((S,1))
    trans_ini = 0.5*np.ones((S,S))
    PP = norm.pdf(d, loc=mean_ini.ravel(), scale=sigma_ini.ravel()**(0.5))
    alpha, beta, coef = Forward_Backward_Algorithm(pi_ini, PP, trans_ini)
    LT_old = -np.sum(np.log(coef)) 
    # 算法主体
    tol, max_it, diff = 0.001, 10, 100
    it  = 0
    while diff > tol and it < max_it:
        r = np.zeros((T,S))
        h = np.zeros((T-1, S, S))
        for t in np.arange(T-1):
            h[t] = alpha[[t],:].T*trans_ini*PP[t+1,:]*beta[t+1,:]          
        r[0:-1,:] = np.sum(h,2)
        r[-1,:] = alpha[-1,:]/np.sum(alpha[-1,:])
        # 参数估计
        pi_hat = r[0]        
        trans_hat = h.sum(0) / r[0:-1,:].sum(0).reshape(S,1)
        mean_hat, sigma_hat = np.zeros((S, 1)), np.zeros((S,1))
        for j in np.arange(S):
            mean_hat[j]  = r[:,[j]].T.dot(d)/np.sum(r[:,j])
            sigma_hat[j] = r[:,[j]].T.dot((d - mean_hat[j])**2)/np.sum(r[:,j])           
        # 计算似然函数
        PP = norm.pdf(d, loc=mean_hat.ravel(), scale=sigma_hat.ravel()**(0.5))
        alpha, beta, coef = Forward_Backward_Algorithm(pi_hat, PP, trans_hat)       
        LT_new = -np.sum(np.log(coef))        
        diff = np.abs(LT_new - LT_old)     
        LT_old, trans_ini, mean_ini, sigma_ini = LT_new, trans_hat, mean_hat, sigma_hat            
        it += 1
    return mean_hat, sigma_hat, pi_hat, trans_hat, LT_new, alpha[-1,:]



## 模拟
T     = 200
S     = 2
pi    = np.array([0.5,0.5])
trans = np.array([[0.3,0.7],[0.6,0.4]])
mean  = np.array([-1,1])
sigma = np.array([0.08,0.08])
states = np.arange(1, S+1)              # 状态
state_seq   = np.zeros(T)               # 隐状态序列
state_seq[0] = np.random.choice(states, p = np.ravel(pi))
for t in np.arange(1,T):     
    state_seq[t] = np.random.choice(states, p = trans[states.tolist().index(state_seq[t-1]),:])
observed_seq = np.zeros((T, 1))         # 观测序列  
for t in range(T):       
    state_index     = states.tolist().index(state_seq[t])
    observed_seq[t] = np.random.randn()*sigma[state_index]**0.5 + mean[state_index]
mean_hat, sigma_hat, pi_hat, trans_hat, LT, alpha_T = Baum_Welch_Algorithm(observed_seq, S)




##############################
## Case Study (section 8.5.4)
##############################

def Viterbi_Algorithm(d, pi, trans, mean, sigma):
    T, S = len(d), len(pi)
    V = [{}]
    path = {}
    PP = np.zeros((T,S))
    for i in range(S):
        PP[:,i] = norm.pdf(d.ravel(), loc = mean[i], scale = sigma[i]**(0.5))
    emit = PP/np.sum(PP,1).reshape(T,1) 
    for i in range(S):
        V[0][i] = pi[i] * emit[0,i]
        path[i] = [i]
    for t in range(1,T):
        V.append({})
        newpath = {}
        for i in range(S):
            (prob, state) = max([(V[t-1][y0] * trans[y0,i] * emit[t,i], y0) for y0 in range(S)])
            V[t][i] = prob
            newpath[i] = path[state] + [i]
        path = newpath 
    (prob, state) = max([(V[T - 1][y], y) for y in range(S)])
    return (prob, path[state])



## 实证
#index_path = r'E:\bookdata\SZ399300.TXT'
index_path = r'D:\hot\book\data\SZ399300.TXT'
index300 = pd.read_table(index_path,\
    encoding = 'cp936',header = None)
idx = index300[:-1]
idx.columns = ['date','o','h','l','c','v','to']
idx.index = idx['date']
idx['rt'] = idx['c'].pct_change()
idx.dropna(inplace=True)
# boxcox转换
y = (idx['rt'] + np.abs(idx['rt'].min())+0.01)*100
lam_range = np.linspace(-2,5,100)
llf = np.zeros(lam_range.shape, dtype=float)
for i,lam in enumerate(lam_range):
    llf[i] = stats.boxcox_llf(lam, y)
lam_best = lam_range[llf.argmax()]
y_boxcox = special.boxcox1p(y, lam_best)
d = y_boxcox.values.reshape(len(idx),1)

training = 250
S = 3
i = 0
actual = []
predict = []
while i+training < len(d):
    print(i)
    d_t = d[i:i+training,:]
    mean_hat, sigma_hat, pi_hat, trans_hat, LT, alpha_T = Baum_Welch_Algorithm(d_t, S)    
    pp = alpha_T.reshape(1,S).dot(trans_hat)
    # 状态预测
    predict.append(np.argmax(pp.ravel()))   
    # 采用viterbi算法获取真实状态
    d_t2 = d[i:i+training+1,:]  
    mean_hat, sigma_hat, pi_hat, trans_hat, LT, alpha_T = Baum_Welch_Algorithm(d_t2, S)    
    prob,path = Viterbi_Algorithm(d_t2, pi_hat, trans_hat, mean_hat, sigma_hat)
    actual.append(path[-1])   
    i += 1
result = pd.DataFrame(index=idx.iloc[training:,:].index)
result['actual'] = actual
result['predict'] = predict
accuracy = np.sum(result['actual'] == result['predict'])/len(result)
