
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

##############################
## Example 7.1 (section 7.1.2)
##############################
## cross validation for local polynomial regression


n = 100
x = np.random.rand(n,1)
error = np.random.randn(n,1)*0.3
y = np.sin(2*np.pi*x) + error

from sklearn.model_selection import KFold
z = np.hstack((x,y))
kf = KFold(n_splits=10)
## cross-validation
cv_seq = []
h_seq = np.linspace(0.01,0.1,20)
for h in h_seq:
    cv = 0
    for train, test in kf.split(z):
        train_x = x[train]
        train_y = y[train]
        test_x = x[test]
        test_y = y[test]
        yhat = nw_est(train_x,train_y,h,test_x)
        cv = cv + np.mean(test_y - yhat)**2
    cv_seq.append(cv/10)

plt.plot(h_seq,cv_seq,'-')
h_cv = h_seq[np.argmin(cv_seq)]





##############################
## Example 7.2 (section 7.2.2)
##############################
## AIC BCIcomparison

rec_IC = []
n = 100 # 500，5000
for rr in range(1000):
    x1 = np.random.randn(n,1)
    x2 = np.random.randn(n,1)
    err = np.random.randn(n,1)*0.8
    y = -1 + 2*x1 + 0*x2  + err
    c = np.ones((n,1))
    X = np.hstack((c,x1))
    X2 = np.hstack((c,x1,x2))
    beta_1 = la.inv(X.T.dot(X)).dot(X.T).dot(y)
    beta_2 = la.inv(X2.T.dot(X2)).dot(X2.T).dot(y)
    err1 = np.mean((y - X.dot(beta_1))**2)
    err2 = np.mean((y - X2.dot(beta_2))**2)
    AIC1 = err1 + 2*2*err2/n  #AIC1 = err1 + 2*2*err1/n
    BIC1 = err1 + np.log(n)*2*err2/n
    AIC2 = err2 + 2*3*err2/n
    BIC2 = err2 + np.log(n)*3*err2/n
    rec_IC.append([AIC1,AIC2,BIC1,BIC2])
matx = np.asarray(rec_IC)


# 增大样本量 n，观察 AIC 和 BIC 是否具有选择相合性
np.sum(matx[:,0]<matx[:,1]) # AIC选择正确模型的次数
np.sum(matx[:,2]<matx[:,3]) # BIC选择正确模型的次数






##############################
## Example 7.3 (section 7.3)
##############################
# effective degree of freedom
# use example of local constant regression

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
n = 100
x = np.random.rand(n,1)
e = np.random.randn(n,1)*0.3
y = np.sin(2*np.pi*x) + e

plt.plot(x,y,'.')

def eff_df(x,h):
    t = (x - x.T)/h
    K = np.exp(-0.5*t**2)/(np.sqrt(2*np.pi))/h
    S = K/np.sum(K,axis = 0)
    df = np.trace(S)
    return df

h = 0.07

df = eff_df(x,h)
print(df)

def nw_est(x,y,h,u):
    fu = []
    for u0 in u:
        kh_x = np.exp(-0.5*((x-u0)/h)**2)/h
        y0 = np.sum(kh_x*y)/np.sum(kh_x)
        fu.append(y0)
    fu = np.asarray(fu)
    return fu

#使用 AIC 准则选择窗宽
import scipy.stats as stats
hs = np.linspace(0.01,0.1,20)
aic_seq = []
for h in hs:
    yhat = nw_est(x,y,h,x)
    df = eff_df(x,h) + 1
    sigma = (np.sum((y.ravel() - yhat)**2)/n)**(0.5)
    LogLike = np.sum(np.log(stats.norm.pdf(y.ravel(),yhat,sigma)))
    aic = -2*LogLike + 2*df
    #aic = n*np.log(sigma**2) + 2*df
    aic_seq.append(aic)

plt.plot(hs,aic_seq)




##############################
## Case Study (section 7.3)
##############################

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
n = 100
x = np.random.rand(n,1)
e = np.random.randn(n,1)*0.3
y = np.sin(2*np.pi*x) + e

plt.plot(x,y,'.')

def eff_df2(x,h):
    n = len(x)
    c = np.ones((n,1))
    e = np.array([1,0,0])
    S = np.zeros((n,n))
    for i,x0 in enumerate(x):
        X = np.hstack((c,x-x0,(x-x0)**2))
        t = (x - x0)/h
        w = np.exp(-0.5*t**2)/h
        W = np.diag(w.ravel())
        beta = e.dot(la.inv(X.T.dot(W).dot(X))).dot(X.T).dot(W)
        S[i,:] = beta
    df = np.trace(S)
    return df



def local2x(x,y,h,u):
    n = len(y)
    fu = []
    c = np.ones((n,1))
    for u0 in u:
        X = np.hstack((c,x-u0,(x-u0)**2))
        t = (x - u0)/h
        w = np.exp(-0.5*t**2)/h
        W = np.diag(w.ravel())
        beta = la.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(y)
        fu.append(beta)
    return fu




call_price = np.array([9.15,8.15,7.15,6.15,5.1,4.15,\
                      3.2,2.26,1.46,0.84,0.44,0.21,0.1,0.06])
call_K =np.linspace(23,36,14).reshape(14,1)
#plt.plot(call_K,call_price)


hs = np.linspace(0.2,0.5,10)

aic_seq = []
for h in hs:
    yhat = local2x(x,y,h,x)
    yhat = np.squeeze(np.asarray(yhat))
    df = eff_df2(x,h) + 1 
    sigma = np.sum((y.ravel() - yhat[:,0])**2)/n
    aic = n*np.log(sigma) + 2*df
    aic_seq.append(aic)

plt.plot(hs,aic_seq)

plt.savefig(r'D:\hot\book\chapter7\fig\c7-optionaic')


h = hs[np.argmin(aic_seq)]
h = 0.3
u = np.linspace(23,36,50)
y = np.asarray(call_price).reshape(14,1)
x = np.asarray(call_K).reshape(14,1)
fu = local2x(x,y,h,u)
fu = np.squeeze(np.asarray(fu))
plt.plot(u,fu[:,2],'b-')
plt.legend(['spd estimate'])
plt.savefig(r'D:\hot\book\chapter7\fig\c7-option2')



