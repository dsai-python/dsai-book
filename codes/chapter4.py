
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
## grouping effect
n = 50
error = np.random.randn(n,1)*0.4
x = np.random.randn(n,1)
y = 2 - 3*x + error
c = np.ones((n,1))
X = np.hstack((c,x))
beta = la.inv(X.T.dot(X)).dot(X.T).dot(y)

X2 = np.hstack((c,x,x))  
beta2 = la.inv(X2.T.dot(X2)).dot(X2.T).dot(y) #  


ld = 0.001
beta_l2 = la.inv(X2.T.dot(X2)+ld*np.eye(3))\
          .dot(X2.T).dot(y)

print(beta_l2)  

beta3 = la.pinv(X2.T.dot(X2)).dot(X2.T).dot(y) # pinv 




##############################
## Example 4.1 (section 4.1.1)
##############################
## regression procedure

## data generation
X = np.random.randn(100,3)
sigma = 0.6
error = np.random.randn(100,1)*sigma
#error = np.random.standard_t(3,size=(100,1))*sigma
beta = np.array([[1,-2,0.5]]).T
y = X.dot(beta) + error

##1 check linearity
plt.plot(X[:,0],y,'o')
#plt.plot(X[:,1],y,'o')

##2 check no strong correlations in X
np.corrcoef(X.T) # 相关系数矩阵

import statsmodels.stats.outliers_influence as infl
[infl.variance_inflation_factor(X,i) for i in range(3)] # VIF 


##3 estimate unknown parameters
beta_ols = la.inv(X.T.dot(X)).dot(X.T).dot(y)

##4 Plots
# QQ plot
res = y - X.dot(beta_ols)
qt = np.linspace(1,99,100)
res_qt = np.percentile(res,qt)
normdata = np.random.randn(100000,)
normal_qt = np.percentile(normdata,qt)
plt.plot(normal_qt,res_qt)

# residual plots
plt.plot(res) # residual in sequence.

yhat = X.dot(beta_ols)
plt.plot(yhat,res,'o') # residual vs fitted value

plt.plot(X[:,1],res,'o') # residual vs each variable

##5 Diagnose: 
# leverage point, influence point
H = X.dot(la.inv(X.T.dot(X))).dot(X.T)
lev = np.diag(H)
lev.sort()[-5:]  # lev.argsort()[-5:] 

#  influence point 练习题
#infl.OLSInfluence.cooks_distance()


##6 Inference 
# confidence interval
MSE = np.mean(res**2)
var_beta = MSE*la.inv(X.T.dot(X))
beta_ols[0] +1.96*(var_beta[0,0])**0.5
beta_ols[0] -1.96*(var_beta[0,0])**0.5

# t-test
j = 2
tstat = beta_ols[j]/(var_beta[j,j])**0.5
# p-value
trnd = np.random.standard_t(97,size=(1000000,))
np.sum(trnd>tstat)/1000000

# R square  练习题



#############################
## Case Study (section 4.1.2) 
#############################

import pandas as pd
path = 'D:/hot/book/data/macro_econ_data.xls'
data = pd.read_excel(path)

debt = data['Revolving Credit'].values[:,np.newaxis]
pce = data['Nominal PCE'].values[:,np.newaxis]
rate = data['Charge-off Rate'].values[:,np.newaxis]

y = debt
x = np.log(pce)
plt.plot(x,y,'o')
n,p = x.shape
c = np.ones((n,1))
X = np.hstack((c,x))
beta = la.inv(X.T.dot(X)).dot(X.T).dot(y)
res = y - X.dot(beta)

res2 = (res - res.mean(axis=0)) / res.std(axis=0)
rate2 = (rate - rate.mean(axis=0)) / rate.std(axis=0)

plt.figure()
plt.plot(res2)
plt.plot(rate2)
plt.legend(['residual','defualt rate'])
plt.savefig(r'D:\hot\book\chapter4\fig\drate-res-ts')

corr_list = [np.corrcoef(res2[:-k].T,rate2[k:].T)[0,1] \
             for k in np.arange(1,15)]

lag = np.array(corr_list).argmax()
plt.plot(res2[:-lag],rate2[lag:],'o')
plt.xlabel('lagged residual')
plt.ylabel('default rate')
plt.savefig(r'D:\hot\book\chapter4\fig\drate-res')







##############################
## Example 4.2 (section 4.2.1)
##############################


def forward0(X,y):
    n,p = X.shape
    seq = []
    inv_seq = list(range(p))
    for j in range(p):
        #beta = la.inv(X[:,seq].T.dot(X[:,seq])).dot(X[:,seq].T).dot(y)
        #使用广义逆
        beta = la.pinv(X[:,seq].T.dot(X[:,seq])).dot(X[:,seq].T).dot(y)
        Z = y - X[:,seq].dot(beta)
        tmp = np.hstack((Z,X[:,inv_seq]))
        corr0 = np.corrcoef(tmp.T)
        id = np.abs(corr0[0,1:]).argmax()
        seq.append(inv_seq[id])
        del inv_seq[id]
    return seq

n = 200
p = 10
x = np.random.rand(n,p)
beta_true = np.array(range(p))
error = np.random.randn(n,1)*0.3
y = x.dot(beta_true.reshape(p,1)) + error
seq0 = forward0(x,y)




#############################
## Case Study section(4.2.2)
#############################
## step 1 / full index
import pandas as pd
import os
index_path = r'D:\hot\book\data\SZ399300.TXT'

index300 = pd.read_table(index_path,\
    encoding = 'cp936',header = None)
idx = index300[:-1]
idx.columns = ['date','o','h','l','c','v','to']
idx.index = idx['date']
## step 2 / data preparation


stock_path = r'D:\hot\book\data\hs300-2\hs300'
names = os.listdir(stock_path)
close = []
for name in names:
    spath = stock_path + '\\' + name
    df0 = pd.read_table(spath,\
        encoding = 'cp936',header = None)
    df1 = df0[:-1]
    df1.columns = ['date','o','h','l','c','v','to']
    df1.index = df1['date']
    df2 = df1.reindex(idx.index,method = 'ffill')
    df3 = df2.fillna(method = 'bfill')
    close.append(df3['c'].values)

data = np.asarray(close).T

retx = (data[1:,:]-data[:-1,:])/data[:-1,:]




#
X = retx # 来自案例 2.5.3
y = np.mean(X,axis = 1).reshape(1339,1)
# 模型训练
seq =  forward0(X[0:500,:],y[0:500,:])
X2 = X[0:500,:]
y2 = y[0:500,:]
id = seq[:50]
beta = la.pinv(X2[:,id].T.dot(X2[:,id])).dot(X2[:,id].T).dot(y2)
beta = beta/np.sum(beta)
# 验证跟踪效果
X3 = X[500:600,:]
y3 = y[500:600,:]
ret_test = X3[:,id].dot(beta)
plt.plot(np.cumprod(1+y3))
plt.plot(np.cumprod(1+ret_test))




plt.legend(['index return','portfolio return'])
plt.savefig(r'D:\hot\book\chapter4\fig\tracking50')



##############################
## Example 4.3 (section 4.2.3)
##############################
def fs(X,y,eps):
    n,p = X.shape
    beta = np.zeros((p,1))
    max_corr = 1
    iter = 0
    beta_matx = []
    while max_corr>0.01 and iter<10000:
        Z = y - X.dot(beta)
        tmp = np.hstack((Z,X))
        corr0 = np.corrcoef(tmp.T)
        id = np.abs(corr0[0,1:]).argmax()
        max_corr = np.abs(corr0[0,1:][id])
        beta[id] = beta[id] + eps*np.sign(corr0[0,1:][id])
        iter = iter +1
        beta_matx.append(beta.copy())
    return beta, beta_matx

n = 200
beta_true = np.array([1,-2,3,-4,5,-6,7,-8])
p = len(beta_true)
X = np.random.rand(n,p)
error = np.random.randn(n,1)*0.3
y = X.dot(beta_true.reshape(p,1)) + error

beta, beta_matx = fs(X,y,0.01)
beta_matx2 = np.squeeze(np.asarray(beta_matx))
plt.plot(beta_matx2)

plt.savefig(r'D:\hot\book\chapter4\fig\fspath')








##############################
## Example 4.4 (section 4.3.2)
##############################



## coordinate decent algorithm
def sfun(t,ld):
    tmp = (np.abs(t)-ld)
    if tmp < 0:
        tmp = 0
    return np.sign(t)*tmp

def coordinate(X,y,beta0,ld):
    beta = beta0.copy()
    n,p = X.shape
    iter = 0
    diff = 1
    VAL = 10000
    while iter<1000 and diff>0.0001:
        for j in range(p):
            beta[j] = 0
            y2 = y - X.dot(beta)
            t = X[:,j].dot(y2)/(X[:,j]**2).sum()
            beta[j] = sfun(t,n*ld/(X[:,j]**2).sum())
        VAL2 = np.sum((y-X.dot(beta))**2) + \
        n*ld*np.sum(np.abs(beta))
        diff = np.abs(VAL2 - VAL)
        VAL = VAL2
        iter = iter + 1
    return beta,iter

n = 100
beta_true = np.array([1,-2,3,-4,5,-6,7,-8])
p = len(beta_true)
X = np.random.randn(n,p)
error = np.random.randn(n,1)*0.3
y = X.dot(beta_true.reshape(p,1)) + error
beta_ols = la.pinv(X.T.dot(X)).dot(X.T).dot(y)
ld = np.log(n)
beta_l1,iter = coordinate(X,y,beta_ols,ld)
print(beta_l1)
# 对比sklearn的lasso函数结果
from sklearn import linear_model
reg = linear_model.Lasso(alpha = ld,fit_intercept=False)
reg.fit (X, y)
reg.coef_



##############################
## Example 4.5 (section 4.3.2)
##############################

def lasso_path(X,y,max_ld,num_ld):
    beta_init = la.pinv(X.T.dot(X)).dot(X.T).dot(y)
    ld_seq = np.linspace(0,max_ld,num_ld)
    beta_path = []
    for ld in ld_seq:
        beta_l1,iter = coordinate(X,y,beta_init,ld)
        beta_init = beta_l1.copy()
        beta_path.append(beta_l1.copy())
    return np.squeeze(np.asarray(beta_path))


beta_path = lasso_path(X,y,10,500)
#plt.plot(beta_path)
#plt.plot(beta_path[::-1,:])

xx = np.sum(np.abs(beta_path[::-1,:]), axis=1)
xx2 = xx/xx[-1]
plt.plot(xx2, beta_path[::-1,:])


plt.savefig(r'D:\hot\book\chapter4\fig\cdpath')


from sklearn import linear_model
_,_,coefs = linear_model.lars_path(X, y.ravel(), method='lasso')
xx = np.sum(np.abs(coefs.T), axis=1)
xx2 = xx/xx[-1]
plt.plot(xx2, coefs.T)





##############################
## Example 4.6 (section 4.4.1)
##############################
n = 50
error = np.random.randn(n,1)*0.4
x = np.random.randn(n,1)
y = 2 - 3*x + error
c = np.ones((n,1))
X = np.hstack((c,x))
beta = la.inv(X.T.dot(X)).dot(X.T).dot(y)

X2 = np.hstack((c,x,x)) #设置后两列值相同
beta2 = la.inv(X2.T.dot(X2)).dot(X2.T).dot(y) # OLS失效
ld = 0.01
beta_l2 = la.inv(X2.T.dot(X2)+ld*np.eye(3))\
          .dot(X2.T).dot(y)
print(beta_l2) #岭回归输出的后两个参数相同

## grouping effect 的SVD解释
u,d,v = la.svd(X2,0)
alpha  = u.T.dot(y)
beta_l2j = np.sum(v[:,j]*alpha.ravel()*d/(d**2 + ld)) # j = 1,2
# ridge regression via svd
[np.sum(v[:,j]*alpha.ravel()*d/(d**2 + ld)) for j in range(3)]



##############################
## Example 4.7 (section 4.4.1)
##############################
## the pinv functio
beta3 = la.pinv(X2.T.dot(X2)).dot(X2.T).dot(y) #use pinv in ols formula
u2,d2,v2 = la.svd(X2.T.dot(X2),0)
inv_d2 = np.zeros((3,3))
inv_d2[:2,:2] = np.diag(d2[:2]**(-1))
v2.T.dot(inv_d2).dot(u2.T) # = la.pinv(X2.T.dot(X2))

[np.sum(v[:2,j]*alpha.ravel()[:2]/d[:2]) for j in range(3)]
# beta3 与主成分回归结果相同






##############################
## Example 4.8 (section 4.4.2)
##############################
## QR algorithm

A = np.random.randn(50,3)
S  = (A.T.dot(A)).copy()
def eig_qr(S):
    vx = np.eye(S.shape[0])
    diff = 1
    iter = 0
    VAL = 0
    while diff >1e-8 and iter <1000:
        Q,R = la.qr(S)
        vx = vx.dot(Q)
        S = R.dot(Q)
        iter = iter + 1
        VAL2 = np.sum(np.abs(vx))
        diff = np.abs(VAL2 - VAL)
        VAL = VAL2
    return np.diag(R),vx

ux,vx = eig_qr(S)
u,v = la.eig(A.T.dot(A))
# 比较v和vx，以及 u和ux
#

