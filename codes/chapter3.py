import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import pandas as pd

##############################
## Case Study (section 3.1.2)
##############################

## descriptive data analysis
#index_path = 'D:\data\SZ399300.TXT'
index_path = r'D:\hot\book\data\SZ399300.TXT'
data = pd.read_table(index_path,\
    encoding = 'cp936',header = None)
hs300 = data[:-1]  # 删除最后一行
hs300.columns = ['date','o','h','l','c','v','to']
hs300.index = hs300['date']
hs300['ret'] = hs300['c'].pct_change().fillna(0) #计算收益率
hs300_ret = hs300['ret']

hs300_ret.describe()  ## 描述性统计分析
hs300_ret.kurtosis()  ## 计算峰度
hs300_ret.skew()      ## 计算偏度
plt.hist(hs300_ret,30)   # 直方图
plt.savefig(r'D:\hot\book\chapter3\fig\hist311')
plt.boxplot(hs300_ret)# 箱线图
plt.savefig(r'D:\hot\book\chapter3\fig\boxplot311')




##############################
## Example 3.1 (section 3.2.1)
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

x = hs300_ret.values
u = np.linspace(-0.087,0.067,100) ## 估计输出点
h = 1.06*np.std(x)*1340**(-0.2) ## 最优窗宽
fu = ourkde(x,h,u)
plt.hist(x,30,normed = True) ## 对比标准化后的直方图
plt.plot(u,fu,'r-')
plt.savefig(r'D:\hot\book\chapter3\fig\hist311')




##############################
## Example 3.2 (section 3.2.1)
##############################
data = np.random.randn(1000, 5)
df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e'])
pd.plotting.scatter_matrix(df, alpha=0.4, diagonal='kde')


## 复现satter_matrix 函数
u = np.linspace(-2.5,2.5,100)
for j in range(5):
    for i in range(5):
        if i !=j:
            plt.subplot(5,5,i+5*j+1)
            plt.plot(data[:,i],data[:,j],'.')
        if i ==j:
            plt.subplot(5,5,i+5*j+1)
            x = data[:,i]
            h = 1.06*np.std(x)*1000**(-0.2)
            fu = ourkde(x,h,u)
            plt.plot(u,fu,'-')




##############################
## Example 3.3 (section 3.3)
##############################
## Kmeans algorithm

def ourkmean(x,k,mu,tol):
    # x: n*p;  mu: k*p
    n,p = x.shape
    dist_matx = np.zeros((n,k))
    id = []
    iter = 0
    max_it = 100
    diff = 100
    VAL2 = 10000
    while diff>tol and iter<max_it:
        # step 1
        for i in range(k):
            dist_matx[:,i] = np.sum((x - mu[i,:])**2,axis = 1)
        id = np.argmin(dist_matx,axis = 1)
        # step 2
        VAL = 0
        for i in range(k):
            mu[i,:] = np.mean(x[id==i,:],axis = 0)
            VAL = VAL + np.sum((x[id==i,:] - mu[i,:])**2)
        diff = np.abs(VAL - VAL2)
        VAL2 = VAL
        iter = iter +1
    return id, mu

n = 100
x1 = np.random.randn(n,2)
x2 = np.random.randn(n,2) + np.array([2,2])
x = np.vstack((x1,x2))
tol = 0.001
k = 2

mu = np.array([[-0.1,-0.1],[1.0,1.0]]) 
id, mu = ourkmean(x,k,mu,tol)

plt.figure()
plt.plot(x[id==0,0],x[id==0,1],'ro')
plt.plot(x[id==1,0],x[id==1,1],'bo')



##############################
## Case Study (section 3.4.3)
##############################
## digital 3 image data

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
path = r'D:\hot\book\data\zip.train\zip.train'
data = np.loadtxt(path) # 7291*257, first column is number

id3 = data[:,0]==3
data3 = data[id3,1:]     # 658*256 矩阵

j = 330
plt.imshow(data3[j,:].reshape(16,16))
mean3 = np.mean(data3,axis = 0)
plt.imshow(mean3.reshape(16,16)) # 均值3作图

covx = np.cov(data3.T)
u,v = la.eig(covx)
## the j-th PC direction
j = 1 # 1,2,3...
plt.imshow(v[:,j-1].reshape(16,16))
## the j-th PC score
j = 3 # 1,2,3...
xi = (data3 -mean3).dot(v[:,j-1:j])
id = xi.ravel().argsort() # 化为一维数组后提取排序索引
for i in range(5):
    plt.subplot(2,5,i+1)
    plt.imshow(data3[id[-i-1]].reshape(16, 16))
for i in range(5):
    plt.subplot(2,5,5+i+1)
    plt.imshow(data3[id[i]].reshape(16, 16))

plt.savefig(r'D:\hot\book\chapter3\fig\num3-pc4')


## variance explain
np.sum(u[0:50])/np.sum(u)
[np.sum(u[0:a])/np.sum(u) for a in range(50)]

## reconstruction
k = 50
xi = (data3 - mean3).dot(v[:,0:k]) # a set of low dim vector
rec_data3 = mean3 + xi.dot(v[:,0:k].T)

j = 400
plt.subplot(1,2,1)
plt.imshow(data3[j,:].reshape(16,16))
plt.subplot(1,2,2)
plt.imshow(rec_data3[j,:].reshape(16,16))


# SVD with centralized data leads to
# engen decomposition
# data3, mean3 来自案例 手写数字3特征分析
u2,d2,v2 = la.svd(data3 - mean3)
# v[:,0]  ==  v2[0,:] # v2按行存储特征向量

u3,d3,v3 = la.svd(data3)
plt.imshow(v3[0,:].reshape(16,16))

mean3s = mean3/(np.sum(mean3**2))**(0.5)
plt.subplot(1,2,1)
plt.imshow(mean3.reshape(16,16))
plt.subplot(1,2,2)
plt.imshow(-mean3s.reshape(16,16))




##############################
## Case Study (section 3.4.4)
##############################
####################
# Data 2 : bond yield analysis
#path = 'D:\data\t-bond rates.xlsx'
path = r'D:\hot\book\data\t-bond rates.xlsx'
data0 = pd.read_excel(path)
data = data0.iloc[:,1:-2].values
plt.plot(data.T)

mu = np.mean(data,axis = 0)
plt.plot(mu)

## pca direction
covx = np.cov(data.T)
import numpy.linalg as la
u,v = la.eig(covx) # u-eigen value
#                    v eigen vectors
plt.plot(v[:,0])
xi  = (data - mu).dot(v[:,0:1])
rec_data = mu + xi.dot(v[:,0:1].T)
plt.plot(rec_data.T)

plt.plot(v[:,1])
xi2  = (data - mu).dot(v[:,1:2])
id = xi2.ravel().argsort()[-5:]





##############################
## Case Study (section 3.4.5)
##############################
## PCA for stock return data
# covx taken from chapter 2


## real data example

## step 1 / full index
import numpy as np
import numpy.linalg as la
import pandas as pd
import os
import matplotlib.pyplot as plt
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





covx = np.cov(retx.T)
u,v = la.eig(covx)
## second pc
id = v[:,1].argsort()[-10:]
sector1 = [names[a] for a in id]    # finance sector

id = v[:,1].argsort()[:10]
sector2 = [names[a] for a in id]    # media and high tech sector

cname_path = 'D:/hot/book/data/A_share_name.xlsx'
namesheet = pd.read_excel(cname_path,'Sheet1',encoding = 'gbk')
cepair  = namesheet.values
cname  = []
for ecode in sector2: # or in sector2
    ecodex = ecode[2:-4]
    id = cepair[:,0] == int(ecodex)
    cname.append(cepair[id,1][0])



## pca for transposed data
covx2 = np.cov(retx)
ux,vx = la.eig(covx2)

mux = np.mean(retx.T,axis = 0) # market/mean return
xi1 = (retx.T - mux).dot(vx[:,0])

id2 = xi1.argsort()[-10:]
sector1v = [names[a] for a in id2]    # finance sector


cname  = []
for ecode in sector1v: # or in sector2
    ecodex = ecode[2:-4]
    id = cepair[:,0] == int(ecodex)
    cname.append(cepair[id,1][0])

cname
