
##############################
## Example 6.1 (section 6.1.1)
##############################
# cubic spline
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
n = 100
x = np.random.rand(n,1)
error = np.random.randn(n,1)*0.3
y = np.sin(2*np.pi*x) + error

plt.plot(x,y,'o')
xi1 = 1/3
xi2 = 2/3
k1 = (x - xi1)**3*(x - xi1>0)
k2 = (x - xi2)**3*(x - xi2>0)

c = np.ones((n,1))
X = np.hstack((c,x,x**2,x**3,k1,k2))
beta = la.inv(X.T.dot(X)).dot(X.T).dot(y)
yhat = X.dot(beta)
rk = x.ravel().argsort()
plt.plot(x,y,'bo')
plt.plot(x[rk],yhat[rk],'r-')

##############################
## Example 6.2 (section 6.2)
##############################
## Kernel trick
tau = 1
K = np.exp(-tau *(x - x.T)**2)
ld = 0.01
alpha = la.inv(K + ld* np.eye(n)).dot(y)
yhat2 = K.dot(alpha)

plt.plot(x[rk],yhat[rk],'r-')
plt.plot(x[rk],yhat2[rk],'k--')
plt.plot(x,y,'bo')
plt.legend(['cubic spline','kernel regression'])
plt.savefig(r'D:\hot\book\chapter6\fig\spline')


##############################
## Example 6.3 (section 6.3.1)
##############################
## KNN method 1d

def knn1(y,x,k,u):
    fu = []
    for u0 in u:
        dist = np.abs(x-u0).ravel()
        id = dist.argsort()[:k]
        fu.append(np.mean(y[id]))
    return fu
#模拟仿真
u = np.linspace(0,1,100)
fu = knn1(y,x,15,u)

plt.plot(u,np.array(fu),'r-')
plt.plot(x,y,'bo')
plt.legend(['knn estimate, k=15'])

plt.savefig(r'D:\hot\book\chapter6\fig\knn1')

##############################
## Example 6.4 (section 6.3.1)
##############################
## KNN method 2d
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
# data generation 
n = 100
x1 = np.random.randn(n,2)
x2 = np.random.randn(n,2) + np.array([2,2])
X = np.vstack((x1,x2))
y1 = np.ones((n,1))
y2 = np.zeros((n,1))
y = np.vstack((y1,y2))
plt.figure()
plt.plot(x1[:,0],x1[:,1],'ro')
plt.plot(x2[:,0],x2[:,1],'bo')



def knn2(y,x,k,u):
    # x and u are two dimentional
    dist = np.sum((x-u)**2,axis = 1).ravel()
    id = dist.argsort()[:k]
    return np.mean(y[id])

u1 = np.linspace(-2,4,100)
u2 = np.linspace(-2,4,100)
rec = []
for u11 in u1:
    val = 1000
    temp = []
    for u22 in u2:
        u = np.array([u11,u22])
        est = knn2(y,X,15,u)
        d0 = np.abs(est-0.5)
        if d0 < val:
            temp.append(u)
            val = d0
    rec.append(temp[-1])
rec2 = np.array(rec)
plt.figure()

#plt.plot(u,fu,'k-')
plt.plot(rec2[:,0],rec2[:,1],'k-')
plt.plot(x1[:,0],x1[:,1],'r^')
plt.plot(x2[:,0],x2[:,1],'bo')
plt.legend(['knn boundary, k=15'])

plt.savefig(r'D:\hot\book\chapter6\fig\knn2')


##############################
## Example 6.5 (section 6.3.3)
##############################
## NW estimate and local lienar estimate

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
n = 100
x = np.random.rand(n,1)
error = np.random.randn(n,1)*0.3
y = np.sin(2*np.pi*x) + error

# 
#定义NW估计函数


def nw_est(x,y,h,u):
    fu = []
    for u0 in u:
        kh_x = np.exp(-0.5*((x-u0)/h)**2)/h
        y0 = np.sum(kh_x*y)/np.sum(kh_x)
        fu.append(y0)
    fu = np.asarray(fu)
    return fu

def local_linear(x,y,h,u):
    n = len(y)
    fu = []
    c = np.ones((n,1))
    for u0 in u:
        X = np.hstack((c,x-u0))
        t = (x - u0)/h
        w = np.exp(-0.5*t**2)/h
        W = np.diag(w.ravel())
        beta = la.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(y)
        fu.append(beta[0])
    fu = np.asarray(fu)
    return fu

h = 0.1
u = np.linspace(0,1,100)
fu1 = nw_est(x,y,h,u)
fu2 = local_linear(x,y,h,u)

#作图

plt.plot(u,fu1,'r-')
plt.plot(u,fu2,'k--')
plt.plot(x,y,'o')
plt.legend(['local constant','local linear'])
plt.savefig(r'D:\hot\book\chapter6\fig\localpoly')



#############################
## Case Study section(6.3.4)
#############################


import numpy.linalg as la
import numpy as np
import matplotlib.pyplot as plt
call_price = np.array([9.15,8.15,7.15,6.15,5.1,4.15,\
                      3.2,2.26,1.46,0.84,0.44,0.21,0.1,0.06])
call_K =np.linspace(23,36,14).reshape(14,1)
plt.plot(call_K,call_price)
plt.xlabel('strike price')
plt.ylabel('option price')
plt.savefig(r'D:\hot\book\chapter6\fig\c6-option1')

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

h = 1
u = np.linspace(23,36,50)
y = np.asarray(call_price).reshape(14,1)
x = np.asarray(call_K).reshape(14,1)
fu = local2x(x,y,h,u)
fu = np.asarray(fu).reshape(50,3)
plt.plot(u,fu[:,2],'r-')
plt.legend(['spd estimate'])
plt.savefig(r'D:\hot\book\chapter6\fig\c6-option2')
