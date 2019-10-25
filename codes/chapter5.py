


import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt




##############################
## Example 5.1 (section 5.1.1)
##############################
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

## LDA estimation
p1 = 0.5
p2 = 0.5
mu1 = np.mean(x1,axis = 0)
mu2 = np.mean(x2,axis = 0)
S = (np.cov(x1.T)*99 + np.cov(x2.T)*99)/198

## LDA Classification and Boundary
delta1 = X.dot(la.inv(S)).dot(mu1) - 0.5*mu1.dot(la.inv(S)).dot(mu1)
delta2 = X.dot(la.inv(S)).dot(mu2) - 0.5*mu2.dot(la.inv(S)).dot(mu2)
id  = delta1 > delta2

b0 = 0.5*mu1.dot(la.inv(S)).dot(mu1) - 0.5*mu2.dot(la.inv(S)).dot(mu2)
b = (la.inv(S)).dot(mu1-mu2)
u = np.linspace(-4,4,100)
fu = b0/b[1] - b[0]/b[1]*u

plt.figure()
plt.plot(X[id==True,0],X[id==True,1],'ro')
plt.plot(X[id==False,0],X[id==False,1],'bo')
plt.plot(u,fu,'k-')




##############################
## Example 5.2 (section 5.1.1)
##############################

## data generation
def logistic(X,y,beta):
    diff = 1
    iter = 0
    while iter <1000 and diff >0.0001:
        like = np.sum(y*(X.dot(beta)) - np.log(1+np.exp(X.dot(beta))))
        p = np.exp(X.dot(beta))/(1+np.exp(X.dot(beta)))
        w = np.diag((p*(1-p)).ravel())
        z = X.dot(beta) + la.inv(w).dot(y-p)
        beta = la.pinv(X.T.dot(w).dot(X)).dot(X.T).dot(w).dot(z)
        like2 = np.sum(y*(X.dot(beta)) - np.log(1+np.exp(X.dot(beta))))
        diff = np.abs(like - like2)
        iter = iter + 1
    return beta

# 使用例子5.1 LDA 模拟仿真数据
c = np.ones((2*n,1))
X2 = np.hstack((c,X))
beta = la.inv(X2.T.dot(X2)).dot(X2.T).dot(y)
beta_lg = logistic(X2,y,beta)
phat = 1/(1+np.exp(-X2.dot(beta_lg)))

ld = 0.5
id2 = phat.ravel() > ld
u = np.linspace(-4,4,100)
fu = -beta_lg[0]/beta_lg[2] - beta_lg[1]/beta_lg[2]*u

plt.figure()
plt.plot(X[id2==True,0],X[id2==True,1],'ro')
plt.plot(X[id2==False,0],X[id2==False,1],'bo')
plt.plot(u,fu,'k-')
# Comparison with standard procedure
from sklearn import linear_model
clf = linear_model.LogisticRegression(C=1e2,fit_intercept=False)
clf.fit(X2,y)
betax = clf.coef_.T
sum(y*(X2.dot(betax))- np.log(1+np.exp(X2.dot(betax))))
sum(y*(X2.dot(beta_lg)) - np.log(1+np.exp(X2.dot(beta_lg))))










#############################
## Case Study section(5.2.3)
#############################
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

#active_ret = raw - np.mean(raw,axis = 1).reshape(200,1)



# obtain traing set and test set
n = 500
n1 = 50
p = 5
train = retx[-n:-n1,:]
ret = train[p:,:].ravel()
X1 = train[4:-1,:].ravel()[:,np.newaxis]
X2 = train[3:-2,:].ravel()[:,np.newaxis]
X3 = train[2:-3,:].ravel()[:,np.newaxis]
X4 = train[1:-4,:].ravel()[:,np.newaxis]
X5 = train[:-5,:].ravel()[:,np.newaxis]
y_train = (ret>0).astype(int)
X_train = np.hstack((X5,X4,X3,X2,X1))

test = retx[-n1:,:]
ret2 = test[p:,:].ravel()
X1 = test[4:-1,:].ravel()[:,np.newaxis]
X2 = test[3:-2,:].ravel()[:,np.newaxis]
X3 = test[2:-3,:].ravel()[:,np.newaxis]
X4 = test[1:-4,:].ravel()[:,np.newaxis]
X5 = test[:-5,:].ravel()[:,np.newaxis]
y_test = (ret2>0).astype(int)
X_test = np.hstack((X5,X4,X3,X2,X1))

# performance in training sample
from sklearn import linear_model
from sklearn.metrics import classification_report
clf = linear_model.LogisticRegression(C=1e2,fit_intercept=True)
clf.fit(X_train,y_train)
y_pred0 = clf.predict(X_train)
print(classification_report(y_train, y_pred0))
np.corrcoef([y_train,y_pred0])

# performance in test sample
y_pred = clf.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
np.corrcoef([y_test,y_pred]) # Information Coefficient, IC

## portfolio



holding_matrix = np.zeros((n1-p,300))
for j in range(n1-p):
    #prob = clf.predict_proba(test[j:j+5,:].T)[:,1]
    prob = clf.predict_proba(test[j:j+p,:].T)[:,1]
    long_position = prob.argsort()[-10:]
    short_position = prob.argsort()[:10]
    holding_matrix[j,long_position] = 0.05
    holding_matrix[j,short_position] = -0.05

tmp_ret = np.sum(holding_matrix*test[p:],axis = 1)
portfolio_ret = np.append(0,tmp_ret)
plt.plot(np.cumprod(1+portfolio_ret))
plt.legend(['Logistic Regression'],loc='upper left')
plt.savefig(r'D:\hot\book\chapter5\fig\stockret-lr')


plt.plot(np.cumprod(1+portfolio_ret))
plt.plot(np.cumprod(1+portfolio_ret_nn),'--')
plt.legend(['Logistic Regression','neural network'])
plt.savefig(r'D:\hot\book\chapter5\fig\stockret-lrnn')








##############################
## Example 5.3 (section 5.4.3)
##############################

## Classification Metrics 
# y, X2, beta_lg 来自例子 5.2
S = len(y)
PO = np.sum(y)
NO = S - PO
prob = 1/(1+np.exp(-X2.dot(beta_lg)))
ld = 0.5
y2 = (prob>ld).astype(int)
PX = np.sum(y2)
NX = S - PX
TP = np.sum(y2*y)
FP = PX - TP 
FN = PO - TP
TN = NO - FP  # = NX - FN 
# Confusion Matrix
np.array([[TP,FP],[FN,TN]]).astype(int)
## F1 Score
TPR = TP/(TP + FN)
PPV = TP/(TP + FP)
F1 = 2*TPR*PPV/(TPR + PPV)
## ROC
fpr_seq = []
tpr_seq = []
for ld in np.linspace(0.0,0.99,5000):
    y2 = (prob>ld).astype(int)
    PX = np.sum(y2)
    NX = S - PX
    TP = np.sum(y2*y)
    FP = PX - TP 
    FN = PO - TP
    TN =  NX - FN 
    TPR = TP/(TP + FN)
    FPR = FP/(TN + FP)
    fpr_seq.append(FPR.copy())
    tpr_seq.append(TPR.copy())
fpr_seq2 = np.asarray(fpr_seq)[::-1]
tpr_seq2 = np.asarray(tpr_seq)[::-1]

fig = plt.figure(figsize=(10,10))
plt.plot(fpr_seq2,tpr_seq2)
plt.xlim([-0.05,1])
plt.ylim([-0.01,1.05])
plt.xlabel('False Positive Rate',fontsize=24)
plt.ylabel('True Positive Rate',fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.title('ROC curve',fontsize=24)

plt.savefig(r'D:\hot\book\chapter5\fig\roc')



## AUC
fpr_seq3 = np.hstack((0,fpr_seq2))
diff = np.diff(fpr_seq3)
our_auc = np.sum(diff*tpr_seq2)
# Comparision with standard package
from sklearn import metrics
y_score = clf.fit(X2,y).decision_function(X2)
fpri, tpri, _ = metrics.roc_curve(y, y_score)
plt.plot(fpri,tpri)
roc_auc = metrics.auc(fpri, tpri)













