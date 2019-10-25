

##############################
## Case Study (section 10.3.4)
##############################

####################################  code from chapter 5
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

# model fitting
from sklearn import linear_model
from sklearn.metrics import classification_report
clf = linear_model.LogisticRegression(C=1e2,fit_intercept=True)
clf.fit(X_train,y_train)

# performance in training sample
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

portfolio_ret = np.sum(holding_matrix*test[p:],axis = 1)
plt.plot(np.cumprod(1+portfolio_ret))


################################################## above code from chapter 5

# model fitting
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from xgboost import XGBClassifier
#clf = GradientBoostingClassifier(n_estimators=10, max_depth=5, min_samples_split=2, learning_rate=0.1)
#clf = RandomForestClassifier(n_estimators=10,max_depth=5,min_samples_leaf=2)
clf = XGBClassifier(learning_rate=0.1)
clf.fit(X_train,y_train)

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

tmp_ret_xg = np.sum(holding_matrix*test[p:],axis = 1)
portfolio_ret_xg = np.append(0,tmp_ret_xg)
plt.plot(np.cumprod(1+portfolio_ret_xg))
plt.legend(['Cumulative Return via XGBoost'])
plt.savefig(r'D:\hot\book\chapter10\fig\stockret-xg')


# model fitting
#clf = GradientBoostingClassifier(n_estimators=10, max_depth=2, min_samples_split=2, learning_rate=0.1)
clf =  RandomForestClassifier(n_estimators=10,max_depth=5,min_samples_split=2)
#clf = XGBClassifier(learning_rate=0.1)
clf.fit(X_train,y_train)

# performance in test sample
y_pred = clf.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
np.corrcoef([y_test,y_pred]) # Information Coefficient, IC

## portfolio
holding_matrix = np.zeros((n1-p,300))
for j in range(n1-p):
    prob = clf.predict_proba(test[j:j+p,:].T)[:,1]
    long_position = prob.argsort()[-10:]
    short_position = prob.argsort()[:10]
    holding_matrix[j,long_position] = 0.05
    holding_matrix[j,short_position] = -0.05

tmp_ret_rf = np.sum(holding_matrix*test[p:],axis = 1)
portfolio_ret_rf = np.append(0,tmp_ret_rf)
plt.plot(np.cumprod(1+portfolio_ret_rf))
plt.legend(['Cumulative Return via RF'],loc='upper left')
plt.savefig(r'D:\hot\book\chapter10\fig\stockret-rf')



# model fitting
clf = GradientBoostingClassifier(n_estimators=10, max_depth=5, min_samples_split=2, learning_rate=0.1)
#clf = RandomForestClassifier(n_estimators=10,min_samples_leaf=2)
#clf = XGBClassifier(learning_rate=0.1)
clf.fit(X_train,y_train)

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

tmp_ret_gb = np.sum(holding_matrix*test[p:],axis = 1)
portfolio_ret_gb = np.append(0,tmp_ret_gb)
plt.plot(np.cumprod(1+portfolio_ret_gb))


plt.legend(['Cumulative Return via RF'],loc='upper left')
plt.savefig(r'D:\hot\book\chapter10\fig\stockret-gb')




plt.plot(np.cumprod(1+portfolio_ret_rf))
plt.plot(np.cumprod(1+portfolio_ret_gb))
plt.plot(np.cumprod(1+portfolio_ret_xg))
plt.legend(['random forest','gbdt','xgboost'])
plt.title('Cumulative Return')
plt.savefig(r'D:\hot\book\chapter10\fig\stockret-trees')


plt.plot(np.cumprod(1+portfolio_ret_rf))
plt.plot(np.cumprod(1+portfolio_ret_gb))
plt.plot(np.cumprod(1+portfolio_ret_xg))
plt.plot(np.cumprod(1+portfolio_ret),'--')
plt.legend(['random forest','gbdt','xgboost','logistic regression'])
plt.savefig(r'D:\hot\book\chapter10\fig\stockret-trees')




#import numpy as np
#import matplotlib.pyplot as plt
#plt.rcParams['font.sans-serif']=['SimHei']
#plt.rcParams['axes.unicode_minus'] = False

#p = np.linspace(0,1,1000)
#err1 = np.array([1-np.max([i,1-i]) for i in p])
#err2 = 2*p*(1-p)
#err3 = -p*np.log(p)-(1-p)*np.log(1-p)

#fig = plt.figure()
#plt.plot(p,err1,'r--',label='错误分类率')
#plt.plot(p,err2,'b-.',label='基尼系数')
#plt.plot(p,err3,'g-',label='交叉熵')
#plt.xticks(fontsize=9)
#plt.yticks(fontsize=9)
#plt.xlabel('p',fontsize=9)
#plt.legend(fontsize=9,loc = 'upper right')
#plt.savefig(r'D:\hot\book\code\classify_err')