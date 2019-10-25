
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import pandas as pd

##############################
## Example 2.1 (section 2.4.3)
##############################
n = 100
x = np.random.randn(n,1)
error = np.random.randn(n,1)*0.4
y = 1 + 2*x + error
X = np.hstack((np.ones((n,1)),x))
beta = la.inv(X.T.dot(X)).dot(X.T).dot(y)
yhat = X.dot(beta)
u = np.linspace(-3,3,100)
fu = beta[0] + beta[1]*u
plt.plot(x,y,'o')  # 散点图
plt.plot(u,fu,'r-')


##############################
## Example 2.2 (section 2.5.1)
##############################
#index_path = r'D:\SZ399300.TXT'  #或者 'D:/SZ399300.TXT'
index_path = 'D:/hot/book/data/SZ399300.TXT'

f = open(index_path, 'r')  #指向文件
close0 = []
for line in f:
    split_line = line.split('\t')
    if len(split_line)>4:   #省略最后一行中文（数据来源：通达信）
        tmp = float(split_line[4])
        close0.append(tmp)
f.close()
plt.plot(close0)


##############################
## Case Study (section 2.5.1)
##############################
#index_path = r'D:\SZ399300.TXT'  #或者 'D:/SZ399300.TXT'
index0 = pd.read_table(index_path,encoding = 'cp936',header = None)
index0.columns=['date','o','h','l','c','v','to']
index1 = index0[:-1] #处理最后一行的中文
index1.index = index1['date'] #把第一列日期设为指标，作为完整指标

#stock_path = 'D:\\hs300'
stock_path = r'D:\hot\book\data\hs300-2\hs300'
import os #操作系统交互模块
names = os.listdir(stock_path) #获得文件夹中300个文件名

close = []
for name in names:
    spath = stock_path + '/' + name # 个股数据位置
    tmp = pd.read_table(spath,\
    encoding = 'cp936',header = None) #读入个股数据
    df = tmp[:-1] #处理最后一行的中文
    df.columns = ['date','o','h','l','c','v','to']
    df.index = df['date'] #设置指标
    df1 = df.reindex(index1.index, method = 'ffill')  #一次向前填补
    df2 = df1.fillna(method = 'bfill')  #一次向后填补
    close.append(df2['c'].values)       #取出收盘价

close = np.asarray(close).T
retx = (close[1:,:] - close[0:-1,:])/close[:-1,:] # 获得收益率矩阵


##############################
## Case Study (section 2.6.2)
##############################

# Function A: input data
def read_close(path,name):
    f = open(path +  name, 'r')
    close_price = []
    for line in f:
        st = line.split('\t')
        if len(st)>4:
            close_price.append(float(st[4]))
    f.close()
    return close_price

# Function B: process data
def process_data(close):
    c = np.asarray(close)
    ret = (c[1:] - c[:-1])/c[:-1]
    return ret

# Function C: plot histogram
def out_put(ret,k=30):
    plt.hist(ret,k)


class process_stock():
    ##--- data and variable --
    stock_name = ''
    stock_path = ''
    stock_code = ''
    stock_quote = {}
    stock_return = []

    ##--- functions and methods --
    def read_close(self):
        path = self.stock_path + self.stock_code
        f = open(path, 'r')
        close_price = []
        for line in f:
            st = line.split('\t')
            if len(st)>4:
                close_price.append(float(st[4]))
        f.close()
        self.stock_quote['close'] = close_price

    def cal_return(self):
        c = np.asarray(self.stock_quote['close'])
        ret = (c[1:] - c[:-1])/c[:-1]
        self.stock_return = ret
    def hist_return(self,k):
        plt.hist(self.stock_return,k)


p1 = process_stock()
p1.stock_path = 'D:\\hs300'
p1.stock_code = '600000.txt'
p1.read_close()
p1.cal_return()
p1.hist_return(15)