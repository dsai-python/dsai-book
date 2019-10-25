##############################
## Case Study (section 12.2.3)
##############################
## DDQN + PER for trading/ Trader version
## states only contain tech index/ some positive results 
##程序框架使用了网页（https://github.com/jaromiru/AI-blog）的DQN程序主体架构 


#import random, numpy, math, gym, scipy
import random, numpy, math, scipy

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros( 2*capacity - 1 )
        self.data = numpy.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

import tensorflow as tf
print(tf.VERSION)  ## 1.14.0
print(tf.keras.__version__)  ## 2.2.4-tf

from keras import backend as K
#import tensorflow.keras.backend as K
#from tensorflow.python.keras import backend as K

def hubert_loss(y_true, y_pred):    # sqrt(1+a^2)-1
    err = y_pred - y_true
    return K.mean( K.sqrt(1+K.square(err))-1, axis=-1 )

#-------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

class Brain:
    def __init__(self, stateCnt, actionCnt,layers_para):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.layer1_para = layers_para[0]
        self.layer2_para = layers_para[1]

        self.model = self._createModel()
        self.model_ = self._createModel()  # target network

    def _createModel(self):
        model = Sequential()
        model.add(Dense(self.layer1_para, kernel_initializer='lecun_uniform', input_shape=(self.stateCnt,)))
        model.add(Activation('relu'))
        #model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

        model.add(Dense(self.layer2_para, kernel_initializer='lecun_uniform'))
        model.add(Activation('relu'))
        #model.add(Dropout(0.2))
        #model.add(Dense(150, init='lecun_uniform'))
        #model.add(Activation('relu'))

        model.add(Dense(3, kernel_initializer='lecun_uniform'))
        model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

        opt = Adam()
        model.compile(loss=hubert_loss, optimizer=opt)

        return model


    def _createModel2(self):
        model = Sequential()
        model.add(Dense(164, kernel_initializer='lecun_uniform', input_shape=(64,)))
        model.add(Activation('relu'))
        #model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

        model.add(Dense(150, kernel_initializer='lecun_uniform'))
        model.add(Activation('relu'))
        #model.add(Dropout(0.2))
        #model.add(Dense(150, init='lecun_uniform'))
        #model.add(Activation('relu'))

        model.add(Dense(1, kernel_initializer='lecun_uniform'))
        model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

        opt = Adam()
        model.compile(loss=hubert_loss, optimizer=opt)

        return model




    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=32, epochs=epoch, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, self.stateCnt), target=target).flatten()

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample) 

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

#-------------------- AGENT ---------------------------
MEMORY_CAPACITY = 80000

BATCH_SIZE = 30

GAMMA = 0.95

MAX_EPSILON = 1
MIN_EPSILON = 0.01

EXPLORATION_STOP = 50000   # at this step epsilon will be 0.01
LAMBDA = - math.log(0.01) / EXPLORATION_STOP  # speed of decay

UPDATE_TARGET_FREQUENCY = 1000

class Agent:
    steps = 0
    eps_steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt,layers_para):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt, layers_para)
        self.memory = Memory(MEMORY_CAPACITY)
        
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return numpy.argmax(self.brain.predictOne(s))


    def observe(self, sample):  # in (s, a, r, s_) format
        x, y, errors = self._getTargets([(0, sample)])
        self.memory.add(errors[0], sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def _getTargets(self, batch):
        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([ o[1][0] for o in batch ])
        states_ = numpy.array([ (no_state if o[1][3] is None else o[1][3]) for o in batch ])

        p = agent.brain.predict(states)

        p_ = agent.brain.predict(states_, target=False)
        pTarget_ = agent.brain.predict(states_, target=True)

        x = numpy.zeros((len(batch), self.stateCnt))
        y = numpy.zeros((len(batch), self.actionCnt))
        errors = numpy.zeros(len(batch))
        
        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]; done = o[4]
            
            t = p[i]
            oldVal = t[a]
            if done:
                t[a] = r
            else:
                t[a] = r + GAMMA * pTarget_[i][ numpy.argmax(p_[i]) ]  # double DQN

            x[i] = s
            y[i] = t
            errors[i] = abs(oldVal - t[a])

        return (x, y, errors)

    def replay(self):    
        batch = self.memory.sample(BATCH_SIZE)
        x, y, errors = self._getTargets(batch)

        #update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        self.brain.train(x, y)



##########################
## start
##########################


## processing data
import talib as ta
import pandas as pd
import matplotlib.pyplot as plt
path_HS300 = r'D:\hot\book\data\SZ399300.TXT'


hs300 = pd.read_table(path_HS300,sep='\s+',header=None,encoding = 'gbk')
hsindex = hs300[0:-1] 

datax = hsindex
datax.columns = ['date','open','high','low','close','volume','turnover']

openx = np.asarray(datax['open'])
closex = np.asarray(datax['close'])
highx = np.asarray(datax['high'])
lowx = np.asarray(datax['low'])
volumex = np.asarray(datax['volume'])
turnoverx = np.asarray(datax['turnover'])

roc1 = ta.ROC(closex,1)
roc5 = ta.ROC(closex,5)
roc10 = ta.ROC(closex,10)
rsi = ta.RSI(closex,14)
ultosc = ta.ULTOSC(highx,lowx,closex)
slowk,slowd = ta.STOCH(highx,lowx,closex)
slow_diff = slowk - slowd
adx = ta.ADX(highx,lowx,closex,14)
ppo = ta.PPO(closex)
willr = ta.WILLR(highx,lowx,closex)
cci = ta.CCI(highx,lowx,closex)
bop = ta.BOP(openx,highx,lowx,closex)
std_p20 = ta.STDDEV(closex,20)
angle = ta.LINEARREG_ANGLE(closex)
mfi = ta.MFI(highx,lowx,closex,volumex)
adosc = ta.ADOSC(highx,lowx,closex,turnoverx)

## obtian training data


start_day = '20150105'
end_day = '20150714'
st = (datax['date']==start_day).nonzero()[0][0]
ed = (datax['date']==end_day).nonzero()[0][0]

features = np.column_stack((roc1, roc5, roc10,rsi,ultosc,slowk,slow_diff,adx,ppo,willr,\
                            cci,bop,std_p20,angle,mfi,adosc))


features2 = np.float64(features[st:ed,:])

close2 = closex[st:ed]


mu1 = np.mean(features2,axis = 0)
std1 = np.std(features2,axis = 0)

Xt = (features2 - mu1)/std1
Xt[Xt>3] = 3
Xt[Xt<-3] = -3

n,p = Xt.shape

stateCnt  = p
actionCnt = 3
layers_para = [100,100]
epochs = 100
agent = Agent(stateCnt, actionCnt, layers_para)


for i in range(epochs):       

    j = 0
    R = 0
    while j<n-1:
          
        s = Xt[j,:]
        a = agent.act(s)
        s_ = Xt[j+1,:]
        r = (close2[j+1] - close2[j])*(a - 1)
        done = False 
        agent.observe( (s.reshape(stateCnt,), a, r, s_.reshape(stateCnt,),done) )
        agent.replay()              
        s = s_
        R += r
        j += 1

    print([i,R])


####################################
## insample test
####################################
j = 0
R = 0
pnl = [R]
actions = []
while j<n-1: 
    s = Xt[j,:]
    a = numpy.argmax(agent.brain.predictOne(s))
    #a = random.randint(0, actionCnt-1)
    actions.append(a)
    r = (close2[j+1] - close2[j])*(a - 1)
    R += r
    pnl.append(R)
    j += 1


R
plt.plot(pnl)
plt.legend(['cumulative insample return'])
plt.savefig(r'D:\hot\book\chapter12\fig\insample')

#R
### plot the results
#u = numpy.arange(n)
#loc2 = numpy.asarray(actions)==2
#u2 = u[loc2]
#close22 = close2[loc2]

#loc0 = numpy.asarray(actions)==0
#u0 = u[loc0]
#close20 = close2[loc0]

#plt.plot(u,close2,'-')
#plt.hold(1)
#plt.plot(u2,close22,'or')
#plt.plot(u0,close20,'og')



####################################
## out-of-sample test
####################################
start_day = '20150715'
end_day = '20151207'


nst = (datax['date']==start_day).nonzero()[0][0]
ned = (datax['date']==end_day).nonzero()[0][0]



featurest = np.float64(features[nst:ned,:])

close2t = closex[nst:ned]
Xtt = (featurest - mu1)/std1
Xtt[Xtt>3] = 3
Xtt[Xtt<-3] = -3

j = 0
R = 0
actions = []
pnl = [R]

qval_seq = []

while j<ned - nst: 
    s = Xtt[j,:]
    qval = agent.brain.predictOne(s)
    a = numpy.argmax(qval)
    qval_seq.append(qval)
    actions.append(a)
    if j <ned - nst -1:
        r = (close2t[j+1] - close2t[j])*(a - 1)
        R += r
        pnl.append(R)
    j += 1


R
plt.plot(pnl)
plt.legend(['cumulative out-of-sample return'])
plt.savefig(r'D:\hot\book\chapter12\fig\outsample')


#u = numpy.arange(ned-nst)
#loc2 = numpy.asarray(actions)==2  ## long is red short is greed
#u2 = u[loc2]
#close22 = close2t[loc2]

#loc0 = numpy.asarray(actions)==0
#u0 = u[loc0]
#close20 = close2t[loc0]

#plt.plot(u,close2t,'-')
#plt.hold(1)
#plt.plot(u2,close22,'or')
#plt.plot(u0,close20,'og')

#plt.plot(u,np.array(pnl) + 2000)

#plot(pnl-close2t+3800)



