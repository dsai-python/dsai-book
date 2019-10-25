##############################
## Case Study (section 11.1.7)
##############################
# NN
# 使用第五章案例分析 5.2.3 的训练样本和预测样本 X_train y_train X_test y_test
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop,Adam 

model = Sequential([
    Dense(8, input_dim=5),
    Activation('relu'),
    Dense(8),
    Activation('relu'),
    Dense(8),
    Activation('relu'),
    Dense(1),
    Activation('sigmoid'),])
model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              #loss = 'kullback_leibler_divergence',
              metrics=['binary_accuracy'])
model.fit(X_train, y_train, epochs=200, batch_size=200)

# performance in test sample
y_pred = model.predict(X_test)
np.corrcoef([y_test,y_pred.ravel()]) # Information Coefficient, IC


holding_matrix = np.zeros((n1-p,300))
for j in range(n1-p):
    prob = model.predict_proba(test[j:j+5,:].T).ravel()   
    long_position = prob.argsort()[-10:]
    short_position = prob.argsort()[:10]
    holding_matrix[j,long_position] = 0.05
    holding_matrix[j,short_position] = -0.05

tmp_ret_nn = np.sum(holding_matrix*test[5:],axis = 1)
portfolio_ret_nn = np.append(0,tmp_ret_nn)
plt.plot(np.cumprod(1+portfolio_ret_nn))
plt.legend(['Cumulative Return via NN'])
plt.savefig(r'D:\hot\book\chapter11\fig\stockret-nn')






##############################
## Case Study (section 11.3.2)
##############################
# autoencoder

## digital 3 image data

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
path = r'D:\hot\book\data\zip.train\zip.train'
data = np.loadtxt(path) # 7291*257, first column is number

id3 = data[:,0]==3
data3 = data[id3,1:]     # 658*256 矩阵

import numpy as np
import numpy.linalg as la
import pandas as pd


from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from sklearn.model_selection import train_test_split


encoding_dim = 100
input_img = Input(shape=(256, ))
encoded = Dense(encoding_dim, activation = 'sigmoid', 
                activity_regularizer = regularizers.l1(10e-8))(input_img)
decoded = Dense(256, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')


#data3 = data3 - np.mean(data3,axis = 0)

x_train,x_test=train_test_split(data3,test_size=0.2) #
autoencoder.fit(x_train, x_train, epochs=1000, batch_size=256, shuffle=True, 
                validation_data=(x_test, x_test),verbose=2)

encoder = Model(input_img, encoded)
encoded_imgs = encoder.predict(x_train)

j = 3 # 1,2,3...
id = encoded_imgs[:,j-1].argsort()
for i in range(5):
    plt.subplot(2,5,i+1)
    plt.imshow(x_train[id[-i-1]].reshape(16, 16))
for i in range(5):
    plt.subplot(2,5,5+i+1)
    plt.imshow(x_train[id[i]].reshape(16, 16))

plt.savefig(r'D:\hot\book\chapter11\fig\num3-ae4')






#encoded_input = Input(shape=(encoding_dim,))
#decoder_layer = autoencoder.layers[-1]
#decoder = Model(encoded_input, decoder_layer(encoded_input))
#encoded_imgs = encoder.predict(x_test)
#decoded_imgs = decoder.predict(encoded_imgs)

#n = 5
#for i in range(n):
#    # display original
#    ax = plt.subplot(2, n, i + 1)
#    plt.imshow(x_test[i].reshape(16, 16))
#    ax.set_axis_off()
#    # display reconstruction
#    ax = plt.subplot(2, n, i + n + 1)
#    plt.imshow(decoded_imgs[i].reshape(16, 16))
#    ax.set_axis_off()
#plt.show()
#encoded_imgs = encoder.predict(x_train)
#decoded_imgs = decoder.predict(encoded_imgs)
#n = 10 
#plt.figure(figsize=(10, 2), dpi=100)
#for i in range(n):
#    # display original
#    ax = plt.subplot(2, n, i + 1)
#    plt.imshow(x_train[i].reshape(16, 16))
#    ax.set_axis_off()
#    # display reconstruction
#    ax = plt.subplot(2, n, i + n + 1)
#    plt.imshow(decoded_imgs[i].reshape(16, 16))
#    ax.set_axis_off()
#plt.show()



















