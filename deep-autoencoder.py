'''
filename: deep-autoencoder
creater:lijhong
time:8-27-2017
this python to implement deep autoencoder function
coding: utf-8
'''
import pandas as pd
import numpy as np
import theano 
from keras.layers import Dense,Activation,Input
from keras.model import Sequential,Model

go = pd.read_csv('.csv')
go_id = go['Gene_ID']
go = go.drop(['Gene_ID'],axis=1)

inputDims = go.values.shape[1]
EncoderDims = 100
AutoEncoder = Sequential()
AutoEncoder.add(Dense(input_dim=inputDims,output_dim
    =EncoderDims,activation='relu'))
AutoEncoder.add(Dense(input_dim=EncoderDims,output_dim
    =inputDims,activation='relu'))
AutoEncoder.compile(optimizer='adadelta',loss='binary_crossentropy')
AutoEncoder.fit(go.values,go.values,batch_size=32,nb_epoch=50,shuffle=True)
#get hidden vector
get_feature = theano.function([AutoEncoder.layers[0].Input],AutoEncoder.layers[0].output,allow_input_downcast=False)
new_go = get_feature(go)


