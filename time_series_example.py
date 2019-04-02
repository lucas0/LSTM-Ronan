import pandas as pd
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import csv
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from os import listdir
from keras.preprocessing import sequence
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Flatten

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

path = 'MovementAAL/dataset/MovementAAL_RSS_'
sequences = list()
for i in range(1,315):
    file_path = path + str(i) + '.csv'
    print(file_path)
    df = pd.read_csv(file_path, header=0)
    values = df.values
    sequences.append(values)

targets = pd.read_csv('MovementAAL/dataset/MovementAAL_target.csv')
targets = targets.values[:,1]

groups = pd.read_csv('MovementAAL/groups/MovementAAL_DatasetGroup.csv', header=0)
groups = groups.values[:,1]

len_sequences = []
for one_seq in sequences:
    len_sequences.append(len(one_seq))
pd.Series(len_sequences).describe()

to_pad = 129
new_seq = []
for one_seq in sequences:
    len_one_seq = len(one_seq)
    last_val = one_seq[-1]
    n = to_pad - len_one_seq
   
    to_concat = np.repeat(one_seq[-1], n).reshape(4, n).transpose()
    new_one_seq = np.concatenate([one_seq, to_concat])
    new_seq.append(new_one_seq)
final_seq = np.stack(new_seq)

#truncate the sequence to length 60
from keras.preprocessing import sequence
seq_len = 60
final_seq=sequence.pad_sequences(final_seq, maxlen=seq_len, padding='post', dtype='float', truncating='post')

train = [final_seq[i] for i in range(len(groups)) if (groups[i]==2)]
validation = [final_seq[i] for i in range(len(groups)) if groups[i]==1]
test = [final_seq[i] for i in range(len(groups)) if groups[i]==3]

train_target = [targets[i] for i in range(len(groups)) if (groups[i]==2)]
validation_target = [targets[i] for i in range(len(groups)) if groups[i]==1]
test_target = [targets[i] for i in range(len(groups)) if groups[i]==3]

train = np.array(train)
validation = np.array(validation)
test = np.array(test)

train_target = np.array(train_target)
train_target = (train_target+1)/2

validation_target = np.array(validation_target)
validation_target = (validation_target+1)/2

test_target = np.array(test_target)
test_target = (test_target+1)/2

'''
split_ratio = 0.8
"""split the data training, test and validate data ratio train:test 80:20, train:validate 80:20"""
trainingval = sequences[0:int(np.ceil(len(sequences)*split_ratio))]

test = sequences[int(np.ceil(len(sequences)*split_ratio)):len(sequences)]
train = trainingval[0:int(np.ceil(len(trainingval)*split_ratio))]
validation = trainingval[int(np.ceil(len(trainingval)*split_ratio)):len(trainingval)]

"""split the target training, test and validate data ratio train:test 80:20, train:validate 80:20"""
trainingval_target = targets[0:int(np.ceil(len(targets)*split_ratio))]

test_target = targets[int(np.ceil(len(targets)*split_ratio)):len(targets)]
train_target = trainingval_target[0:int(np.ceil(len(trainingval_target)*split_ratio))]
validation_target = trainingval_target[int(np.ceil(len(trainingval_target)*split_ratio)):len(trainingval_target)]

train = np.array(train)
validation = np.array(validation)
test = np.array(test)

train_target = np.array(train_target)
train_target = train_target.reshape(4,1,1)
#train_target = ((len(train_target)) + 1)/2

validation_target = np.array(validation_target)
#validation_target = ((len(validation_target)) + 1)/2

test_target = np.array(test_target)
#test_target = ((len(test_target)) + 1)/2
'''

"""create LSTM model length of each frame"""
model = Sequential()
model.add(LSTM(256, input_shape=(60, 4)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.summary()
#print(train_target)
#print(train.shape)
#print(train_target.shape)

adam = Adam(lr=0.001)
chk = ModelCheckpoint('best_model.pkl', monitor='val_acc', save_best_only=True, mode='max', verbose=1)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.fit(train, train_target, epochs=200, batch_size=128, callbacks=[chk], validation_data=(validation,validation_target))

#loading the model and checking accuracy on the test data
model = load_model('best_model.pkl')

from sklearn.metrics import accuracy_score
test_preds = model.predict_classes(test)
accuracy_score(test_target, test_preds)

