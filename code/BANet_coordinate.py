# This is for our paper "Learning Bodily and Temporal Attention in Protective Movement Behavior Detection" published at ACIIW'19.

# Code Author: Chongyang Wang (My PhD was supported by UCL Overseas Research Scholarship and Graduate Research Scholarship, 
# and partially by the EU Future and Emerging Technologies (FET) Proactive Programme H2020-EU.1.2.2 (Grant agreement 824160; EnTimeMent).

# Revision Date: 05 01 2022

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
np.random.seed(2)
import tensorflow as tf
import scipy.io
import h5py
import keras
import hdf5storage
import xlwt as xw

from keras.layers import *
from keras.layers.core import *
from keras.models import *
from keras.regularizers import *
from keras.optimizers import *
from keras.losses import *
from keras import metrics
from keras import backend as K
from keras.backend import sum, mean, max
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping
from matplotlib import pyplot

def loadata(testname, num):   # loading training and validation data, can be adjusted to your environment.
    traindata = hdf5storage.loadmat('Augtrain' + num + '.mat')
    X_train0 = traindata['data']
    y_train = traindata['label']
    testdata = scipy.io.loadmat(testname + num + '.mat')
    testdata1 = scipy.io.loadmat(testname + num + 'label'+'.mat')
    X_valid0 = testdata['subdata']
    y_valid = testdata1['sublabel']
    num_classes = 2  # classes
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes)
    return X_train0, X_valid0, y_train, y_valid

def crop(dimension, start, end):
    # Thanks to marc-moreaux on Github page:https://github.com/keras-team/keras/issues/890 who created this beautiful and sufficient function: )
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)

def build_model():
    timestep = 180   # length of an input frame
    dimension = 66   # dimension of an input frame, 66 = 22 joints by 3 xyz coordinates, the 4 coordinates of the foot are removed.
    BodyNum = 22     # number of body segments (different sensors) to consider

    #Model 1: Temporal Information encoding model for BANet (keras Model API)
    singleinput = Input(shape=(180, 3,))
    lstm_units = 8
    LSTM1 = LSTM(lstm_units, return_sequences=True)(singleinput) # Refer to https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM to decide if 'CuDNN' is needed.
    Dropout1 = Dropout(0.5)(LSTM1)
    LSTM2 = LSTM(lstm_units, return_sequences=True)(Dropout1)
    Dropout2 = Dropout(0.5)(LSTM2)
    LSTM3 = LSTM(lstm_units, return_sequences=True)(Dropout2)
    Dropout3 = Dropout(0.5)(LSTM3)
    TemporalProcessmodel = Model(inputs=[singleinput], outputs=[Dropout3])
    # TemporalProcessmodel.summary()

    # Model 2: Main Structure, starting with independent temporal information encoding and attention learning
    inputs = Input(shape=(180, 66,))      # The input data is 180 timesteps by 66 features (22 joints by 3 xyz coordinates)
                                          # The information each body segment provides is the coordinates of each joint

    x1 = crop(2, 0, 1)(inputs)
    y1 = crop(2, 22, 23)(inputs)
    z1 = crop(2, 44,45)(inputs)
    B1 = concatenate([x1, y1, z1], axis=-1)
    Anglefullout1 = TemporalProcessmodel(B1)
    TemporalAttention1 = Conv1D(1, 1, strides=1)(Anglefullout1) # Temporal Attention Module for each body segment will starts with 1 X 1 Convolution
    TemporalAttention1 = Softmax(axis=-2, name='TemporalAtten1')(TemporalAttention1) # You need Keras >= 2.1.3 to call Softmax as a layer
    AngleAttout1 = multiply([Anglefullout1, TemporalAttention1])
    AngleAttout1 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout1)
    Blast1 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout1)

    x2 = crop(2, 1, 2)(inputs)
    y2 = crop(2, 23, 24)(inputs)
    z2 = crop(2, 45, 46)(inputs)
    B2 = concatenate([x2, y2, z2], axis=-1)
    Anglefullout2 = TemporalProcessmodel(B2)
    TemporalAttention2 = Conv1D(1, 1, strides=1)(Anglefullout2)
    TemporalAttention2 = Softmax(axis=-2, name='TemporalAtten2')(TemporalAttention2)
    AngleAttout2 = multiply([Anglefullout2, TemporalAttention2])
    AngleAttout2 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout2)
    Blast2 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout2)

    x3 = crop(2, 2, 3)(inputs)
    y3 = crop(2, 24, 25)(inputs)
    z3 = crop(2, 46, 47)(inputs)
    B3 = concatenate([x3, y3, z3], axis=-1)
    Anglefullout3 = TemporalProcessmodel(B3)
    TemporalAttention3 = Conv1D(1, 1, strides=1)(Anglefullout3)
    TemporalAttention3 = Softmax(axis=-2, name='TemporalAtten3')(TemporalAttention3)
    AngleAttout3 = multiply([Anglefullout3, TemporalAttention3])
    AngleAttout3 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout3)
    Blast3 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout3)

    x4 = crop(2, 3, 4)(inputs)
    y4 = crop(2, 25, 26)(inputs)
    z4 = crop(2, 47, 48)(inputs)
    B4 = concatenate([x4, y4, z4], axis=-1)
    Anglefullout4 = TemporalProcessmodel(B4)
    TemporalAttention4 = Conv1D(1, 1, strides=1)(Anglefullout4)
    TemporalAttention4 = Softmax(axis=-2, name='TemporalAtten4')(TemporalAttention4)
    AngleAttout4 = multiply([Anglefullout4, TemporalAttention4])
    AngleAttout4 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout4)
    Blast4 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout4)

    x5 = crop(2, 4, 5)(inputs)
    y5 = crop(2, 26, 27)(inputs)
    z5 = crop(2, 48, 49)(inputs)
    B5 = concatenate([x5, y5, z5], axis=-1)
    Anglefullout5 = TemporalProcessmodel(B5)
    TemporalAttention5 = Conv1D(1, 1, strides=1)(Anglefullout5)
    TemporalAttention5 = Softmax(axis=-2, name='TemporalAtten5')(TemporalAttention5)
    AngleAttout5 = multiply([Anglefullout5, TemporalAttention5])
    AngleAttout5 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout5)
    Blast5 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout5)

    x6 = crop(2, 5, 6)(inputs)
    y6 = crop(2, 27, 28)(inputs)
    z6 = crop(2, 49, 50)(inputs)
    B6 = concatenate([x6, y6, z6], axis=-1)
    Anglefullout6 = TemporalProcessmodel(B6)
    TemporalAttention6 = Conv1D(1, 1, strides=1)(Anglefullout6)
    TemporalAttention6 = Softmax(axis=-2, name='TemporalAtten6')(TemporalAttention6)
    AngleAttout6 = multiply([Anglefullout6, TemporalAttention6])
    AngleAttout6 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout6)
    Blast6 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout6)

    x7 = crop(2, 6, 7)(inputs)
    y7 = crop(2, 28, 29)(inputs)
    z7 = crop(2, 50, 51)(inputs)
    B7 = concatenate([x7, y7, z7], axis=-1)
    Anglefullout7 = TemporalProcessmodel(B7)
    TemporalAttention7 = Conv1D(1, 1, strides=1)(Anglefullout7)
    TemporalAttention7 = Softmax(axis=-2, name='TemporalAtten7')(TemporalAttention7)
    AngleAttout7 = multiply([Anglefullout7, TemporalAttention7])
    AngleAttout7 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout7)
    Blast7 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout7)

    x8 = crop(2, 7, 8)(inputs)
    y8 = crop(2, 29, 30)(inputs)
    z8 = crop(2, 51, 52)(inputs)
    B8 = concatenate([x8, y8, z8], axis=-1)
    Anglefullout8 = TemporalProcessmodel(B8)
    TemporalAttention8 = Conv1D(1, 1, strides=1)(Anglefullout8)
    TemporalAttention8 = Softmax(axis=-2, name='TemporalAtten8')(TemporalAttention8)
    AngleAttout8 = multiply([Anglefullout8, TemporalAttention8])
    AngleAttout8 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout8)
    Blast8 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout8)

    x9 = crop(2, 8, 9)(inputs)
    y9 = crop(2, 30, 31)(inputs)
    z9 = crop(2, 52, 53)(inputs)
    B9 = concatenate([x9, y9, z9], axis=-1)
    Anglefullout9 = TemporalProcessmodel(B9)
    TemporalAttention9 = Conv1D(1, 1, strides=1)(Anglefullout9)
    TemporalAttention9 = Softmax(axis=-2, name='TemporalAtten9')(TemporalAttention9)
    AngleAttout9 = multiply([Anglefullout9, TemporalAttention9])
    AngleAttout9 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout9)
    Blast9 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout9)

    x10 = crop(2, 9, 10)(inputs)
    y10 = crop(2, 31, 32)(inputs)
    z10 = crop(2, 53, 54)(inputs)
    B10 = concatenate([x10, y10, z10], axis=-1)
    Anglefullout10 = TemporalProcessmodel(B10)
    TemporalAttention10 = Conv1D(1, 1, strides=1)(Anglefullout10)
    TemporalAttention10 = Softmax(axis=-2, name='TemporalAtten10')(TemporalAttention10)
    AngleAttout10 = multiply([Anglefullout10, TemporalAttention10])
    AngleAttout10 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout10)
    Blast10 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout10)

    x11 = crop(2, 10, 11)(inputs)
    y11 = crop(2, 32, 33)(inputs)
    z11 = crop(2, 54, 55)(inputs)
    B11 = concatenate([x11, y11, z11], axis=-1)
    Anglefullout11 = TemporalProcessmodel(B11)
    TemporalAttention11 = Conv1D(1, 1, strides=1)(Anglefullout11)
    TemporalAttention11 = Softmax(axis=-2, name='TemporalAtten11')(TemporalAttention11)
    AngleAttout11 = multiply([Anglefullout11, TemporalAttention11])
    AngleAttout11 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout11)
    Blast11 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout11)

    x12 = crop(2, 11, 12)(inputs)
    y12 = crop(2, 33, 34)(inputs)
    z12 = crop(2, 55, 56)(inputs)
    B12 = concatenate([x12, y12, z12], axis=-1)
    Anglefullout12 = TemporalProcessmodel(B12)
    TemporalAttention12 = Conv1D(1, 1, strides=1)(Anglefullout12)
    TemporalAttention12 = Softmax(axis=-2, name='TemporalAtten12')(TemporalAttention12)
    AngleAttout12 = multiply([Anglefullout12, TemporalAttention12])
    AngleAttout12 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout12)
    Blast12 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout12)

    x13 = crop(2, 12, 13)(inputs)
    y13 = crop(2, 34, 35)(inputs)
    z13 = crop(2, 56, 57)(inputs)
    B13 = concatenate([x13, y13, z13], axis=-1)
    Anglefullout13 = TemporalProcessmodel(B13)
    TemporalAttention13 = Conv1D(1, 1, strides=1)(Anglefullout13)
    TemporalAttention13 = Softmax(axis=-2, name='TemporalAtten13')(TemporalAttention13)
    AngleAttout13 = multiply([Anglefullout13, TemporalAttention13])
    AngleAttout13 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout13)
    Blast13 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout13)

    x14 = crop(2, 13, 14)(inputs)
    y14 = crop(2, 35, 36)(inputs)
    z14 = crop(2, 57, 58)(inputs)
    B14 = concatenate([x14, y14, z14], axis=-1)
    Anglefullout14 = TemporalProcessmodel(B14)
    TemporalAttention14 = Conv1D(1, 1, strides=1)(Anglefullout14)
    TemporalAttention14 = Softmax(axis=-2, name='TemporalAtten14')(TemporalAttention14)
    AngleAttout14 = multiply([Anglefullout14, TemporalAttention14])
    AngleAttout14 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout14)
    Blast14 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout14)

    x15 = crop(2, 14, 15)(inputs)
    y15 = crop(2, 36, 37)(inputs)
    z15 = crop(2, 58, 59)(inputs)
    B15 = concatenate([x15, y15, z15], axis=-1)
    Anglefullout15 = TemporalProcessmodel(B15)
    TemporalAttention15 = Conv1D(1, 1, strides=1)(Anglefullout15)
    TemporalAttention15 = Softmax(axis=-2, name='TemporalAtten15')(TemporalAttention15)
    AngleAttout15 = multiply([Anglefullout15, TemporalAttention15])
    AngleAttout15 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout15)
    Blast15 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout15)

    x16 = crop(2, 15, 16)(inputs)
    y16 = crop(2, 37, 38)(inputs)
    z16 = crop(2, 59, 60)(inputs)
    B16 = concatenate([x16, y16, z16], axis=-1)
    Anglefullout16 = TemporalProcessmodel(B16)
    TemporalAttention16 = Conv1D(1, 1, strides=1)(Anglefullout16)
    TemporalAttention16 = Softmax(axis=-2, name='TemporalAtten16')(TemporalAttention16)
    AngleAttout16 = multiply([Anglefullout16, TemporalAttention16])
    AngleAttout16 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout16)
    Blast16 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout16)

    x17 = crop(2, 16, 17)(inputs)
    y17 = crop(2, 38, 39)(inputs)
    z17 = crop(2, 60, 61)(inputs)
    B17 = concatenate([x17, y17, z17], axis=-1)
    Anglefullout17 = TemporalProcessmodel(B17)
    TemporalAttention17 = Conv1D(1, 1, strides=1)(Anglefullout17)
    TemporalAttention17 = Softmax(axis=-2, name='TemporalAtten17')(TemporalAttention17)
    AngleAttout17 = multiply([Anglefullout17, TemporalAttention17])
    AngleAttout17 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout17)
    Blast17 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout17)

    x18 = crop(2, 17, 18)(inputs)
    y18 = crop(2, 39, 40)(inputs)
    z18 = crop(2, 61, 62)(inputs)
    B18 = concatenate([x18, y18, z18], axis=-1)
    Anglefullout18 = TemporalProcessmodel(B18)
    TemporalAttention18 = Conv1D(1, 1, strides=1)(Anglefullout18)
    TemporalAttention18 = Softmax(axis=-2, name='TemporalAtten18')(TemporalAttention18)
    AngleAttout18 = multiply([Anglefullout18, TemporalAttention18])
    AngleAttout18 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout18)
    Blast18 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout18)

    x19 = crop(2, 18, 19)(inputs)
    y19 = crop(2, 40, 41)(inputs)
    z19 = crop(2, 62, 63)(inputs)
    B19 = concatenate([x19, y19, z19], axis=-1)
    Anglefullout19 = TemporalProcessmodel(B19)
    TemporalAttention19 = Conv1D(1, 1, strides=1)(Anglefullout19)
    TemporalAttention19 = Softmax(axis=-2, name='TemporalAtten19')(TemporalAttention19)
    AngleAttout19 = multiply([Anglefullout19, TemporalAttention19])
    AngleAttout19 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout19)
    Blast19 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout19)

    x20 = crop(2, 19, 20)(inputs)
    y20 = crop(2, 41, 42)(inputs)
    z20 = crop(2, 63, 64)(inputs)
    B20 = concatenate([x20, y20, z20], axis=-1)
    Anglefullout20 = TemporalProcessmodel(B20)
    TemporalAttention20 = Conv1D(1, 1, strides=1)(Anglefullout20)
    TemporalAttention20 = Softmax(axis=-2, name='TemporalAtten20')(TemporalAttention20)
    AngleAttout20 = multiply([Anglefullout20, TemporalAttention20])
    AngleAttout20 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout20)
    Blast20 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout20)

    x21 = crop(2, 20, 21)(inputs)
    y21 = crop(2, 42, 43)(inputs)
    z21 = crop(2, 64, 65)(inputs)
    B21 = concatenate([x21, y21, z21], axis=-1)
    Anglefullout21 = TemporalProcessmodel(B21)
    TemporalAttention21 = Conv1D(1, 1, strides=1)(Anglefullout21)
    TemporalAttention21 = Softmax(axis=-2, name='TemporalAtten21')(TemporalAttention21)
    AngleAttout21 = multiply([Anglefullout21, TemporalAttention21])
    AngleAttout21 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout21)
    Blast21 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout21)

    x22 = crop(2, 21, 22)(inputs)
    y22 = crop(2, 43, 44)(inputs)
    z22 = crop(2, 65, 66)(inputs)
    B22 = concatenate([x22, y22, z22], axis=-1)
    Anglefullout22 = TemporalProcessmodel(B22)
    TemporalAttention22 = Conv1D(1, 1, strides=1)(Anglefullout22)
    TemporalAttention22 = Softmax(axis=-2, name='TemporalAtten22')(TemporalAttention22)
    AngleAttout22 = multiply([Anglefullout22, TemporalAttention22])
    AngleAttout22 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout22)
    Blast22 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout22)

    # Model 3: Feature Concatenation for Bodily Attention Learning
    # The size of the output from each body segment is k X 1, while k is the number of LSTM hidden units
    # In early experiments, we found that it is better to keep the dimension k instead of merging them into one

    DATA = concatenate([Blast1, Blast2, Blast3, Blast4, Blast5, Blast6, Blast7, Blast8,
                        Blast9, Blast10, Blast11, Blast12, Blast13, Blast14, Blast15, Blast16,
                        Blast17, Blast18, Blast19, Blast20, Blast21, Blast22
                        ], axis=2)

    # Bodily Attention Module
    a = Dense(BodyNum, activation='tanh')(DATA)
    a = Dense(BodyNum, activation='softmax', name='bodyattention')(a)
    attentionresult = multiply([DATA, a])
    attentionresult = Flatten()(attentionresult)

    output = Dense(2, activation='softmax',name='mainoutput')(attentionresult)
    model = Model(inputs=[inputs], outputs=[output])
    # model.summary()

    return model

# Main Implementation Part
if __name__ == '__main__':

    list = np.arange(13, 31, 1)  # Number of subjects, can be adjusted to your environment
    typelist = np.arange(1, 7, 1) # Number of movement types, can be adjusted to your environment
    movement = ['Bend', 'Olg', 'Sits', 'Stsi', 'Rf']
    wb = xw.Workbook(encoding='ascii')
    ws = wb.add_sheet('AEBANet')
    ws.write(1, 0, label=u'Last Accuracy')
    ws.write(2, 0, label=u'F1-score')
    ws.write(3, 0, label=u'Confusion Matrix')
    ws.write(6, 0, label=u'Best Accuracy')
    ws.write(7, 0, label=u'F1-score')
    ws.write(8, 0, label=u'Confusion Matrix')
    a1 = np.zeros((2, 2))

    for index in range(len(list)):
        person = str(list[index])
        num = list[index]

        if list[index]<13:
            X_train0, X_valid0, y_train, y_train1, y_valid, y_valid1 = loadata('C', person, movement[1]) #In my case, the healthy and CP subjects come with different first character, 'C' or 'P'.
        else:
            X_train0, X_valid0, y_train, y_train1, y_valid, y_valid1 = loadata('P', person, movement[1])

        _, samplenum1, dim1 = y_train.shape # Starting from some versions of Keras, the first dimension of the label is usually '1', which should be deleted.
        _, samplenum3, dim3 = y_valid.shape 
        y_train = np.reshape(y_train, (samplenum1, dim1)) 
        y_valid = np.reshape(y_valid, (samplenum3, dim3)) # Check for yourself if these four sentences are needed or not.

        metricmonitor='val_categorical_accuracy'

        # callback 1: Save the better result after each epoch,
        checkpointer = keras.callbacks.ModelCheckpoint(filepath='BanetBaseCoordiBest' + person + '.hdf5',
                                                       monitor=metricmonitor, verbose=1,
                                                       save_best_only=True)

        # callback 2: Stop if Acc=1
        class EarlyStoppingByValAcc(keras.callbacks.Callback):
            def __init__(self, monitor, value=1.00000, verbose=0):
                super(keras.callbacks.Callback, self).__init__()
                self.monitor = monitor
                self.value = value
                self.verbose = verbose
            def on_epoch_end(self, epoch, logs={}):
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)
                if current == self.value:
                    if self.verbose > 0:
                        print("Epoch %05d: early stopping THR" % epoch)
                        self.model.stop_training = True

        callbacks = [
            EarlyStoppingByValAcc(monitor=metricmonitor, value=1.00000, verbose=1),
            checkpointer
                    ]

        model = build_model()
        model.reset_states()

        # ada = SGD(lr=5e3, momentum=0.9, decay=5e4)
        ada = Adam(lr=0.0005)

        model.compile(optimizer=ada,
                      loss=['categorical_crossentropy'],
                      metrics=['categorical_accuracy'])

        H = model.fit(X_train0,
                      y_train,
                      batch_size=500,
                      epochs=50,
                      shuffle=False,
                      callbacks=callbacks,
                      validation_data=(X_valid0, y_valid))
        
        model.save_weights('BanetBaseCoordiLast' + person + '.hdf5')

        print('---This is result for %s th subject---' % person)
        from sklearn.metrics import *

        persontype='P'

        model.load_weights('BanetBaseCoordiLast' + person + '.hdf5')
        y_predraw = model.predict(X_valid0, batch_size=500)
        y_pred = np.argmax(y_predraw, axis=1)
        y_true = np.argmax(y_valid, axis=1)
        cf_matrix = confusion_matrix(y_true, y_pred)
        print('The last DoP value is')
        print(cf_matrix)
        acc = accuracy_score(y_true, y_pred)
        class_wise_f1 = np.round(f1_score(y_true, y_pred, average=None) * 100) * 0.01
        print('the mean-f1 score: {:.2f}'.format(np.mean(class_wise_f1)))
        ws.write(0, num, label=persontype + person)
        ws.write(1, num, label=u'{:.4f}'.format(acc))
        ws.write(2, num, label=u'{:.4f}'.format(np.mean(class_wise_f1)))
        ws.write(3, num, label=str(cf_matrix[0]))
        ws.write(4, num, label=str(cf_matrix[1]))

        print('---Using The Best Val Accuracy Model of %s th subject---' % person)
        model.load_weights('BanetBaseCoordiBest' + person + '.hdf5')
        y_pred2raw = model.predict(X_valid0, batch_size=500)
        y_pred2 = np.argmax(y_pred2raw, axis=1)
        y_true = np.argmax(y_valid, axis=1)
        cf_matrix12 = confusion_matrix(y_true, y_pred2)
        print('The Best value is')
        print(cf_matrix12)
        acc = accuracy_score(y_true, y_pred2)
        class_wise_f12 = np.round(f1_score(y_true, y_pred2, average=None) * 100) * 0.01
        print('the accuracy: {:.4f}'.format(acc))
        print('the mean-f1 score: {:.2f}'.format(np.mean(class_wise_f12)))
        ws.write(10, num, label=u'{:.4f}'.format(acc))
        ws.write(11, num, label=u'{:.4f}'.format(np.mean(class_wise_f12)))
        ws.write(12, num, label=str(cf_matrix12[0]))
        ws.write(13, num, label=str(cf_matrix12[1]))
        del X_train0
        del X_valid0
        
        if np.mean(class_wise_f1) > np.mean(class_wise_f12):
            # scipy.io.savemat('P'+person+'BANetCoordiresults', {'DOPesults':y_predraw})
            a1 = a1 + cf_matrix
        else:
            # scipy.io.savemat('P'+person+'BANetCoordiresults', {'DOPesults': y_pred2raw})
            a1 = a1 + cf_matrix12

    print('---The Total Confusion Matrix is---')
    print(a1)
    # ws.write(20, 13, label=str(a1[0]))
    # ws.write(21, 13, label=str(a1[1]))
    # wb.save('BanetBaseCoordinate-1.xls')