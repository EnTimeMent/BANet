# This is for our paper "Learning Bodily and Temporal Attention in Protective Movement Behavior Detection" published at ACIIW'19.

# Code Author: Chongyang Wang (My PhD was supported by UCL Overseas Research Scholarship and Graduate Research Scholarship, 
# and partially by the EU Future and Emerging Technologies (FET) Proactive Programme H2020-EU.1.2.2 (Grant agreement 824160; EnTimeMent).

# Revision Date: 05 01 2022

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable nonfatal warnings
import numpy as np
np.random.seed(2)
import scipy.io
import h5py
import keras

from keras.layers import *
from keras.layers.core import *
from keras.models import *
from keras.regularizers import *
from keras import metrics
from keras import backend as K
from keras.backend import sum, mean, max
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping
from keras_self_attention import SeqSelfAttention

def loadata(testname, num):   # loading training and validation data, can be adjusted to your environment.
    data = scipy.io.loadmat('train' + num + '.mat')
    testdata = scipy.io.loadmat(testname + num + '.mat')
    testlabel = scipy.io.loadmat(testname + num + 'label.mat')
    X_train0 = data['data']
    y_train = data['label']
    X_valid0 = testdata['subdata']
    y_valid = testlabel['sublabel']
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
    dimension = 30   # dimension of an input frame, 30 = 13 joint angles + 13 joint energies + 4 sEMG
    BodyNum = 13     # number of body segments (different sensors) to consider

    #Model 1: Temporal Information encoding model (keras Model API)
    singleinput = Input(shape=(180, 2,))
    lstm_units = 8
    LSTM1 = LSTM(lstm_units, return_sequences=True)(singleinput) # Refer to https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM to decide if 'CuDNN' is needed.
    Dropout1 = Dropout(0.5)(LSTM1)
    LSTM2 = LSTM(lstm_units, return_sequences=True)(Dropout1)
    Dropout2 = Dropout(0.5)(LSTM2)
    LSTM3 = LSTM(lstm_units, return_sequences=True)(Dropout2)
    Dropout3 = Dropout(0.5)(LSTM3)
    TemporalProcessmodel = Model(input=singleinput, output=Dropout3)
    # TemporalProcessmodel.summary()

    # Model 2: Main Structure, starting with independent temporal information encoding and attention learning
    inputs = Input(shape=(180, 30,))        # The input data is 180 timesteps by 30 features (13 angles + 13 energies + 4 sEMG)
                                            # The information each body segment included is the angle and energy

    Angle1 = crop(2, 0, 1)(inputs)
    Acc1 = crop(2, 13, 14)(inputs)
    B1 = concatenate([Angle1, Acc1], axis=-1)
    Anglefullout1 = TemporalProcessmodel(B1)
    TemporalAttention1 = Conv1D(1, 1, strides=1)(Anglefullout1) # Temporal Attention Module for each body segment will starts with 1 X 1 Convolution
    TemporalAttention1 = Softmax(axis=-2, name='TemporalAtten1')(TemporalAttention1) # You need Keras >= 2.1.3 to call Softmax as a layer
    AngleAttout1 = multiply([Anglefullout1, TemporalAttention1])
    AngleAttout1 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout1)
    Blast1 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout1)

    Angle2 = crop(2, 1, 2)(inputs)
    Acc2 = crop(2, 14, 15)(inputs)
    B2 = concatenate([Angle2, Acc2], axis=-1)
    Anglefullout2 = TemporalProcessmodel(B2)
    TemporalAttention2 = Conv1D(1, 1, strides=1)(Anglefullout2)
    TemporalAttention2 = Softmax(axis=-2, name='TemporalAtten2')(TemporalAttention2)
    AngleAttout2 = multiply([Anglefullout2, TemporalAttention2])
    AngleAttout2 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout2)
    Blast2 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout2)

    Angle3 = crop(2, 2, 3)(inputs)
    Acc3 = crop(2, 15, 16)(inputs)
    B3 = concatenate([Angle3, Acc3], axis=-1)
    Anglefullout3 = TemporalProcessmodel(B3)
    TemporalAttention3 = Conv1D(1, 1, strides=1)(Anglefullout3)
    TemporalAttention3 = Softmax(axis=-2, name='TemporalAtten3')(TemporalAttention3)
    AngleAttout3 = multiply([Anglefullout3, TemporalAttention3])
    AngleAttout3 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout3)
    Blast3 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout3)

    Angle4 = crop(2, 3, 4)(inputs)
    Acc4 = crop(2, 16, 17)(inputs)
    B4 = concatenate([Angle4, Acc4], axis=-1)
    Anglefullout4 = TemporalProcessmodel(B4)
    TemporalAttention4 = Conv1D(1, 1, strides=1)(Anglefullout4)
    TemporalAttention4 = Softmax(axis=-2, name='TemporalAtten4')(TemporalAttention4)
    AngleAttout4 = multiply([Anglefullout4, TemporalAttention4])
    AngleAttout4 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout4)
    Blast4 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout4)

    Angle5 = crop(2, 4, 5)(inputs)
    Acc5 = crop(2, 17, 18)(inputs)
    B5 = concatenate([Angle5, Acc5], axis=-1)
    Anglefullout5 = TemporalProcessmodel(B5)
    TemporalAttention5 = Conv1D(1, 1, strides=1)(Anglefullout5)
    TemporalAttention5 = Softmax(axis=-2, name='TemporalAtten5')(TemporalAttention5)
    AngleAttout5 = multiply([Anglefullout5, TemporalAttention5])
    AngleAttout5 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout5)
    Blast5 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout5)

    Angle6 = crop(2, 5, 6)(inputs)
    Acc6 = crop(2, 18, 19)(inputs)
    B6 = concatenate([Angle6, Acc6], axis=-1)
    Anglefullout6 = TemporalProcessmodel(B6)
    TemporalAttention6 = Conv1D(1, 1, strides=1)(Anglefullout6)
    TemporalAttention6 = Softmax(axis=-2, name='TemporalAtten6')(TemporalAttention6)
    AngleAttout6 = multiply([Anglefullout6, TemporalAttention6])
    AngleAttout6 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout6)
    Blast6 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout6)

    Angle7 = crop(2, 6, 7)(inputs)
    Acc7 = crop(2, 19, 20)(inputs)
    B7 = concatenate([Angle7, Acc7], axis=-1)
    Anglefullout7 = TemporalProcessmodel(B7)
    TemporalAttention7 = Conv1D(1, 1, strides=1)(Anglefullout7)
    TemporalAttention7 = Softmax(axis=-2, name='TemporalAtten7')(TemporalAttention7)
    AngleAttout7 = multiply([Anglefullout7, TemporalAttention7])
    AngleAttout7 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout7)
    Blast7 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout7)

    Angle8 = crop(2, 7, 8)(inputs)
    Acc8 = crop(2, 20, 21)(inputs)
    B8 = concatenate([Angle8, Acc8], axis=-1)
    Anglefullout8 = TemporalProcessmodel(B8)
    TemporalAttention8 = Conv1D(1, 1, strides=1)(Anglefullout8)
    TemporalAttention8 = Softmax(axis=-2, name='TemporalAtten8')(TemporalAttention8)
    AngleAttout8 = multiply([Anglefullout8, TemporalAttention8])
    AngleAttout8 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout8)
    Blast8 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout8)

    Angle9 = crop(2, 8, 9)(inputs)
    Acc9 = crop(2, 21, 22)(inputs)
    B9 = concatenate([Angle9, Acc9], axis=-1)
    Anglefullout9 = TemporalProcessmodel(B9)
    TemporalAttention9 = Conv1D(1, 1, strides=1)(Anglefullout9)
    TemporalAttention9 = Softmax(axis=-2, name='TemporalAtten9')(TemporalAttention9)
    AngleAttout9 = multiply([Anglefullout9, TemporalAttention9])
    AngleAttout9 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout9)
    Blast9 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout9)

    Angle10 = crop(2, 9, 10)(inputs)
    Acc10 = crop(2, 22, 23)(inputs)
    B10 = concatenate([Angle10, Acc10], axis=-1)
    Anglefullout10 = TemporalProcessmodel(B10)
    TemporalAttention10 = Conv1D(1, 1, strides=1)(Anglefullout10)
    TemporalAttention10 = Softmax(axis=-2, name='TemporalAtten10')(TemporalAttention10)
    AngleAttout10 = multiply([Anglefullout10, TemporalAttention10])
    AngleAttout10 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout10)
    Blast10 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout10)

    Angle11 = crop(2, 10, 11)(inputs)
    Acc11 = crop(2, 23, 24)(inputs)
    B11 = concatenate([Angle11, Acc11], axis=-1)
    Anglefullout11 = TemporalProcessmodel(B11)
    TemporalAttention11 = Conv1D(1, 1, strides=1)(Anglefullout11)
    TemporalAttention11 = Softmax(axis=-2, name='TemporalAtten11')(TemporalAttention11)
    AngleAttout11 = multiply([Anglefullout11, TemporalAttention11])
    AngleAttout11 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout11)
    Blast11 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout11)

    Angle12 = crop(2, 11, 12)(inputs)
    Acc12 = crop(2, 24, 25)(inputs)
    B12 = concatenate([Angle12, Acc12], axis=-1)
    Anglefullout12 = TemporalProcessmodel(B12)
    TemporalAttention12 = Conv1D(1, 1, strides=1)(Anglefullout12)
    TemporalAttention12 = Softmax(axis=-2, name='TemporalAtten12')(TemporalAttention12)
    AngleAttout12 = multiply([Anglefullout12, TemporalAttention12])
    AngleAttout12 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout12)
    Blast12 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout12)

    Angle13 = crop(2, 12, 13)(inputs)
    Acc13 = crop(2, 25, 26)(inputs)
    B13 = concatenate([Angle13, Acc13], axis=-1)
    Anglefullout13 = TemporalProcessmodel(B13)
    TemporalAttention13 = Conv1D(1, 1, strides=1)(Anglefullout13)
    TemporalAttention13 = Softmax(axis=-2, name='TemporalAtten13')(TemporalAttention13)
    AngleAttout13 = multiply([Anglefullout13, TemporalAttention13])
    AngleAttout13 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(AngleAttout13)
    Blast13 = Permute((2, 1), input_shape=(1, lstm_units))(AngleAttout13)


    # Model 3: Feature Concatenation for Bodily Attention Learning
    # The size of the output from each body segment is k X 1, while k is the number of LSTM hidden units
    # In early experiments, we found that it is better to keep the dimension k instead of merging them into one

    DATA = concatenate([Blast1, Blast2, Blast3, Blast4, Blast5, Blast6, Blast7, Blast8,
                        Blast9, Blast10, Blast11, Blast12, Blast13
                        ], axis=2)

    #Bodily Attention Module
    a = Dense(BodyNum, activation='tanh')(DATA)
    a = Dense(BodyNum, activation='softmax', name='bodyattention')(a)
    attentionresult = multiply([DATA, a])
    attentionresult = Flatten()(attentionresult)
    output = Dense(2, activation='softmax')(attentionresult)

    model = Model(input=inputs, output=output)
    # model.summary()

    return model

# Main Implementation Part
if __name__ == '__main__':

    list = np.arange(1, 31, 1)  # Number of subjects, can be adjusted to your environment

    for index in range(len(list)):
        person = str(list[index])

        if list[index]<13:
            X_train0, X_valid0, y_train, y_valid = loadata('C', person) #In my case, the healthy and CP subjects come with different first character, 'C' or 'P'.
        else:
            X_train0, X_valid0, y_train, y_valid = loadata('P', person)

        _, samplenum1, dim1 = y_train.shape # Starting from some versions of Keras, the first dimension of the label is usually '1', which should be deleted.
        _, samplenum2, dim2 = y_valid.shape 
        y_train = np.reshape(y_train, (samplenum1, dim1)) 
        y_valid = np.reshape(y_valid, (samplenum2, dim2)) # Check for yourself if these four sentences are needed or not.

        # callback 1: Save the better result after each epoch,
        checkpointer = keras.callbacks.ModelCheckpoint(filepath='PATH+FileName' + person + '.hdf5',
                                                       monitor='val_binary_accuracy', verbose=1,
                                                       save_best_only=True)
        # callback 2: Stop if Acc=1
        class EarlyStoppingByValAcc(keras.callbacks.Callback):
            def __init__(self, monitor='val_acc', value=1.00000, verbose=0):
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
            EarlyStoppingByValAcc(monitor='val_binary_accuracy', value=1.00000, verbose=1),
            checkpointer
                    ]

        model = build_model()
        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=ada,
                      metrics=['binary_accuracy'])
        ada = keras.optimizers.Adam(lr=0.003)
        H = model.fit(X_train0, y_train,
                      batch_size=40,
                      epochs=80,
                      shuffle=False,
                      callbacks=callbacks,
                      validation_data=(X_valid0, y_valid))

        print('---This is result for %s th subject---' % person)
        model.load_weights('PATH+FileName' + person + '.hdf5')
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import f1_score
        y_pred = np.argmax(model.predict(X_valid0, batch_size=15), axis=1)
        y_true = np.argmax(y_valid, axis=1)
        cf_matrix = confusion_matrix(y_true, y_pred)
        print(cf_matrix)
        class_wise_f1 = np.round(f1_score(y_true, y_pred, average=None) * 100) * 0.01
        print('the mean-f1 score: {:.2f}'.format(np.mean(class_wise_f1)))
