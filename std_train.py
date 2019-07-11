import pickle
import time
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda
from keras.models import Sequential, Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from spatial_transformer import SpatialTransformer
import numpy as np
from keras.preprocessing import image
from function import *


def build_model():
    model = Sequential()
    model.add(Conv2D(filters=16,kernel_size=7,padding='same',activation='relu',input_shape=train_tensors.shape[1:]))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=32, kernel_size=5, padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=64,kernel_size=5,padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=128,kernel_size=5,strides=2,padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    return model


train_filename = "data_preprocessed/train_data_sample_rgb2.p"
(train_labels, train_data, train_tensors) = pickle.load(open(train_filename, mode='rb'))

valid_filename = "data_preprocessed/valid_data_sample_rgb2.p"
(valid_labels, valid_data, valid_tensors) = pickle.load(open(valid_filename, mode='rb'))

test_filename = "data_preprocessed/test_data_sample_rgb2.p"
(test_labels, test_data, test_tensors) = pickle.load(open(test_filename, mode='rb'))

model = build_model()
model.compile(optimizer='sgd', loss='mean_squared_error',
              metrics=[precision_threshold(threshold=0.5),
                       recall_threshold(threshold=0.5),
                       fbeta_score_threshold(beta=0.5, threshold=0.5),
                       'accuracy'])

epochs = 100
batch_size = 32

earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=16, verbose=1, mode='auto')
log = CSVLogger('saved_models/log_bCNN_rgb2.csv')
checkpointer = ModelCheckpoint(filepath='saved_models/bCNN2.best.from_scratch.hdf5',
                               verbose=1, save_best_only=True)

start = time.time()

model.fit(train_tensors, train_labels,
          validation_data=(valid_tensors, valid_labels),
          epochs=epochs, batch_size=batch_size, callbacks=[checkpointer, log, earlystop], verbose=1)

# Show total training time
print("training time: %.2f minutes" % ((time.time() - start) / 60))

model.load_weights('saved_models/bCNN2.best.from_scratch.hdf5')
prediction = model.predict(test_tensors)

threshold = 0.5
beta = 0.5

pre = K.eval(precision_threshold(threshold=threshold)(K.variable(value=test_labels),
                                                      K.variable(value=prediction)))
rec = K.eval(recall_threshold(threshold=threshold)(K.variable(value=test_labels),
                                                   K.variable(value=prediction)))
fsc = K.eval(fbeta_score_threshold(beta=beta, threshold=threshold)(K.variable(value=test_labels),
                                                                   K.variable(value=prediction)))

print("Precision: %f %%\nRecall: %f %%\nFscore: %f %%" % (pre, rec, fsc))

print(K.eval(binary_accuracy(K.variable(value=test_labels), K.variable(value=prediction))))

threshold = 0.4
beta = 0.5

pre = K.eval(precision_threshold(threshold=threshold)(K.variable(value=test_labels),
                                                      K.variable(value=prediction)))
rec = K.eval(recall_threshold(threshold=threshold)(K.variable(value=test_labels),
                                                   K.variable(value=prediction)))
fsc = K.eval(fbeta_score_threshold(beta=beta, threshold=threshold)(K.variable(value=test_labels),
                                                                   K.variable(value=prediction)))

print("Precision: %f %%\nRecall: %f %%\nFscore: %f %%" % (pre, rec, fsc))

threshold = 0.6
beta = 0.5

pre = K.eval(precision_threshold(threshold=threshold)(K.variable(value=test_labels),
                                                      K.variable(value=prediction)))
rec = K.eval(recall_threshold(threshold=threshold)(K.variable(value=test_labels),
                                                   K.variable(value=prediction)))
fsc = K.eval(fbeta_score_threshold(beta=beta, threshold=threshold)(K.variable(value=test_labels),
                                                                   K.variable(value=prediction)))

print("Precision: %f %%\nRecall: %f %%\nFscore: %f %%" % (pre, rec, fsc))
