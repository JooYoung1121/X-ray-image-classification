import os
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras import regularizers, initializers, optimizers, applications
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from keras.layers import Input, merge, concatenate
from spatial_transformer import SpatialTransformer
from function import *
import tensorflow as tf
import pickle
import time

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def locnet():
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((64, 6), dtype='float32')
    weights = [W, b.flatten()]
    locnet = Sequential()

    locnet.add(Conv2D(16, (7, 7), padding='valid', input_shape=train_tensors.shape[1:]))
    locnet.add(MaxPooling2D(pool_size=(2, 2)))
    locnet.add(Conv2D(32, (5, 5), padding='valid'))
    locnet.add(MaxPooling2D(pool_size=(2, 2)))
    locnet.add(Conv2D(64, (3, 3), padding='valid'))
    locnet.add(MaxPooling2D(pool_size=(2, 2)))

    locnet.add(Flatten())
    locnet.add(Dense(128, activation='elu'))
    locnet.add(Dense(64, activation='elu'))
    locnet.add(Dense(6, weights=weights))

    return locnet

def build_model():
    with tf.device('/device:GPU:2'):
        base_model = applications.VGG19(weights='imagenet',include_top=False,input_shape=train_tensors.shape[1:])
        add_model = Sequential()
        add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        added0_model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
        stn_model = Sequential()
        stn_model.add(Lambda(
        		lambda x: 2 * x - 1.,
        		input_shape=train_tensors.shape[1:],
        		output_shape=train_tensors.shape[1:]))
        stn_model.add(BatchNormalization())
        stn_model.add(SpatialTransformer(localization_net=locnet(),
                                     output_size=train_tensors.shape[1:3]))
        added_model = Model(inputs=stn_model.input, outputs=added0_model(stn_model.output))

        inp = Input(batch_shape=(None, train_data.shape[1]))
        extra_model = Model(input=inp, output=inp)

        x = concatenate([added_model.output,extra_model.output])
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1, activation='sigmoid')(x)

        model = Model(input=[added_model.input,extra_model.input],output=x)

        model.summary()
        return model

train_filename = "data_preprocessed/train_data_sample_rgb.p"
(train_labels, train_data, train_tensors) = pickle.load(open(train_filename, mode='rb'))

valid_filename = "data_preprocessed/valid_data_sample_rgb.p"
(valid_labels, valid_data, valid_tensors) = pickle.load(open(valid_filename, mode='rb'))

test_filename = "data_preprocessed/test_data_sample_rgb.p"
(test_labels, test_data, test_tensors) = pickle.load(open(test_filename, mode='rb'))

model = build_model()
model.compile(optimizer=optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True),
              loss='mean_squared_error',
              metrics=['accuracy',
                      precision_threshold(threshold = 0.5),
                       recall_threshold(threshold = 0.5),
                       fbeta_score_threshold(beta=0.5, threshold = 0.5)])

epochs = 100
batch_size = 32

earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=18, verbose=1, mode='auto')
log = CSVLogger('saved_models/log_vgg19_small_CNN.csv')
checkpointer = ModelCheckpoint(filepath='saved_models/VGG19_CNN_small.best.from_scratch.hdf5',
                               verbose=1, save_best_only=True)

start = time.time()

model.fit([train_tensors, train_data], train_labels,
          validation_data=([valid_tensors, valid_data], valid_labels),
          epochs=epochs, batch_size=batch_size, callbacks=[checkpointer, log, earlystop], verbose=1)

# Show total training time
print("training time: %.2f minutes"%((time.time()-start)/60))

model.load_weights('saved_models/VGG19_CNN_small.best.from_scratch.hdf5')
prediction = model.predict([test_tensors,test_data])

threshold = 0.5
beta = 0.5

pre = K.eval(precision_threshold(threshold = threshold)(K.variable(value=test_labels),
                                   K.variable(value=prediction)))
rec = K.eval(recall_threshold(threshold = threshold)(K.variable(value=test_labels),
                                   K.variable(value=prediction)))
fsc = K.eval(fbeta_score_threshold(beta = beta, threshold = threshold)(K.variable(value=test_labels),
                                   K.variable(value=prediction)))

print ("Precision: %f %%\nRecall: %f %%\nFscore: %f %%"% (pre, rec, fsc))

print(K.eval(binary_accuracy(K.variable(value=test_labels),
                       K.variable(value=prediction))))
