import tensorflow as tf
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import  Dense, Flatten, Activation, Dropout, Embedding, Conv1D, Conv2D, MaxPooling2D, MaxPooling1D, Concatenate, BatchNormalization, GaussianNoise
from tensorflow.keras.layers import LSTM, TimeDistributed, Permute, Reshape, Lambda, RepeatVector, Input, Multiply, SimpleRNN, GRU, LeakyReLU
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention
import site
import pandas as pd
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


site.addsitedir(os.path.realpath(__file__))
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 40)

from ncRNA_utils import *

def baseline_CNN_finalist_128(model_name, inshape, num_classes = 13):

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv1D(128 ,10 ,padding='same' ,input_shape=inshape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.MaxPooling1D(2))

    model.add(tf.keras.layers.GaussianNoise(1))
    model.add(tf.keras.layers.Dropout(rate=0.5))

    model.add(tf.keras.layers.Conv1D(128 ,10 ,padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.MaxPooling1D(4))

    model.add(tf.keras.layers.GaussianNoise(1))
    model.add(tf.keras.layers.Dropout(rate=0.5))

    model.add(tf.keras.layers.Conv1D(256 ,10 ,padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.MaxPooling1D(2))

    model.add(tf.keras.layers.Conv1D(256 ,10 ,padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.MaxPooling1D(4))

    model.add(tf.keras.layers.GaussianNoise(1))
    model.add(tf.keras.layers.Dropout(rate=0.5))

    model.add(tf.keras.layers.Conv1D(256 ,10 ,padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.MaxPooling1D(4))

    model.add(tf.keras.layers.GaussianNoise(1))
    model.add(tf.keras.layers.Dropout(rate=0.5))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))

    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))

    model.add(tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax))
    model._name = model_name

    return model

def baseline_CNN_finalist_256(model_name, inshape, num_classes = 13):

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv1D(256 ,10 ,padding='same' ,input_shape=inshape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.MaxPooling1D(2))

    model.add(tf.keras.layers.GaussianNoise(1))
    model.add(tf.keras.layers.Dropout(rate=0.5))

    model.add(tf.keras.layers.Conv1D(256 ,10 ,padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.MaxPooling1D(4))

    model.add(tf.keras.layers.GaussianNoise(1))
    model.add(tf.keras.layers.Dropout(rate=0.5))

    model.add(tf.keras.layers.Conv1D(256 ,10 ,padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.MaxPooling1D(2))

    model.add(tf.keras.layers.Conv1D(256 ,10 ,padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.MaxPooling1D(4))

    model.add(tf.keras.layers.GaussianNoise(1))
    model.add(tf.keras.layers.Dropout(rate=0.5))

    model.add(tf.keras.layers.Conv1D(256 ,10 ,padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.MaxPooling1D(4))

    model.add(tf.keras.layers.GaussianNoise(1))
    model.add(tf.keras.layers.Dropout(rate=0.5))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))

    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))

    model.add(tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax))
    model._name = model_name

    return model

def model_with_pure_rnn_finalist(model_name, input_shape = (1000, 8,),num_classes = 13):

    # RNN part
    inputs = Input(shape=input_shape)
    lstm_one = Bidirectional \
        (GRU(256, return_sequences=True, kernel_initializer='RandomNormal', dropout= 0.5, recurrent_dropout = 0.5, recurrent_initializer='RandomNormal', bias_initializer='zero'))(inputs)
    lstm_two = Bidirectional \
        (GRU(128, return_sequences=True, kernel_initializer='RandomNormal', dropout= 0.5, recurrent_dropout = 0.5, recurrent_initializer='RandomNormal', bias_initializer='zero'))(lstm_one)
    attention = SeqWeightedAttention()(lstm_two)
    attention = Flatten()(attention)
    rnnoutput = Dense(256 ,kernel_initializer='RandomNormal', bias_initializer='zeros')(attention)
    rnnoutput = BatchNormalization()(rnnoutput)
    rnnoutput = GaussianNoise(1)(rnnoutput)
    rnnoutput = Dropout(0.4)(rnnoutput)

    # Dense Feed-forward
    dense_one = Dense(128, kernel_initializer='RandomNormal', bias_initializer='zeros')(rnnoutput)
    dense_one = LeakyReLU()(dense_one)
    dense_one = Dropout(0.5)(dense_one)
    dense_one = BatchNormalization()(dense_one)
    dense_two = Dense(64, kernel_initializer='RandomNormal', bias_initializer='zeros')(dense_one)
    dense_two = LeakyReLU()(dense_two)
    dense_two = Dropout(0.4)(dense_two)

    # Output
    output = Dense(num_classes, activation='softmax')(dense_two)
    model = Model([inputs], output, name = model_name)
    return model


def model_combination(model_name, input_shape,num_classes = 13):
    model = Sequential([
        tf.keras.Input(shape=input_shape),
        BatchNormalization(),
        Dense(256, kernel_initializer='RandomNormal', bias_initializer='zeros'),
        LeakyReLU(),
        Dropout(0.6),
        Dense(128, kernel_initializer='RandomNormal', bias_initializer='zeros', kernel_regularizer = tf.keras.regularizers.l1(1e-3)),
        LeakyReLU(),
        Dropout(0.6),
        Dense(32, kernel_initializer='RandomNormal', bias_initializer='zeros', kernel_regularizer = tf.keras.regularizers.l1(1e-2)),
        LeakyReLU(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ], name=model_name)
    return model


def compile_and_fit_model_basic(  model_func,
                                  model_name,
                                  input_shape,
                                  X_train,
                                  Y_train,
                                  save_max_epoch=True,
                                  save_final=False,
                                  patience_count = None,
                                  **kwargs):
    m = None
    if isinstance(model_func, tf.keras.models.Model):
        m = model_func
        m._name = model_name
    else:
        m = model_func(model_name, input_shape)

    callbacks_used = []
    if save_max_epoch:
        callbacks_used.append(ModelCheckpoint(f'{m.name}' + '_model_{epoch:03d}_{val_accuracy:0.3f}',
                                              save_weights_only=False,
                                              monitor='val_accuracy',
                                              mode='max',
                                              save_best_only=True))
    if patience_count is not None:
        callbacks_used.append(tf.keras.callbacks.EarlyStopping(patience=patience_count))

    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = m.fit(X_train, Y_train, callbacks=callbacks_used, verbose=2, **kwargs)
    if save_final:
        make_dir_if_not_exist(model_name)
        m.save(f"{m.name}_saved_model_after_fit")  # Save the model
    return (m, history)

def compile_and_fit_model_basic_noVal(  model_func,
                                  model_name,
                                  input_shape,
                                  X_train,
                                  Y_train,
                                  save_max_epoch=True,
                                  save_final=False,
                                  patience_count = None,
                                  **kwargs):
    m = None
    if isinstance(model_func, tf.keras.models.Model):
        m = model_func
        m._name = model_name
    else:
        m = model_func(model_name, input_shape)

    callbacks_used = []
    if save_max_epoch:
        callbacks_used.append(ModelCheckpoint(f'{m.name}' + '_model_{epoch:03d}_{accuracy:0.3f}',
                                              save_weights_only=False,
                                              monitor='accuracy',
                                              mode='max',
                                              save_best_only=True))
    if patience_count is not None:
        callbacks_used.append(tf.keras.callbacks.EarlyStopping(patience=patience_count))

    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = m.fit(X_train, Y_train, callbacks=callbacks_used, verbose=2, **kwargs)
    if save_final:
        make_dir_if_not_exist(model_name)
        m.save(f"{m.name}_saved_model_after_fit")  # Save the model
    return (m, history)


if __name__ == "__main__":
    os.chdir(os.path.realpath(__file__))
    
    X_train_1000e, Y_train_1000e, X_test_1000e, Y_test_1000e, X_val_1000e, Y_val_1000e = getE2eData13()
    X_new_train = np.concatenate( (X_train_1000e, X_val_1000e), axis=0 )
    Y_new_train = np.concatenate( (Y_train_1000e, Y_val_1000e), axis=0 )
    
    cnn_128, history_cnn_128 = compile_and_fit_model_basic_noVal(baseline_CNN_finalist_128,
                                                  f"cnn_128",
                                                  X_new_train[0].shape,
                                                  X_new_train,
                                                  Y_new_train,
                                                  save_max_epoch=True,
                                                  save_final=True,
                                                  patience_count=100,
                                                  batch_size=2048,
                                                  epochs=500,
                                                  class_weight=None)
    get_sp_pr_rc_f1_acc(cnn_128,X_test_1000e, Y_test_1000e)
    cnn_256, history_cnn_256 = compile_and_fit_model_basic_noVal(baseline_CNN_finalist_256,
                                                  f"cnn_256",
                                                  X_new_train[0].shape,
                                                  X_new_train,
                                                  Y_new_train,
                                                  save_max_epoch=True,
                                                  save_final=True,
                                                  patience_count=100,
                                                  batch_size=2048,
                                                  epochs=500,
                                                  class_weight=None)
    get_sp_pr_rc_f1_acc(cnn_256,X_test_1000e, Y_test_1000e)
    
    rnn_E, history_rnn_E = compile_and_fit_model_basic_noVal(model_with_pure_rnn_finalist,
                                                  f"rnn_{it}",
                                                  X_new_train[0].shape,
                                                  X_new_train,
                                                  Y_new_train,
                                                  save_max_epoch=True,
                                                  save_final=True,
                                                  patience_count=100,
                                                  batch_size=2048,
                                                  epochs=500,
                                                  class_weight=None)
    get_sp_pr_rc_f1_acc(rnn_E,X_test_1000e, Y_test_1000e)


    # cnn_256 = load_model("./data/CNN_baseline_May16_e2e1000_256.h5")    
    # rnn_E = load_model("./data/RNN_baseline_17May_180.h5",  custom_objects=SeqWeightedAttention.get_custom_objects())
    # tf.keras.utils.plot_model(cnn_256)
    # tf.keras.utils.plot_model(rnn_E)
    
    # LD(A) + LD(E)
    to_combine_ld_no2nd_rNo2nd = [
        (cnn_256 , "dense_25", None),   # change to the appropriate dense layer name
        (rnn_E,"dense_2", None) # change to the appropriate dense layer name
    ]
    
    combined_models_ld_no2nd_rNo2nd, data_train_ld_no2nd_rNo2nd, data_test_ld_no2nd_rNo2nd, data_access_ld_no2nd_rNo2nd = get_combined_features_from_models(
        to_combine_ld_no2nd_rNo2nd ,
        [ X_new_train, X_new_train],
        [ Y_new_train, Y_new_train], 
        [ X_test_1000e, X_test_1000e],
        [ Y_test_1000e, Y_test_1000e],
        reverse_one_hot=False)
    
        
    rcnn_combine_model = model_combination("rcnn_combined_models_ld_no2nd_rNo2nd", data_train_ld_no2nd_rNo2nd[0][0].shape  )
    rcnn_combine_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
    
    callbacks_used_rcnn_combine = [ModelCheckpoint(f'{rcnn_combine_model.name}' + '_model_{epoch:03d}_{accuracy:0.3f}',
                                                save_weights_only=False,
                                                monitor='accuracy',
                                                mode='max',
                                                save_best_only=True),
                        tf.keras.callbacks.EarlyStopping(patience=100)
                        ]
    history_rcnn_combine = rcnn_combine_model.fit(data_train_ld_no2nd_rNo2nd[0], 
                                                  data_train_ld_no2nd_rNo2nd[1][0], 
                                                  callbacks=callbacks_used_rcnn_combine, 
                                                  verbose=2, 
                                                  epochs = 500, 
                                                  batch_size=64)
    rcnn_combine_model.save(f"{rcnn_combine_model.name}.h5")    
    # rcnn_combine_model.evaluate(data_test_ld_no2nd_rNo2nd[0],data_test_ld_no2nd_rNo2nd[1][0]) 

    measures_rcnn = []
    for it in range(15):
        print(it)
        rcnn_combine_model = model_combination(f"rcnn_combined_models_ld_no2nd_rNo2nd_{it}", data_train_ld_no2nd_rNo2nd[0][0].shape  )
        rcnn_combine_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
        callbacks_used_rcnn_combine = [ModelCheckpoint(f'{rcnn_combine_model.name}' + '_model_{epoch:03d}_{accuracy:0.3f}',
                                                    save_weights_only=False,
                                                    monitor='accuracy',
                                                    mode='max',
                                                    save_best_only=True),
                            tf.keras.callbacks.EarlyStopping(patience=100)
                        ]
        rcnn_combine_model.fit(data_train_ld_no2nd_rNo2nd[0], 
                                                  data_train_ld_no2nd_rNo2nd[1][0], 
                                                  callbacks=callbacks_used_rcnn_combine, 
                                                  verbose=2, 
                                                  epochs = 500, 
                                                  batch_size=64)
        measures_rcnn.append(get_sp_pr_rc_f1_acc(rcnn_combine_model,data_test_ld_no2nd_rNo2nd[0],data_test_ld_no2nd_rNo2nd[1][0]))


    import pandas as pd
    pd.DataFrame(np.array(measures_rcnn)).to_csv("./data/ncRNA_measures_rcnn.csv", index=False, header=True)


