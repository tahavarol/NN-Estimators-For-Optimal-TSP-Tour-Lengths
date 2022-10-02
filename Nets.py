from PrepareData import *
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, SimpleRNN, Conv2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras import models
from keras import layers
from keras import optimizers
from keras import initializers
from keras import regularizers
from keras import Input
import tensorflow as tf


def FCNNSmall(size, data, label, k, mode, n_layers, n_neurons, p_dropout):
    
    #data,label = prepare_without_moats_parallel(size, order=k, mode=mode)
    
    X_train, X_val, y_train, y_val = train_test_split(data,label, test_size=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)
    
    filepath = "models/hpt/FCNNSmallBest_{}_{}.hdf5".format(size,k)
    checkpoint = ModelCheckpoint(filepath, monitor='val_mape', verbose=1, 
                                 save_best_only=True, mode='min', save_weights_only=False)

    earlystopping = EarlyStopping(monitor='val_mape', mode='min', verbose=1, patience=100)

    def scheduler(epoch, lr):
        if epoch < 10:
             return lr
        else:
            return lr * 0.995

    lr_scheduler = LearningRateScheduler(scheduler)

    callbacks_list = [checkpoint,earlystopping,lr_scheduler]

    tf.random.set_seed(1)
    model = Sequential()
    model.add(Dense(n_neurons, input_dim=X_train.shape[1], kernel_initializer='glorot_normal', activation='relu', use_bias = True))
    model.add(Dropout(p_dropout))
    for _ in range(n_layers-3):
        model.add(Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True))
        model.add(Dropout(p_dropout))
    model.add(Dense(1, kernel_initializer='glorot_normal', activation='relu', use_bias = True))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape', 'mean_squared_error'])
    model.summary()
    
    history = model.fit(x=X_train, y=y_train, batch_size=128, epochs=100, verbose=1, shuffle=True, 
                    validation_data=(X_val,y_val), callbacks=callbacks_list)
    
    model = load_model("models/hpt/FCNNSmallBest_{}_{}.hdf5".format(size,k))
    
    history.history["test_metrics"] = model.evaluate(X_test,y_test)
    
    return history.history




def FCNNLarge(size, data, label, k, mode, n_layers, n_neurons, p_dropout):
    
    #data,label = prepare_with_moats_parallel(size, order=k, mode=mode)
    
    X_train, X_val, y_train, y_val = train_test_split(data,label, test_size=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)
    
    filepath = "models/hpt/FCNNLargeBest_{}_{}.hdf5".format(size,k)
    checkpoint = ModelCheckpoint(filepath, monitor='val_mape', verbose=1, 
                                 save_best_only=True, mode='min', save_weights_only=False)

    earlystopping = EarlyStopping(monitor='val_mape', mode='min', verbose=1, patience=100)

    def scheduler(epoch, lr):
        if epoch < 10:
             return lr
        else:
            return lr * 0.995

    lr_scheduler = LearningRateScheduler(scheduler)

    callbacks_list = [checkpoint,earlystopping,lr_scheduler]

    tf.random.set_seed(1)
    model = Sequential()
    model.add(Dense(n_neurons, input_dim=X_train.shape[1], kernel_initializer='glorot_normal', activation='relu', use_bias = True))
    model.add(Dropout(p_dropout))
    for _ in range(n_layers-3):
        model.add(Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True))
        model.add(Dropout(p_dropout))
    model.add(Dense(1, kernel_initializer='glorot_normal', activation='relu', use_bias = True))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape', 'mean_squared_error'])
    model.summary()
    
    history = model.fit(x=X_train, y=y_train, batch_size=128, epochs=100, verbose=1, shuffle=True, 
                    validation_data=(X_val,y_val), callbacks=callbacks_list)
    
    model = load_model("models/hpt/FCNNLargeBest_{}_{}.hdf5".format(size,k))
    
    history.history["test_metrics"] = model.evaluate(X_test,y_test)
    
    return history.history

def PCNNSmall(size, data, label, k, mode, n_layers, n_neurons, p_dropout):
    
    #data,label = prepare_without_moats_parallel(size, order=k, mode=mode)
    
    X_train, X_val, y_train, y_val = train_test_split(data,label, test_size=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)
    
    filepath = "models/hpt/PCNNSmallBest_{}_{}.hdf5".format(size,k)
    checkpoint = ModelCheckpoint(filepath, monitor='val_mape', verbose=1, 
                                 save_best_only=True, mode='min', save_weights_only=False)

    earlystopping = EarlyStopping(monitor='val_mape', mode='min', verbose=1, patience=100)

    def scheduler(epoch, lr):
        if epoch < 10:
             return lr
        else:
            return lr * 0.995

    lr_scheduler = LearningRateScheduler(scheduler)

    callbacks_list = [checkpoint,earlystopping,lr_scheduler]
    
    
    X_D = layers.Input(shape=(2,))
    X_NC = layers.Input(shape=(size*2,))
    X_KED = layers.Input(shape=(size*k,))

    tf.random.set_seed(1)
    
    H1 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(X_D)
    H1 = layers.Dropout(p_dropout)(H1)
    H2 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(X_NC)
    H2 = layers.Dropout(p_dropout)(H2)
    H3 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(X_KED)
    H3 = layers.Dropout(p_dropout)(H3)
    
    for _ in range(n_layers-3):
        H1 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(H1)
        H1 = layers.Dropout(p_dropout)(H1)
        H2 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(H2)
        H2 = layers.Dropout(p_dropout)(H2)
        H3 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(H3)
        H3 = layers.Dropout(p_dropout)(H3)

    H = layers.Concatenate()([H1, H2, H3])
    H = layers.Dropout(p_dropout)(H)
    output = layers.Dense(1, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(H)
    
    model = models.Model(inputs=[X_D,X_NC,X_KED], outputs=output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape', 'mean_squared_error'])
    model.summary()
    
    history = model.fit(x=[X_train[:,:2],X_train[:,2:(2+(size*2))],X_train[:,(2+(size*2)):]], y=y_train, batch_size=128, epochs=100, verbose=1, shuffle=True, 
                    validation_data=([X_val[:,:2],X_val[:,2:(2+(size*2))],X_val[:,(2+(size*2)):]],y_val), callbacks=callbacks_list)
    
    model = load_model("models/hpt/PCNNSmallBest_{}_{}.hdf5".format(size,k))
    
    history.history["test_metrics"] = model.evaluate([X_test[:,:2],X_test[:,2:(2+(size*2))],X_test[:,(2+(size*2)):]],y_test)
    
    return history.history




def PCNNLarge(size,data,label, k, mode, n_layers, n_neurons, p_dropout):
    
    #data,label = prepare_with_moats_parallel(size, order=k, mode=mode)
    
    X_train, X_val, y_train, y_val = train_test_split(data,label, test_size=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)
    
    filepath = "models/hpt/PCNNLargeBest_{}_{}.hdf5".format(size,k)
    checkpoint = ModelCheckpoint(filepath, monitor='val_mape', verbose=1, 
                                 save_best_only=True, mode='min', save_weights_only=False)

    earlystopping = EarlyStopping(monitor='val_mape', mode='min', verbose=1, patience=100)

    def scheduler(epoch, lr):
        if epoch < 10:
             return lr
        else:
            return lr * 0.995

    lr_scheduler = LearningRateScheduler(scheduler)

    callbacks_list = [checkpoint,earlystopping,lr_scheduler]
    
    
    X_D = layers.Input(shape=(2,))
    X_NC = layers.Input(shape=(size*2,))
    X_KED = layers.Input(shape=(size*k,))
    X_PS = layers.Input(shape=((2*size)-1,))
    
    tf.random.set_seed(1)
    
    H1 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(X_D)
    H1 = layers.Dropout(p_dropout)(H1)
    H2 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(X_NC)
    H2 = layers.Dropout(p_dropout)(H2)
    H3 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(X_KED)
    H3 = layers.Dropout(p_dropout)(H3)
    H4 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(X_PS)
    H4 = layers.Dropout(p_dropout)(H4)


    
    for _ in range(n_layers-3):
        H1 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(H1)
        H1 = layers.Dropout(p_dropout)(H1)
        H2 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(H2)
        H2 = layers.Dropout(p_dropout)(H2)
        H3 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(H3)
        H3 = layers.Dropout(p_dropout)(H3)
        H4 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(H4)
        H4 = layers.Dropout(p_dropout)(H4)


    H = layers.Concatenate()([H1, H2, H3, H4])
    H = layers.Dropout(p_dropout)(H)
    output = layers.Dense(1, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(H)
    
    model = models.Model(inputs=[X_D,X_NC,X_KED,X_PS], outputs=output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape', 'mean_squared_error'])
    model.summary()
    
    history = model.fit(x=[X_train[:,:2],X_train[:,2:(2+(size*2))],X_train[:,(2+(size*2)):(2+(size*(2+k)))],X_train[:,(2+(size*(2+k))):]], y=y_train, batch_size=128, epochs=100, verbose=1, shuffle=True, 
                    validation_data=([X_val[:,:2],X_val[:,2:(2+(size*2))],X_val[:,(2+(size*2)):(2+(size*(2+k)))],X_val[:,(2+(size*(2+k))):]],y_val), callbacks=callbacks_list)
    
    model = load_model("models/hpt/PCNNLargeBest_{}_{}.hdf5".format(size,k))
    
    history.history["test_metrics"] = model.evaluate([X_test[:,:2],X_test[:,2:(2+(size*2))],X_test[:,(2+(size*2)):(2+(size*(2+k)))],X_test[:,(2+(size*(2+k))):]],y_test)
    
    return history.history



def UniversalApproximatorNet(size, n_layers=3, n_neurons=1000, p_dropout=0.0):
    
    data = load_raw(size)
    
    l = data[:,-1]
    d = data[:,:2*size]
    
    X_train, X_val, y_train, y_val = train_test_split(d,l, test_size=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)
    
    filepath = "models/hpt/UniversalApproximatorNetBest_{}.hdf5".format(size)
    checkpoint = ModelCheckpoint(filepath, monitor='val_mape', verbose=1, 
                                 save_best_only=True, mode='min', save_weights_only=False)

    earlystopping = EarlyStopping(monitor='val_mape', mode='min', verbose=1, patience=100)

    def scheduler(epoch, lr):
        if epoch < 10:
             return lr
        else:
            return lr * 0.995

    lr_scheduler = LearningRateScheduler(scheduler)

    callbacks_list = [checkpoint,earlystopping,lr_scheduler]

    tf.random.set_seed(1)
    model = Sequential()
    model.add(Dense(n_neurons, input_dim=X_train.shape[1], kernel_initializer='glorot_normal', activation='relu', use_bias = True))
    model.add(Dropout(p_dropout))
    for _ in range(n_layers-3):
        model.add(Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True))
        model.add(Dropout(p_dropout))
    model.add(Dense(1, kernel_initializer='glorot_normal', activation='relu', use_bias = True))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape', 'mean_squared_error'])
    model.summary()
    
    history = model.fit(x=X_train, y=y_train, batch_size=128, epochs=100, verbose=1, shuffle=True, 
                    validation_data=(X_val,y_val), callbacks=callbacks_list)
    
    model = load_model("models/hpt/UniversalApproximatorNetBest_{}.hdf5".format(size))
    
    history.history["test_metrics"] = model.evaluate(X_test,y_test)
    
    return history.history



def LiteratureMLP(size, n_layers, n_neurons, p_dropout):
    
    data = pd.read_csv("TrainingData/LiteratureFeatures/LiteratureFeatures_{}.csv".format(size)).to_numpy()
    label = pd.read_csv("TrainingData/Raw/{}.csv".format(size)).to_numpy()[:,-1]
    
    
    X_train, X_val, y_train, y_val = train_test_split(data,label, test_size=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


    
    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)

    filepath = "models/hpt/{}Best_{}.hdf5".format("Literature",size)
    
    checkpoint = ModelCheckpoint(filepath, monitor='val_mape', verbose=1, 
                                 save_best_only=True, mode='min', save_weights_only=False)

    earlystopping = EarlyStopping(monitor='val_mape', mode='min', verbose=1, patience=100)

    def scheduler(epoch, lr):
        if epoch < 10:
             return lr
        else:
            return lr * 0.995

    lr_scheduler = LearningRateScheduler(scheduler)

    callbacks_list = [checkpoint,earlystopping,lr_scheduler]

    tf.random.set_seed(1)
    model = Sequential()
    model.add(Dense(n_neurons, input_dim=X_train.shape[1], kernel_initializer='glorot_normal', activation='relu', use_bias = True))
    model.add(Dropout(p_dropout))
    for _ in range(n_layers-3):
        model.add(Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True))
        model.add(Dropout(p_dropout))
    model.add(Dense(1, kernel_initializer='glorot_normal', activation='relu', use_bias = True))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape', 'mean_squared_error'])
    model.summary()

    history = model.fit(x=X_train, y=y_train, batch_size=128, epochs=100, verbose=1, shuffle=True, 
                    validation_data=(X_val,y_val), callbacks=callbacks_list)

    model = load_model("models/hpt/{}Best_{}.hdf5".format("Literature",size))

    history.history["test_metrics"] = model.evaluate(X_test,y_test)
    
    return history.history



def LiteratureHybridFCNNSmall(size, k, mode, n_layers, n_neurons, p_dropout):
    data, label = prepare_without_moats_litf_parallel(size, order=k, mode=4)
    
    X_train, X_val, y_train, y_val = train_test_split(data,label, test_size=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)
    
    filepath = "models/hpt/FCNNSmallLiteratureBest_{}_{}.hdf5".format(size,k)
    checkpoint = ModelCheckpoint(filepath, monitor='val_mape', verbose=1, 
                                 save_best_only=True, mode='min', save_weights_only=False)

    earlystopping = EarlyStopping(monitor='val_mape', mode='min', verbose=1, patience=100)

    def scheduler(epoch, lr):
        if epoch < 10:
             return lr
        else:
            return lr * 0.995

    lr_scheduler = LearningRateScheduler(scheduler)

    callbacks_list = [checkpoint,earlystopping,lr_scheduler]

    tf.random.set_seed(1)
    model = Sequential()
    model.add(Dense(n_neurons, input_dim=X_train.shape[1], kernel_initializer='glorot_normal', activation='relu', use_bias = True))
    model.add(Dropout(p_dropout))
    for _ in range(n_layers-3):
        model.add(Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True))
        model.add(Dropout(p_dropout))
    model.add(Dense(1, kernel_initializer='glorot_normal', activation='relu', use_bias = True))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape', 'mean_squared_error'])
    model.summary()
    
    history = model.fit(x=X_train, y=y_train, batch_size=128, epochs=100, verbose=1, shuffle=True, 
                    validation_data=(X_val,y_val), callbacks=callbacks_list)
    
    model = load_model("models/hpt/FCNNSmallLiteratureBest_{}_{}.hdf5".format(size,k))
    
    history.history["test_metrics"] = model.evaluate(X_test,y_test)
    
    return history.history


def LiteratureHybridFCNNLarge(size, k, mode, n_layers, n_neurons, p_dropout):
    
    data, label = prepare_with_moats_litf_parallel(size, order=k, mode=4)
    
    X_train, X_val, y_train, y_val = train_test_split(data,label, test_size=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)
    
    filepath = "models/hpt/FCNNLargeLiteratureBest_{}_{}.hdf5".format(size,k)
    checkpoint = ModelCheckpoint(filepath, monitor='val_mape', verbose=1, 
                                 save_best_only=True, mode='min', save_weights_only=False)

    earlystopping = EarlyStopping(monitor='val_mape', mode='min', verbose=1, patience=100)

    def scheduler(epoch, lr):
        if epoch < 10:
             return lr
        else:
            return lr * 0.995

    lr_scheduler = LearningRateScheduler(scheduler)

    callbacks_list = [checkpoint,earlystopping,lr_scheduler]

    tf.random.set_seed(1)
    model = Sequential()
    model.add(Dense(n_neurons, input_dim=X_train.shape[1], kernel_initializer='glorot_normal', activation='relu', use_bias = True))
    model.add(Dropout(p_dropout))
    for _ in range(n_layers-3):
        model.add(Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True))
        model.add(Dropout(p_dropout))
    model.add(Dense(1, kernel_initializer='glorot_normal', activation='relu', use_bias = True))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape', 'mean_squared_error'])
    model.summary()
    
    history = model.fit(x=X_train, y=y_train, batch_size=128, epochs=100, verbose=1, shuffle=True, 
                    validation_data=(X_val,y_val), callbacks=callbacks_list)
    
    model = load_model("models/hpt/FCNNLargeLiteratureBest_{}_{}.hdf5".format(size,k))
    
    history.history["test_metrics"] = model.evaluate(X_test,y_test)
    
    return history.history


def LiteratureHybridPCNNSmall(size, k, mode, n_layers, n_neurons, p_dropout):
    
    data, label = prepare_without_moats_litf_parallel(size, order=k, mode=4)
    
    X_train, X_val, y_train, y_val = train_test_split(data,label, test_size=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)
    
    filepath = "models/hpt/PCNNSmallLiteratureBest_{}_{}.hdf5".format(size,k)
    checkpoint = ModelCheckpoint(filepath, monitor='val_mape', verbose=1, 
                                 save_best_only=True, mode='min', save_weights_only=False)

    earlystopping = EarlyStopping(monitor='val_mape', mode='min', verbose=1, patience=100)

    def scheduler(epoch, lr):
        if epoch < 10:
             return lr
        else:
            return lr * 0.995

    lr_scheduler = LearningRateScheduler(scheduler)

    callbacks_list = [checkpoint,earlystopping,lr_scheduler]
   
    X_D = layers.Input(shape=(2,))
    X_NC = layers.Input(shape=(size*2,))
    X_KED = layers.Input(shape=(size*k,))
    X_LITF = layers.Input(shape=(36,))
    
    tf.random.set_seed(1)
    
    H1 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(X_D)
    H1 = layers.Dropout(p_dropout)(H1)
    H2 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(X_NC)
    H2 = layers.Dropout(p_dropout)(H2)
    H3 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(X_KED)
    H3 = layers.Dropout(p_dropout)(H3)
    H4 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(X_LITF)
    H4 = layers.Dropout(p_dropout)(H4)


    
    for _ in range(n_layers-3):
        H1 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(H1)
        H1 = layers.Dropout(p_dropout)(H1)
        H2 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(H2)
        H2 = layers.Dropout(p_dropout)(H2)
        H3 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(H3)
        H3 = layers.Dropout(p_dropout)(H3)
        H4 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(H4)
        H4 = layers.Dropout(p_dropout)(H4)

    H = layers.Concatenate()([H1, H2, H3, H4])
    H = layers.Dropout(p_dropout)(H)
    output = layers.Dense(1, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(H)
    
    model = models.Model(inputs=[X_D,X_NC,X_KED,X_LITF], outputs=output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape', 'mean_squared_error'])
    model.summary()
    
    history = model.fit(x=[X_train[:,:2],
                           X_train[:,2:(2+(size*2))],
                           X_train[:,(2+(size*2)):(2+(size*(k+2)))],
                           X_train[:,(2+(size*(k+2))):]], 
                        y=y_train, 
                        batch_size=128, 
                        epochs=100, 
                        verbose=1, 
                        shuffle=True, 
                    validation_data=([X_val[:,:2],
                                      X_val[:,2:(2+(size*2))],
                                      X_val[:,(2+(size*2)):(2+(size*(k+2)))],
                                      X_val[:,(2+(size*(k+2))):]],y_val), 
                        callbacks=callbacks_list)
    
    model = load_model("models/hpt/PCNNSmallLiteratureBest_{}_{}.hdf5".format(size,k))
    
    history.history["test_metrics"] = model.evaluate([X_test[:,:2],
                                                      X_test[:,2:(2+(size*2))],
                                                      X_test[:,(2+(size*2)):(2+(size*(k+2)))],
                                                      X_test[:,(2+(size*(k+2))):]],y_test)
    
    return history.history



def LiteratureHybridPCNNLarge(size, k, mode, n_layers, n_neurons, p_dropout):
    
    data, label = prepare_with_moats_litf_parallel(size, order=k, mode=4)
    
    X_train, X_val, y_train, y_val = train_test_split(data,label, test_size=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)
    
    filepath = "models/hpt/PCNNLargeLiteratureBest_{}_{}.hdf5".format(size,k)
    checkpoint = ModelCheckpoint(filepath, monitor='val_mape', verbose=1, 
                                 save_best_only=True, mode='min', save_weights_only=False)

    earlystopping = EarlyStopping(monitor='val_mape', mode='min', verbose=1, patience=100)

    def scheduler(epoch, lr):
        if epoch < 10:
             return lr
        else:
            return lr * 0.995

    lr_scheduler = LearningRateScheduler(scheduler)

    callbacks_list = [checkpoint,earlystopping,lr_scheduler]
   
    X_D = layers.Input(shape=(2,))
    X_NC = layers.Input(shape=(size*2,))
    X_KED = layers.Input(shape=(size*k,))
    X_PS = layers.Input(shape=((2*size)-1,))
    X_LITF = layers.Input(shape=(36,))
    
    tf.random.set_seed(12)
    
    H1 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(X_D)
    H1 = layers.Dropout(p_dropout)(H1)
    H2 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(X_NC)
    H2 = layers.Dropout(p_dropout)(H2)
    H3 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(X_KED)
    H3 = layers.Dropout(p_dropout)(H3)
    H4 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(X_PS)
    H4 = layers.Dropout(p_dropout)(H4)
    H5 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(X_LITF)
    H5 = layers.Dropout(p_dropout)(H5)



    
    for _ in range(n_layers-3):
        H1 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(H1)
        H1 = layers.Dropout(p_dropout)(H1)
        H2 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(H2)
        H2 = layers.Dropout(p_dropout)(H2)
        H3 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(H3)
        H3 = layers.Dropout(p_dropout)(H3)
        H4 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(H4)
        H4 = layers.Dropout(p_dropout)(H4)
        H5 = layers.Dense(n_neurons, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(H5)
        H5 = layers.Dropout(p_dropout)(H5)



    H = layers.Concatenate()([H1, H2, H3, H4, H5])
    H = layers.Dropout(p_dropout)(H)
    output = layers.Dense(1, kernel_initializer='glorot_normal', activation='relu', use_bias = True)(H)
    
    model = models.Model(inputs=[X_D,X_NC,X_KED,X_PS,X_LITF], outputs=output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape', 'mean_squared_error'])
    model.summary()
    
    history = model.fit(x=[X_train[:,:2],
                           X_train[:,2:(2+(size*2))],
                           X_train[:,(2+(size*2)):(2+(size*(k+2)))],
                           X_train[:,(2+(size*(k+2))):(1+(size*(k+4)))],
                           X_train[:,(1+(size*(k+4))):]], 
                        y=y_train, 
                        batch_size=128, 
                        epochs=100, 
                        verbose=1, 
                        shuffle=True, 
                    validation_data=([X_val[:,:2],
                                      X_val[:,2:(2+(size*2))],
                                      X_val[:,(2+(size*2)):(2+(size*(k+2)))],
                                      X_val[:,(2+(size*(k+2))):(1+(size*(k+4)))],
                                      X_val[:,(1+(size*(k+4))):]],y_val), 
                        callbacks=callbacks_list)
    
    model = load_model("models/hpt/PCNNLargeLiteratureBest_{}_{}.hdf5".format(size,k))
    
    history.history["test_metrics"] = model.evaluate([X_test[:,:2],
                                                      X_test[:,2:(2+(size*2))],
                                                      X_test[:,(2+(size*2)):(2+(size*(k+2)))],
                                                      X_test[:,(2+(size*(k+2))):(1+(size*(k+4)))],
                                                      X_test[:,(1+(size*(k+4))):]],y_test)
    
    return history.history



def RNNSmall(size, k, mode, n_neurons, p_dropout):
    
    data,label = prepare_without_moats_parallel(size, order=k, mode=mode)
    X_train, X_val, y_train, y_val = train_test_split(data,label, test_size=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    X_train = np.expand_dims(X_train, axis=1)
    X_val = np.expand_dims(X_val, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    
    
    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)

    filepath = "models/hpt/RNNSmallBest_{}_{}.hdf5".format(size,k)
    checkpoint = ModelCheckpoint(filepath, monitor='val_mape', verbose=1, 
                                 save_best_only=True, mode='min', save_weights_only=False)

    earlystopping = EarlyStopping(monitor='val_mape', mode='min', verbose=1, patience=100)

    def scheduler(epoch, lr):
        if epoch < 10:
             return lr
        else:
            return lr * 0.995

    lr_scheduler = LearningRateScheduler(scheduler)

    callbacks_list = [checkpoint,earlystopping,lr_scheduler]
    
    tf.random.set_seed(1)
    model = Sequential()
    model.add(SimpleRNN(n_neurons, input_shape=(X_train.shape[1],X_train.shape[2]), dropout = p_dropout, return_sequences=False))
    model.add(Dense(1, kernel_initializer='glorot_normal', activation='relu', use_bias = True))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape', 'mean_squared_error'])
    model.summary()

    history = model.fit(x=X_train, y=y_train, batch_size=128, epochs=100, verbose=1, shuffle=True, 
                    validation_data=(X_val,y_val), callbacks=callbacks_list)

    model = load_model("models/hpt/RNNSmallBest_{}_{}.hdf5".format(size,k))

    history.history["test_metrics"] = model.evaluate(X_test,y_test)

    print(history.history["test_metrics"])
    
    return history.history

def RNNLarge(size, k, mode, n_neurons, p_dropout):
    
    data,label = prepare_with_moats_parallel(size, order=k, mode=mode)
    X_train, X_val, y_train, y_val = train_test_split(data,label, test_size=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    X_train = np.expand_dims(X_train, axis=1)
    X_val = np.expand_dims(X_val, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    
    
    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)

    filepath = "models/hpt/RNNLargeBest_{}_{}.hdf5".format(size,k)
    checkpoint = ModelCheckpoint(filepath, monitor='val_mape', verbose=1, 
                                 save_best_only=True, mode='min', save_weights_only=False)

    earlystopping = EarlyStopping(monitor='val_mape', mode='min', verbose=1, patience=100)

    def scheduler(epoch, lr):
        if epoch < 10:
             return lr
        else:
            return lr * 0.995

    lr_scheduler = LearningRateScheduler(scheduler)

    callbacks_list = [checkpoint,earlystopping,lr_scheduler]
    
    tf.random.set_seed(1)
    model = Sequential()
    model.add(SimpleRNN(n_neurons, input_shape=(X_train.shape[1],X_train.shape[2]), dropout = p_dropout, return_sequences=False))
    model.add(Dense(1, kernel_initializer='glorot_normal', activation='relu', use_bias = True))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape', 'mean_squared_error'])
    model.summary()

    history = model.fit(x=X_train, y=y_train, batch_size=128, epochs=100, verbose=1, shuffle=True, 
                    validation_data=(X_val,y_val), callbacks=callbacks_list)

    model = load_model("models/hpt/RNNLargeBest_{}_{}.hdf5".format(size,k))

    history.history["test_metrics"] = model.evaluate(X_test,y_test)

    print(history.history["test_metrics"])
    
    return history.history
    

    
    
def CNNSmall(size, k, mode, kernel_width, num_filters, p_dropout):
    
    data,label = prepare_without_moats_parallel(size, order=k, mode=mode)
    X_train, X_val, y_train, y_val = train_test_split(data,label, test_size=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    X_train = np.expand_dims(X_train, axis=1)
    X_val = np.expand_dims(X_val, axis=1)
    X_test = np.expand_dims(X_test, axis=1)


    X_train = np.expand_dims(X_train, axis=3)
    X_val = np.expand_dims(X_val, axis=3)
    X_test = np.expand_dims(X_test, axis=3)
    
    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)

    filepath = "models/hpt/CNNSmallBest_{}_{}.hdf5".format(size,k)
    checkpoint = ModelCheckpoint(filepath, monitor='val_mape', verbose=1, 
                                 save_best_only=True, mode='min', save_weights_only=False)

    earlystopping = EarlyStopping(monitor='val_mape', mode='min', verbose=1, patience=100)

    def scheduler(epoch, lr):
        if epoch < 10:
             return lr
        else:
            return lr * 0.995

    lr_scheduler = LearningRateScheduler(scheduler)

    callbacks_list = [checkpoint,earlystopping,lr_scheduler]

    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    
    
    tf.random.set_seed(1)
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            layers.Conv2D(num_filters, kernel_size=(1, kernel_width), activation="relu"),
            layers.MaxPooling2D(pool_size=(1, kernel_width)),
            layers.Flatten(),
            layers.Dropout(p_dropout),
            layers.Dense(1, activation="relu"),
        ]
    )

    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape', 'mean_squared_error'])

    history = model.fit(x=X_train, y=y_train, batch_size=128, epochs=100, verbose=1, shuffle=True, 
                    validation_data=(X_val,y_val), callbacks=callbacks_list)

    
    model = load_model("models/hpt/CNNSmallBest_{}_{}.hdf5".format(size,k))

    history.history["test_metrics"] = model.evaluate(X_test,y_test)

    print(history.history["test_metrics"])
    
    return history.history


def CNNLarge(size, k, mode, kernel_width, num_filters, p_dropout):
    
    data,label = prepare_with_moats_parallel(size, order=k, mode=mode)
    X_train, X_val, y_train, y_val = train_test_split(data,label, test_size=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    X_train = np.expand_dims(X_train, axis=1)
    X_val = np.expand_dims(X_val, axis=1)
    X_test = np.expand_dims(X_test, axis=1)


    X_train = np.expand_dims(X_train, axis=3)
    X_val = np.expand_dims(X_val, axis=3)
    X_test = np.expand_dims(X_test, axis=3)
    
    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)

    filepath = "models/hpt/CNNLargeBest_{}_{}.hdf5".format(size,k)
    checkpoint = ModelCheckpoint(filepath, monitor='val_mape', verbose=1, 
                                 save_best_only=True, mode='min', save_weights_only=False)

    earlystopping = EarlyStopping(monitor='val_mape', mode='min', verbose=1, patience=100)

    def scheduler(epoch, lr):
        if epoch < 10:
             return lr
        else:
            return lr * 0.995

    lr_scheduler = LearningRateScheduler(scheduler)

    callbacks_list = [checkpoint,earlystopping,lr_scheduler]

    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    
    
    tf.random.set_seed(1)
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            layers.Conv2D(num_filters, kernel_size=(1, kernel_width), activation="relu"),
            layers.MaxPooling2D(pool_size=(1, kernel_width)),
            layers.Flatten(),
            layers.Dropout(p_dropout),
            layers.Dense(1, activation="relu"),
        ]
    )

    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape', 'mean_squared_error'])

    history = model.fit(x=X_train, y=y_train, batch_size=128, epochs=100, verbose=1, shuffle=True, 
                    validation_data=(X_val,y_val), callbacks=callbacks_list)

    
    model = load_model("models/hpt/CNNLargeBest_{}_{}.hdf5".format(size,k))

    history.history["test_metrics"] = model.evaluate(X_test,y_test)

    print(history.history["test_metrics"])
    
    return history.history








#def BlockNet():

#def TabTransformer():
