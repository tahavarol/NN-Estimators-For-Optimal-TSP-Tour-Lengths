from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras import models
from keras import layers
from keras import optimizers
from keras import initializers
from keras import regularizers
from keras import Input
import tensorflow as tf
import pandas as pd
from PrepareData import *
import pickle 
from itertools import chain, combinations




def powerset(iterable):
    s = list(iterable)
    a = chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))
    return [list(item) for item in list(a)]


def BlockMLP(size, block_list, features, label, n_layers, n_neurons, p_dropout):
    
    X_train, X_val, y_train, y_val = train_test_split(features,label, test_size=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)
            
    
    filepath = "models/feature_blocks/Best_{}_{}.hdf5".format(size,"_".join(block_list))
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

    tf.random.set_seed(12)
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
    
    model = load_model("models/feature_blocks/Best_{}_{}.hdf5".format(size,"_".join(block_list)))
    
    history.history["test_metrics"] = model.evaluate(X_test,y_test)
    
    print(history.history["test_metrics"])
    
    return history.history


def BlockExperiments(size,k):
    
    exp_dict_ = {}
    
    
    d,l = prepare_with_moats_parallel(data=size,order=k,mode=4)
    feats = []
    blocks = ["D", "NC", "KED", "PS"]
    block_idxs = [0,1,2,3]
    block_cmb = powerset(block_idxs)
    feats.append(d[:,:2])
    feats.append(d[:,2:((2*size) + 2)])
    feats.append(d[:,((2*size)+2):(((k+2)*size)+2)])
    feats.append(d[:,(((k+2)*size)+2):])
    
    
    for item in block_cmb:
        feat_nm = []
        data_blocks = []
        if len(item)==1:
            data = feats[item[0]]
            feat_nm.append(blocks[item[0]])
            print(data.shape,feat_nm)

        if len(item)>1:
            for itm in item:
                data_blocks.append(feats[itm])
                feat_nm.append(blocks[itm])
            data = np.column_stack(data_blocks)
            print(data.shape,feat_nm)
        
        exp_dict_[tuple(feat_nm)] = BlockMLP(size=size, block_list=feat_nm, features=data, label=l, n_layers=3, n_neurons=100, p_dropout=0.0)

        print(exp_dict_)
    
    return exp_dict_


if __name__ == "__main__":
    experiments = {}
    sizes = [20,50,100]
    ks = [5,10]
    for size in sizes:
        for k in ks:
            experiments[(size,k)] = BlockExperiments(size,k)
    
    pickle.dump(experiments, open("models/feature_blocks/experiments.pkl", "wb"))
            
       