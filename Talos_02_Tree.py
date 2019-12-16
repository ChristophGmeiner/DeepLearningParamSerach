import numpy as np
import pickle
import os
import shutil
from keras import Sequential
from keras import layers
from keras.layers import Dense, Dropout
from keras.optimizers import Adamax, Adam, Nadam, Adadelta, Adagrad
from keras.activations import relu, sigmoid
from keras.callbacks import EarlyStopping

import talos
from talos.utils import hidden_layers
from talos.model.normalizers import lr_normalizer

import warnings
warnings.filterwarnings('ignore')

path = "/home/ubuntu/G/"
with open(path + "01_Data", "rb") as datafile:
    data = pickle.load(datafile)
    
X_df_train = data[0]
y_df_train = data[1]
X_df_test = data[2]
y_df_test = data[3]
PredData = data[4]
PredData1_binary = data[6]
baseseed = data[8]

datafile.close()

dropcols = ['ATC_1', 'ATC_2', 'ATC_3', 'ATC_4', 'ATC_5', 'ATC_6', 'ATC_7', 'ATC_8']
X_df_train.drop(dropcols, axis=1, inplace=True)
X_df_test.drop(dropcols, axis=1, inplace=True)
PredData.drop(dropcols, axis=1, inplace=True)

from sklearn.model_selection import train_test_split
X_df_train2, X_df_val, y_df_train2, y_df_val = train_test_split(X_df_train, y_df_train, test_size=0.33,
                                                                  random_state=baseseed, shuffle=True)

p = {'optimizer': [Adamax, Nadam, Adagrad, Adadelta],
     'first_neuron': [5120, 1024, 512, 128],
     'batch_size': [100, 500, 1000, 2500, 5000, 10000],
     'epochs': [45],
     'hidden_layers':[0, 1, 2, 3, 4],
     'kernel_initializer': ['uniform', 'normal'],
     'dropout': [0.0, 0.25, 0.5, 0.65, 2, 3, 4, 5, 10],
     'losses': ["mse"],
     'shapes': ['brick', 'triangle', 'funnel'],
     'activation': ['relu'],
     'lr': list(np.linspace(0, 1, 10))
    }

def fcmodel(X_train, y_train, x_val, y_val, params):
    
    model = Sequential()
    
    esr = 2
    callbacks = [EarlyStopping(monitor="val_mean_absolute_error", patience=esr, mode="min", 
                               min_delta = 0.001, baseline=0.1)]
    
    model.add(Dense(params['first_neuron'], input_dim=X_train.shape[1],
                    activation="relu",
                    kernel_initializer = params['kernel_initializer'] ))
    model.add(Dropout(params['dropout']))
    
    ## hidden layers
    hidden_layers(model, params, 1)
    
    model.add(Dense(1, activation="sigmoid",
                    kernel_initializer=params['kernel_initializer']))
    
    model.compile(loss=params["losses"], 
                  optimizer=params['optimizer'](lr_normalizer(params['lr'], 
                                                params['optimizer'])),
                  metrics=['mae'])
    
    history = model.fit(X_train, y_train, 
                        validation_data=[x_val, y_val],
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        callbacks=callbacks,
                        verbose=2)
    return history, model


proj = "Talos02Tree_1"
path2 = path + "LeNA_2/TalosTest/" + proj + "/"

if os.path.exists(path2):
    shutil.rmtree(path2)

t = talos.Scan(x=X_df_train2.values,
            y=y_df_train2.values,
            model=fcmodel,
            params=p,
            x_val=X_df_val.values,
            y_val=y_df_val.values,
            seed=baseseed,
            experiment_name=proj,
            print_params=True,
            round_limit=100,
            reduction_method="tree",
            reduction_interval= 20,
            reduction_window=20,
            reduction_threshold=0.2,
            reduction_metric='mean_average_error',
            minimize_loss=True)


anafilename = os.listdir(path2)[0]
anafile = path2 + anafilename
r = talos.Reporting(anafile)

and_df = r.data
anadf = r.data.sort_values("val_mean_absolute_error")

path3 = path + "LeNA_2/TalosTest/"

if os.path.exists(path3 + proj + "_dep.zip"):
    os.remove(path3 + proj + "_dep.zip")
    
if os.path.exists(path3 + proj + "_dep"):
    shutil.rmtree(path3 + proj + "_dep")
    
td = talos.Deploy(t, proj + "_dep", metric="val_mean_absolute_error", asc=True)

path_dep = path + "LeNA_2/TalosTest/"
talosfileslist = os.listdir(path_dep)
talosfileslist = [x for x in talosfileslist if x.find(proj + "_dep.zip") != -1]
talosfile = talosfileslist[0]

tr = talos.Restore(talosfile)

e = talos.Evaluate(t)

a = e.evaluate(X_df_test.values, y_df_test.values, metric="mae",
           task="continuous", folds=5, model_id=anadf.val_mean_absolute_error.idxmin())

print(a)
