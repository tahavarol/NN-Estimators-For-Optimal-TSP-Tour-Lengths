import pickle
import pandas as pd
from Nets import *

sizes = [20,50,100]
ks = [1,2,5,10]
Models = ["FCNNSmall", "PCNNSmall", "FCNNLarge", "PCNNLarge"]
layers = [3,4,5,6]
neurons = [5,25,100]
dropouts = [0.0,0.01,0.1]

print(len(sizes)*len(ks)*len(Models)*len(layers)*len(neurons)*len(dropouts), "Runs")

exp_dict = {}
iternum = 0
for size in sizes:
    for k in ks:
        for model_ in Models:
            if model_ in ["FCNNSmall", "PCNNSmall"]:
                data,label = prepare_without_moats_parallel(data=size, order=k, mode=4)
            else:
                data,label = prepare_with_moats_parallel(data=size, order=k, mode=4)
            for layer in layers:
                for neuron in neurons:
                    for drop in dropouts:
                        key = (size,k,model_,layer,neuron,drop)
                        print(key)
                        iternum = iternum+1
                        print(iternum)
                        if key[2] == "FCNNSmall":
                            exp_dict[key] = FCNNSmall(size=size, data=data, label=label, k=k, mode=4, n_layers=layer, n_neurons=neuron, p_dropout=drop)
                            
                        if key[2] == "PCNNSmall":
                            exp_dict[key] = PCNNSmall(size=size, data=data, label=label, k=k, mode=4, n_layers=layer, n_neurons=neuron, p_dropout=drop)

                        if key[2] == "FCNNLarge":
                            exp_dict[key] = FCNNLarge(size=size, data=data, label=label, k=k, mode=4, n_layers=layer, n_neurons=neuron, p_dropout=drop)

                        if key[2] == "PCNNLarge":
                            exp_dict[key] = PCNNLarge(size=size, data=data, label=label, k=k, mode=4, n_layers=layer, n_neurons=neuron, p_dropout=drop)
                            
                            
                            
pickle.dump(exp_dict, open("models/hpt/exp_logs.pkl", "wb"))


logs = pickle.load(open("models/hpt/exp_logs.pkl", "rb"))

key_list = list(logs.keys())

size_list = [item[0] for item in key_list]
k_list = [item[1] for item in key_list]
model_list = [item[2] for item in key_list]
layers_list = [item[3] for item in key_list]
neurons_list = [item[4] for item in key_list]
dropout_list = [item[5] for item in key_list]
test_mape_list = [logs[item]["test_metrics"][1] for item in key_list]
column_names = ["size","k","model","layers","neurons","dropout","test_mape"]

data = pd.DataFrame([size_list,k_list,model_list,layers_list,neurons_list,dropout_list,test_mape_list]) 
data = data.transpose()
data.columns=column_names

data.to_csv("models/hpt/nn_models.csv",index=False)

sizes = [20,50,100]
Models = ["FCNNSmall", "PCNNSmall", "FCNNLarge", "PCNNLarge"]

param_list = []

for size in sizes:
    for model in Models:
        hp = data[data["test_mape"]==data.loc[(data['size'] == size) & (data['model'] == model)]["test_mape"].min()].to_dict('records')[0]
        param_list.append(hp)

for hp in param_list:
    
    if hp["model"] == "FCNNSmall":
        data,label = prepare_without_moats_parallel(data=hp["size"], order=hp["k"], mode=4)
        lgs = FCNNSmall(size=hp["size"], data=data, label=label, k=hp["k"], mode=4, n_layers=hp["layers"], n_neurons=hp["neurons"], p_dropout=hp["dropout"])

    elif hp["model"] == "FCNNLarge":
        data,label = prepare_with_moats_parallel(data=hp["size"], order=hp["k"], mode=4)
        lgs = FCNNLarge(size=hp["size"], data=data, label=label, k=hp["k"], mode=4, n_layers=hp["layers"], n_neurons=hp["neurons"], p_dropout=hp["dropout"])

    elif hp["model"] == "PCNNSmall":
        data,label = prepare_without_moats_parallel(data=hp["size"], order=hp["k"], mode=4)
        lgs = PCNNSmall(size=hp["size"], data=data, label=label, k=hp["k"], mode=4, n_layers=hp["layers"], n_neurons=hp["neurons"], p_dropout=hp["dropout"])

    elif hp["model"] == "PCNNLarge":
        data,label = prepare_with_moats_parallel(data=hp["size"], order=hp["k"], mode=4)
        lgs = PCNNLarge(size=hp["size"], data=data, label=label, k=hp["k"], mode=4, n_layers=hp["layers"], n_neurons=hp["neurons"], p_dropout=hp["dropout"])







                        