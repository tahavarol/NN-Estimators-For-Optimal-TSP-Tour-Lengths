from Nets import *
from MLModels import *
from PrepareData import *
from glob import glob
import pickle
import time


def block_model_config_parser(model_path_str):
    ls = model_path_str.split("/")[-1].split(".")[0].split("_")
    ls.pop(0)
    return ls


def MAPE(predictions, labels):
    
    return abs((predictions/labels)-1).mean()*100


def EvaluateBlockModel(model_path, benchmark_data):
    
    print(model_path)
    config = block_model_config_parser(model_path)
    print(config)
    size = int(config[0])
    k=10
        
    test_data = pd.read_csv("BenchmarkInstances/{}_{}.csv".format(benchmark_data,size),header=None).to_numpy()

    test_data = prepare_without_moats(test_data, order=k, mode=4)

    test_data, test_data_scales = test_data[:,:-1], test_data[:,-1]
    test_data, test_data_labels = test_data[:,:-1], test_data[:,-1]
    
    feats_ls = []
    
    if "D" in config:
        X_D = test_data[:,:2]
        feats_ls.append(X_D)

    if "NC" in config:
        X_NC = test_data[:,2:((2*size)+2)]
        feats_ls.append(X_NC)

    if "KED" in config:
        X_KED = test_data[:,((2*size)+2):]
        feats_ls.append(X_KED)

    if "PS" in config:
        X_PS = pd.read_csv("BenchmarkInstances/Moats_{}_{}.csv".format(benchmark_data,size),header=None).to_numpy()
        X_PS = X_PS / test_data_scales[:,None]
        feats_ls.append(X_PS)
        
    
    model = load_model(model_path)
    model.summary()    
    
    inputs = np.column_stack(feats_ls)
    
    preds = model.predict(inputs).flatten()
    scores = preds/test_data_labels
    eval_score = model.evaluate(inputs,test_data_labels)
    config.append(benchmark_data)
    
    return tuple(config), scores, eval_score


    
if __name__ == "__main__":
    
    nn_models_list = glob("models/feature_blocks/*.hdf5", recursive = True)
    nn_models_list.sort(reverse=True)
    print(nn_models_list)
    instance_types = ["G1", "G2_2", "G3_1", "G3_2", "G3_3", "G4", "SG", "US", "UR", "NS", "NR"]
    
    
    evaluation_dict = {}
    
    for model_path in nn_models_list:
        for benchmark_data in instance_types:
            k,s,e = EvaluateBlockModel(model_path, benchmark_data)
            print(k)
            evaluation_dict[k] = {}
            evaluation_dict[k]["Scores"] = s
            evaluation_dict[k]["MAPE"] = e
            
    print(len(nn_models_list)*len(instance_types), len(evaluation_dict))           
    pickle_out = open("BenchmarkInstances/FeatureBlockBenchmarking.pkl","wb")
    pickle.dump(evaluation_dict, pickle_out)
    