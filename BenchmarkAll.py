from Nets import *
from MLModels import *
from PrepareData import *
from glob import glob
import pickle
from Nicola_CS_Kool import *
import time

def model_config_parser(model_path_str):
    
    return model_path_str.split("/")[-1].split(".")[0].split("_")[0],int(model_path_str.split("/")[-1].split(".")[0].split("_")[1]),int(model_path_str.split("/")[-1].split(".")[0].split("_")[2])


def MAPE(predictions, labels):
    
    return abs((predictions/labels)-1).mean()*100




def EvaluateModel(model_path, benchmark_data):
    
    try:
        print(model_path)
        name, size, k = model_config_parser(model_path)
        print(name, size, k)
        
        test_data = pd.read_csv("BenchmarkInstances/{}_{}.csv".format(benchmark_data,size),header=None).to_numpy()

        test_data = prepare_without_moats(test_data, order=k, mode=4)

        test_data, test_data_scales = test_data[:,:-1], test_data[:,-1]
        test_data, test_data_labels = test_data[:,:-1], test_data[:,-1]




        X_D = test_data[:,:2]
        X_NC = test_data[:,2:((2*size)+2)]
        X_KED = test_data[:,((2*size)+2):]

        print(X_D.shape,X_NC.shape,X_KED.shape)

        if "Large" in name:
            X_PS = pd.read_csv("BenchmarkInstances/Moats_{}_{}.csv".format(benchmark_data,size),header=None).to_numpy()
            X_PS = X_PS / test_data_scales[:,None]
            print(X_D.shape,X_NC.shape,X_KED.shape,X_PS.shape)

        if "Literature" in name:
            X_LITF = pd.read_csv("BenchmarkInstances/LiteratureFeatures_{}_{}.csv".format(benchmark_data,size),header=None).to_numpy()
            colms = list(range(36))
            fix = [0,22,23,24,25,26,27,35]
            scl = list(set(colms)-set(fix))
            X_LITF[:,scl] = X_LITF[:,scl] / test_data_scales[:,None]




        if "FCNNSmall" in name and "Literature" not in name:
            model = load_model(model_path)
            model.summary()
            inputs = np.column_stack((X_D,X_NC,X_KED))
            preds = model.predict(inputs).flatten()
            scores = preds/test_data_labels
            eval_score = model.evaluate(inputs,test_data_labels)

        if "FCNNSmallLiterature" in name:
            model = load_model(model_path)
            model.summary()
            inputs = np.column_stack((X_D,X_NC,X_KED,X_LITF))
            preds = model.predict(inputs).flatten()
            scores = preds/test_data_labels
            eval_score = model.evaluate(inputs,test_data_labels)


        if "FCNNLarge" in name and "Literature" not in name:
            model = load_model(model_path)
            model.summary()
            inputs = np.column_stack((X_D,X_NC,X_KED,X_PS))
            preds = model.predict(inputs).flatten()
            scores = preds/test_data_labels
            eval_score = model.evaluate(inputs,test_data_labels)

        if "FCNNLargeLiterature" in name:
            model = load_model(model_path)
            model.summary()
            inputs = np.column_stack((X_D,X_NC,X_KED,X_PS,X_LITF))
            preds = model.predict(inputs).flatten()
            scores = preds/test_data_labels
            eval_score = model.evaluate(inputs,test_data_labels)






        if "PCNNSmall" in name and "Literature" not in name:
            model = load_model(model_path)
            model.summary()
            inputs = [X_D,X_NC,X_KED]
            preds = model.predict(inputs).flatten()
            scores = preds/test_data_labels
            eval_score = model.evaluate(inputs,test_data_labels)

        if "PCNNSmallLiterature" in name:
            model = load_model(model_path)
            model.summary()
            inputs = [X_D,X_NC,X_KED,X_LITF]
            preds = model.predict(inputs).flatten()
            scores = preds/test_data_labels
            eval_score = model.evaluate(inputs,test_data_labels)


        if "PCNNLarge" in name and "Literature" not in name:
            model = load_model(model_path)
            model.summary()
            inputs = [X_D,X_NC,X_KED,X_PS]
            preds = model.predict(inputs).flatten()
            scores = preds/test_data_labels
            eval_score = model.evaluate(inputs,test_data_labels)

        if "PCNNLargeLiterature" in name:
            model = load_model(model_path)
            model.summary()
            inputs = [X_D,X_NC,X_KED,X_PS,X_LITF]
            preds = model.predict(inputs).flatten()
            scores = preds/test_data_labels
            eval_score = model.evaluate(inputs,test_data_labels)





        if "RNNSmall" in name:
            model = load_model(model_path)
            model.summary()
            inputs = np.column_stack((X_D,X_NC,X_KED))
            inputs = np.expand_dims(inputs, axis=1)
            preds = model.predict(inputs).flatten()
            scores = preds/test_data_labels
            eval_score = model.evaluate(inputs,test_data_labels)

        if "RNNLarge" in name:
            model = load_model(model_path)
            model.summary()
            inputs = np.column_stack((X_D,X_NC,X_KED,X_PS))
            inputs = np.expand_dims(inputs, axis=1)
            preds = model.predict(inputs).flatten()
            scores = preds/test_data_labels
            eval_score = model.evaluate(inputs,test_data_labels)

        if "CNNSmall" in name and "PCNNSmall" not in name and "FCNNSmall" not in name:
            model = load_model(model_path)
            model.summary()
            inputs = np.column_stack((X_D,X_NC,X_KED))
            inputs = np.expand_dims(inputs, axis=1)
            inputs = np.expand_dims(inputs, axis=3)
            preds = model.predict(inputs).flatten()
            scores = preds/test_data_labels
            eval_score = model.evaluate(inputs,test_data_labels)

        if "CNNLarge" in name and "PCNNLarge" not in name and "FCNNLarge" not in name:
            model = load_model(model_path)
            model.summary()
            inputs = np.column_stack((X_D,X_NC,X_KED,X_PS))
            inputs = np.expand_dims(inputs, axis=1)
            inputs = np.expand_dims(inputs, axis=3)
            preds = model.predict(inputs).flatten()
            scores = preds/test_data_labels
            eval_score = model.evaluate(inputs,test_data_labels)








        if "LRSmall" in name:
            model = pickle.load(open(model_path, 'rb'))
            inputs = np.column_stack((X_D,X_NC,X_KED))
            preds = model.predict(inputs).flatten()
            scores = preds/test_data_labels
            eval_score = MAPE(preds,test_data_labels)

        if "LRLarge" in name:
            model = pickle.load(open(model_path, 'rb'))
            inputs = np.column_stack((X_D,X_NC,X_KED,X_PS))
            preds = model.predict(inputs).flatten()
            scores = preds/test_data_labels
            eval_score = MAPE(preds,test_data_labels)

        if "DTSmall" in name:
            model = pickle.load(open(model_path, 'rb'))
            inputs = np.column_stack((X_D,X_NC,X_KED))
            preds = model.predict(inputs).flatten()
            scores = preds/test_data_labels
            eval_score = MAPE(preds,test_data_labels)

        if "DTLarge" in name:
            model = pickle.load(open(model_path, 'rb'))
            inputs = np.column_stack((X_D,X_NC,X_KED,X_PS))
            preds = model.predict(inputs).flatten()
            scores = preds/test_data_labels
            eval_score = MAPE(preds,test_data_labels)

        if "GBMSmall" in name:
            model = pickle.load(open(model_path, 'rb'))
            inputs = np.column_stack((X_D,X_NC,X_KED))
            preds = model.predict(inputs).flatten()
            scores = preds/test_data_labels
            eval_score = MAPE(preds,test_data_labels)

        if "GBMLarge" in name:
            model = pickle.load(open(model_path, 'rb'))
            inputs = np.column_stack((X_D,X_NC,X_KED,X_PS))
            preds = model.predict(inputs).flatten()
            scores = preds/test_data_labels
            eval_score = MAPE(preds,test_data_labels)

    except:
        print(model_path)
        name, size = model_path.split("/")[-1].split(".")[0].split("_")[0], int(model_path.split("/")[-1].split(".")[0].split("_")[1])
        if "Universal" in name:
            model = load_model(model_path)
            model.summary()
            test_data = pd.read_csv("BenchmarkInstances/{}_{}.csv".format(benchmark_data,size),header=None).to_numpy()
            inputs, test_data_labels = test_data[:,:(2*size)], test_data[:,-1]
            preds = model.predict(inputs).flatten()
            scores = preds/test_data_labels
            eval_score = model.evaluate(inputs,test_data_labels)
            
        if "LiteratureBest" in name:
            model = load_model(model_path)
            model.summary()
            inputs = pd.read_csv("BenchmarkInstances/LiteratureFeatures_{}_{}.csv".format(benchmark_data,size),header=None).to_numpy()
            test_data_labels = pd.read_csv("BenchmarkInstances/{}_{}.csv".format(benchmark_data,size),header=None).to_numpy()[:,-1]
            preds = model.predict(inputs).flatten()
            scores = preds/test_data_labels
            eval_score = model.evaluate(inputs,test_data_labels)

    
    key = name, size, benchmark_data
    print(eval_score)
    
    return key, scores, eval_score
    
    

        
    
if __name__ == "__main__":
    
    nn_models_list = glob("models/hpt/*.hdf5", recursive = True)
    ml_models_list = glob("models/hpt/*.sav", recursive = True)
    models_list = nn_models_list + ml_models_list
    
    sizes = [20,50,100]
    instance_types = ["G1", "G2_2", "G3_1", "G3_2", "G3_3", "G4", "SG", "US", "UR", "NS", "NR"]
    
    print("total number of benchmarking scenarios is:", len(models_list)*len(instance_types))
    
    evaluation_dict = {}
    
    for model_path in models_list:
        for benchmark_data in instance_types:
            k,s,e = EvaluateModel(model_path, benchmark_data)
            print(k)
            evaluation_dict[k] = {}
            evaluation_dict[k]["Scores"] = s
            evaluation_dict[k]["MAPE"] = e
            
                
    for size in sizes:
        for benchmark_data in instance_types:
            k,s,e = cavdar_sokol_batch(benchmark_data,size)
            print(k)
            evaluation_dict[k] = {}
            evaluation_dict[k]["Scores"] = s
            evaluation_dict[k]["MAPE"] = e
            
    for size in sizes:
        for benchmark_data in instance_types:
            k,s,e = NicolaRC_batch(benchmark_data,size)
            print(k)
            evaluation_dict[k] = {}
            evaluation_dict[k]["Scores"] = s
            evaluation_dict[k]["MAPE"] = e
           
    for size in sizes:
        for benchmark_data in instance_types:
            k,s,e = estimate_tour_lengths_rl(benchmark_data,size)
            print(k)
            evaluation_dict[k] = {}
            evaluation_dict[k]["Scores"] = s
            evaluation_dict[k]["MAPE"] = e
            

            
    pickle_out = open("BenchmarkInstances/ExhaustiveBenchmarking.pkl","wb")
    pickle.dump(evaluation_dict, pickle_out)
    pickle_out.close()