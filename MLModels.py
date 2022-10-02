from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from PrepareData import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import pickle
import pandas as pd


def LR(data,label):
    
    X_train, X_val, y_train, y_val = train_test_split(data,label, test_size=0.1, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    LRmodel = LinearRegression(n_jobs=-1)
    
    LRmodel.fit(X_train, y_train)
    
    
    
    details = {}
    
    TrainR2 = LRmodel.score(X_train, y_train)
    TestR2 = LRmodel.score(X_test, y_test)
    
    p = LRmodel.get_params()
    
    details["parameters"] = p
    details["TrainR2"] = TrainR2
    details["TestR2"] = TestR2
    
    preds = LRmodel.predict(X_train).flatten()
    details["Train_mape"] = mean_absolute_percentage_error(y_train.flatten(), preds)*100
    preds = LRmodel.predict(X_test).flatten()
    details["Test_mape"] = mean_absolute_percentage_error(y_test.flatten(), preds)*100
    
    return details, LRmodel


def GBM(data,label):
    
    X_train, X_val, y_train, y_val = train_test_split(data,label, test_size=0.1, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    GBMmodel = HistGradientBoostingRegressor(random_state=1)
    
    GBMmodel.fit(X_train, y_train)
    
    details = {}
    
    p = GBMmodel.get_params()
    
    details["parameters"] = p
    
    preds = GBMmodel.predict(X_train).flatten()
    details["Train_mape"] = mean_absolute_percentage_error(y_train.flatten(), preds)*100
    
    preds = GBMmodel.predict(X_test).flatten()
    details["Test_mape"] = mean_absolute_percentage_error(y_test.flatten(), preds)*100

    return details, GBMmodel    


def DT(data,label):
    
    X_train, X_val, y_train, y_val = train_test_split(data,label, test_size=0.1, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    DTmodel = DecisionTreeRegressor(random_state=1)
    
    DTmodel.fit(X_train, y_train)
    
    details = {}
    
    p = DTmodel.get_params()
    
    details["parameters"] = p
    
    preds = DTmodel.predict(X_train).flatten()
    details["Train_mape"] = mean_absolute_percentage_error(y_train.flatten(), preds)*100
    
    preds = DTmodel.predict(X_test).flatten()
    details["Test_mape"] = mean_absolute_percentage_error(y_test.flatten(), preds)*100

    return details, DTmodel     



if __name__ == "__main__":
    
    sizes = [20,50,100]
    ks = [1,2,5,10]
    features = ["Small", "Large"]
    models = ["LR", "GBM", "DT"]

    log_dict = {}
    iternum = 0

    for size in sizes:
        for k in ks:
            for feature in features:

                if feature == "Small":
                    data,label = prepare_without_moats_parallel(data=size, order=k, mode=4)

                if feature == "Large":
                    data,label = prepare_with_moats_parallel(data=size, order=k, mode=4)

                for model in models:

                    key = (model,feature,size,k)
                    print(key)
                    iternum = iternum + 1 
                    print(iternum)

                    if model == "LR":
                        log_dict[key],_ = LR(data,label)
                        print(key, " is finished!")
                        print(log_dict[key])
                    if model == "GBM":
                        log_dict[key],_ = GBM(data,label)
                        print(key, " is finished!")
                        print(log_dict[key])
                    if model == "DT":
                        log_dict[key],_ = DT(data,label)
                        print(key, " is finished!")
                        print(log_dict[key])

    pickle.dump(log_dict, open("models/hpt/ml_logs.pkl", "wb"))
    
    logs = pickle.load(open("models/hpt/ml_logs.pkl", "rb"))

    key_list = list(logs.keys())
    
    model_list = [item[0] for item in key_list]
    features_list = [item[1] for item in key_list]
    size_list = [item[2] for item in key_list]
    k_list = [item[3] for item in key_list]
    test_mape_list = [logs[item]["Test_mape"] for item in key_list]
    column_names = ["model","features","size","k","test_mape"]
    
    
    data = pd.DataFrame([model_list,features_list,size_list,k_list,test_mape_list]) 
    data = data.transpose()
    data.columns=column_names
    data.to_csv("models/hpt/ml_models.csv",index=False)
    
    sizes = [20,50,100]
    Models = ["LR", "DT", "GBM"]
    features = ["Small", "Large"]

    param_list = []

    for size in sizes:
        for model in Models:
            for feats in features:
                hp = data[data["test_mape"]==data.loc[(data['size'] == size) & (data['model'] == model) & (data['features'] == feats)]["test_mape"].min()].to_dict('records')[0]
                param_list.append(hp)
                
    
    
    for hp in param_list:
    
        if hp["features"] == "Small":
            data, label = prepare_without_moats_parallel(data=hp["size"], order=hp["k"], mode=4)

        elif hp["features"] == "Large":
            data, label = prepare_with_moats_parallel(data=hp["size"], order=hp["k"], mode=4)

        if hp["model"] == "LR":
            _, md = LR(data,label)
            print(_)
            filename = 'models/hpt/'+hp["model"]+hp["features"]+"Best"+"_"+str(hp["size"])+"_"+str(hp["k"])+".sav"
            pickle.dump(md, open(filename, 'wb'))

        elif hp["model"] == "GBM":
            _, md = GBM(data,label)
            print(_)
            filename = 'models/hpt/'+hp["model"]+hp["features"]+"Best"+"_"+str(hp["size"])+"_"+str(hp["k"])+".sav"
            pickle.dump(md, open(filename, 'wb'))

        elif hp["model"] == "DT":
            _, md = DT(data,label)
            print(_)
            filename = 'models/hpt/'+hp["model"]+hp["features"]+"Best"+"_"+str(hp["size"])+"_"+str(hp["k"])+".sav"
            pickle.dump(md, open(filename, 'wb'))




    