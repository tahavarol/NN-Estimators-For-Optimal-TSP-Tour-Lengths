from Nets import RNNSmall, RNNLarge, CNNSmall, CNNLarge
import pickle

if __name__ == "__main__":
    
    altnetdict = {}
    
    sizes = [20,50,100]
    
    k = 10
    
    models = ["RNNSmall", "RNNLarge", "CNNSmall", "CNNLarge"]
    
    for size in sizes:
        for model in models:
            
            key = (model,size)
            print(key)
            if model == "RNNSmall":
                log = RNNSmall(size=size, k=k, mode=4, n_neurons=100, p_dropout=0.01)
            if model == "RNNLarge":
                log = RNNLarge(size=size, k=k, mode=4, n_neurons=100, p_dropout=0.01)
            if model == "CNNSmall":
                log = CNNSmall(size=size, k=k, mode=4, kernel_width=5, num_filters=100, p_dropout=0.01)
            if model == "CNNLarge":
                log = CNNLarge(size=size, k=k, mode=4, kernel_width=5, num_filters=100, p_dropout=0.01)
            
            altnetdict[key] = log
            
    pickle.dump(altnetdict, open("models/hpt/altnetlogs.pkl", "wb"))