from Nets import LiteratureHybridFCNNSmall, LiteratureHybridFCNNLarge, LiteratureHybridPCNNSmall, LiteratureHybridPCNNLarge, LiteratureMLP
import pickle

if __name__ == "__main__":
    
    litnetdict = {}
    
    sizes = [20,50,100]
    
    k = 10
    
    models = ["FCNNSmallLiterature", "FCNNLargeLiterature", 
              "PCNNSmallLiterature", "PCNNLargeLiterature",
              "Literature"]
    
    for size in sizes:
        for model in models:
            
            key = (model,size)
            print(key)
            if model == "FCNNSmallLiterature":
                log = LiteratureHybridFCNNSmall(size=size, k=k, mode=4, n_layers=3, n_neurons=100, p_dropout=0.01)
            if model == "FCNNLargeLiterature":
                log = LiteratureHybridFCNNLarge(size=size, k=k, mode=4, n_layers=3, n_neurons=100, p_dropout=0.01)
            if model == "PCNNSmallLiterature":
                log = LiteratureHybridPCNNSmall(size=size, k=k, mode=4, n_layers=3, n_neurons=100, p_dropout=0.01)
            if model == "PCNNSmallLiterature":
                log = LiteratureHybridPCNNLarge(size=size, k=k, mode=4, n_layers=3, n_neurons=100, p_dropout=0.01)
            if model == "Literature":
                log = LiteratureMLP(size=size, n_layers=3, n_neurons=100, p_dropout=0.01)


            litnetdict[key] = log
            
    pickle.dump(litnetdict, open("models/hpt/litnetlogs.pkl", "wb"))