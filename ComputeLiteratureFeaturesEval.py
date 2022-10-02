from LiteratureFeaturesTraining import *
import time

def ComputeEvalLiteratureFeatures(itype, size):
    
    load_path = "BenchmarkInstances/"
    save_path = "BenchmarkInstances/"
    
    data = pd.read_csv(load_path + itype + "_" + str(size) + ".csv", header=None).to_numpy()
    
    coords = data[:,:2*size]
    
    litfeats = LiteratureFeaturesBatch(coords)

    np.savetxt(save_path + "LiteratureFeatures" + "_" + itype + "_" + str(size) + ".csv", litfeats, delimiter=',')
    
    
    
if __name__ == "__main__":
    instance_names = ["G1", "G2_2", "G3_1", "G3_2", "G3_3", "G4", "SG", "US", "UR", "NS", "NR"]
    sizes = [20,50,100]
    for size in sizes:
        for instance in instance_names:
            print(size,instance)
            s = time.time()
            ComputeEvalLiteratureFeatures(instance,size)
            print(time.time()-s, "seconds passed for ", str(size),instance," type")