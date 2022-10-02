import numpy as np
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm
import copy
import os
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial import distance
import gurobipy as gp
from gurobipy import GRB
from itertools import repeat
import itertools
import time


    
def moat_dual_new_gurobi(data,combs,mempty_array):
    size = int(data.shape[0]/2)
    #combs = list(itertools.combinations(list(range(size)),2))
    data=data.reshape(1,2*size)

    X = np.column_stack((np.array(data[0,0:size]), data[0,size:(2*size)]))
    Z = linkage(np.column_stack((np.array(data[0,0:size]), data[0,size:(2*size)])), 'ward')
    d_ij = list(distance.pdist(X, "euclidean"))
    my_sets = dict.fromkeys(list(range(2*size-2)))

    for i in range(size):
        my_sets[i] = {i}

    for i in range(Z.shape[0]-1):
        my_sets[i+size] = set.union(my_sets[int(Z[i,0])],my_sets[int(Z[i,1])])


    [my_sets.pop(i, None) for i in list(range(size))]
    empty_array = mempty_array * 1

    for key in my_sets:
        empty_array.append(list(map(lambda my_arr : 1 if len(my_arr.intersection(my_sets[key]))==1 else 0, combs)))
    
    empty_array = np.array(empty_array)
    empty_array.shape
    m = gp.Model("matrix1")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1 
    x = m.addMVar(shape=len(combs), vtype=GRB.CONTINUOUS, name="xij")
    m.setObjective(np.array(d_ij) @ x, GRB.MINIMIZE)
    m.addConstr(empty_array @ x >= 1)
    m.optimize()
    
    cstr = m.getConstrs()
    soln = []
    for i in range(2*(size-1)):
        soln.append(cstr[i].Pi)
    soln.append(sum(soln)*2)
    
    
    #return m.objVal*2
    return np.array(soln)

def moat_dual_new_batch_gurobi(data):
    
    size = int(data.shape[1]/2)
    combs = list(itertools.combinations(list(range(size)),2))
    empty_array = np.zeros((size, (size**2-size)//2),dtype=np.int8)
    for (i,item) in zip(range(len(combs)),combs):
        empty_array[item,i] = 1
    combs = list(map(set, combs))
    empty_array = empty_array.tolist()


    
    soln=[]
    
    for i in tqdm(range(data.shape[0])):
        s = moat_dual_new_gurobi(data[i,:],combs,empty_array)
        soln.append(s)
    
    
    return np.array(soln)    

def moat_dual_new_batch_gurobi_parallel(data):
    
    cores=mp.cpu_count()
    
    
    pool = Pool(cores)
    
    df_split = np.array_split(data, cores, axis=0)
    
    df_out = np.vstack(pool.map(moat_dual_new_batch_gurobi, df_split))
        
    pool.close()
    pool.join()
    pool.clear()
   
    return df_out    





def ComputeEvalMoats(itype,size):
    
    load_path = "BenchmarkInstances/"
    save_path = "BenchmarkInstances/"
    
    data = pd.read_csv(load_path + itype + "_" + str(size) + ".csv", header=None).to_numpy()
    coords = data[:,:2*size]
    moats = moat_dual_new_batch_gurobi(coords)
    np.savetxt(save_path + "Moats" + "_" + itype + "_" + str(size) + ".csv", moats, delimiter=',')
    
    
    
if __name__ == "__main__":
    instance_names = ["G1", "G2_2", "G3_1", "G3_2", "G3_3", "G4", "SG", "US", "UR", "NS", "NR"]
    sizes = [20,50,100]
    for size in sizes:
        for instance in instance_names:
            print(size,instance)
            s = time.time()
            ComputeEvalMoats(instance,size)
            print(time.time()-s, "seconds passed for ", str(size),instance," type")