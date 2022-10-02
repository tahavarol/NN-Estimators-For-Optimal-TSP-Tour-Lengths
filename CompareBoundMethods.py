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
m = gp.Model("matrix1")
m.Params.OutputFlag = 0
from itertools import repeat
import itertools
from matplotlib import pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=False)
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree


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


'''1-tree relaxation of TSP'''

def one_tree_relaxation_TSP(data):
    size = int(data.shape[0]/2)
    X = np.column_stack((np.array(data[0:size]), data[size:(2*size)]))
    X = squareform(pdist(X))
    arr1 = np.sort(X[0,1:])
    e2 = sum(arr1[:2])
    X1 = X[1:,:]
    X2 = X1[:,1:]
    edge_list = minimum_spanning_tree(X2)
    e = edge_list.toarray()
    return e.sum() + e2

def one_tree_relaxation_TSP_batch(data):
    
    size = int(data.shape[1]/2)
    soln=[]
    
    for i in tqdm(range(data.shape[0])):
        s = one_tree_relaxation_TSP(data[i,:])
        soln.append(s)
    
    
    return np.array(soln)    


def ComputeMoatGap(size, itype):
    
    load_path = "BenchmarkInstances/"
    
    data = pd.read_csv(load_path+ itype + "_" + str(size)+".csv", header=None).to_numpy()
    coords = data[:,:2*size]
    moats = moat_dual_new_batch_gurobi(coords)
    return (1-(moats[:,-1]/data[:,-1]))*100

def Compute1TreeGap(size, itype):
    
    load_path = "BenchmarkInstances/"
    
    data = pd.read_csv(load_path+ itype + "_" + str(size)+".csv", header=None).to_numpy()
    coords = data[:,:2*size]
    tree_lb = one_tree_relaxation_TSP_batch(coords)
    
    return (1-(tree_lb/data[:,-1]))*100


def ExportGapPlots(gap_dict):
    
    ax = pd.DataFrame(gap_dict).plot(kind='box',
             color=dict(boxes='r', whiskers='r', medians='r', caps='r'),
             boxprops=dict(linestyle='-', linewidth=1.5),
             flierprops=dict(linestyle='-', linewidth=1.5),
             medianprops=dict(linestyle='-', linewidth=1.5, color='black'),
             whiskerprops=dict(linestyle='-', linewidth=1.5),
             capprops=dict(linestyle='-', linewidth=1.5),
             showfliers=False, grid=True, rot=0, vert=False, fontsize=12, figsize=(10,6))

    ax.set_xlabel('% Gap To Optimality', fontsize=12)
    ax.set_ylabel('(Instance Size, Morphology Type, Method)', fontsize=12)
    plt.savefig('LowerBoundMethodsPlots/Hardness.jpg',dpi=250,bbox_inches='tight',transparent=False)
    plt.show()
    


gap_dict = {}

sizes = [20,50,100]
types = ["US", "NR"]
pathology = ["S", "P"]

    
for size in sizes:
    for itype in types:
        key1 = "({},{},Dendrogram)".format(size,pathology[types.index(itype)])
        key2 = "({},{},1-Tree)".format(size,pathology[types.index(itype)])
        gap_dict[key1] = ComputeMoatGap(size, itype)
        gap_dict[key2] = Compute1TreeGap(size, itype)


ExportGapPlots(gap_dict)




