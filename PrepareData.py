import numpy as np
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm
import copy
import os
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

def cast_instance(minstance):
    
    instance = copy.deepcopy(minstance)
    
    size = int(instance.shape[0]/2)
    
    ll_x = np.min(instance[:size])
    ur_x = np.max(instance[:size])
    width = ur_x - ll_x
    
    ll_y = np.min(instance[size:])
    ur_y = np.max(instance[size:])
    height = ur_y - ll_y
    
    scale_ratio = max(width,height)
    

    instance[:size] = instance[:size] - ll_x
    instance[size:] = instance[size:] - ll_y
    instance = instance/scale_ratio
    
    return np.array(instance), scale_ratio

def cast_dataset(mdata):
    
    data = copy.deepcopy(mdata)
    size = int((data.shape[1])/2)
    
    scales = []
    for i in range(data.shape[0]):
        data[i],s = cast_instance(data[i])
        scales.append(s)

    return data, np.array(scales)
    
def cast_dataset_parallel(mdata):
    
    cores=mp.cpu_count()
    pool = Pool(cores)
    
    df_split = np.array_split(data, cores, axis=0)
    df_out, _ = np.vstack(pool.map(cast_dataset, df_split))
        
    pool.close()
    pool.join()
    pool.clear()
   
    return df_out, np.array(_)     
        



def closest_points(data, order):
    
    distances = []
    size = int(data.shape[1]/2)
    
    for i in range(data.shape[0]):
        
        ins = np.column_stack((np.array(data[i,0:size]), data[i,size:(2*size)]))
        
        b = euclidean_distances(ins, ins)
        
        np.fill_diagonal(b, -100)
        
        my_dist=[]
        
        for j in range(size):
            
            closest_pts = np.sort(b[j,:][b[j,:] != -100])[:order].tolist()
            
            my_dist = my_dist + closest_pts
        
        distances.append(my_dist)
        
    distances = np.array(distances)
    
    return distances


def closest_points_parallel(data, order):
    
    cores=mp.cpu_count()
    
    pool = Pool(cores)
    
    df_split = np.array_split(data, cores, axis=0)
    orders = [order]*cores
    
    df_out = np.vstack(pool.map(closest_points, df_split, orders))
        
    pool.close()
    pool.join()
    pool.clear()
   
    return df_out


# this is where we calculate the tightest frame dependent features namely width and height of the tightest rectangle 
def dimensions(data):
    size = int(data.shape[1]/2)
    width = []
    height = []
    
    for i in range(data.shape[0]):
        a = data[i,0:size].min()
        b = data[i,0:size].max()
        x = float(b - a)
        c = data[i,size:(2*size)].min()
        d = data[i,size:(2*size)].max()
        y = float(d - c)
        width.append(x)
        height.append(y)
    
    
    width = np.array(width)
    height = np.array(height)
    
    return np.column_stack((width,height))




# this function normalizes the coordinates in 5 modes: 
#1:with respect to bottom left corner of the tightest rectange, 
#2:randomly picking one node and normalizing the rest 
#3:randomly picking one node and normalizing the rest and take the absolute value
#4: normalizing coordinates wrt to the left most point
#5: based on the node in the first index

def normalize_coords(data,mode=4):
    
    
    size = int(data.shape[1]/2)
    
    if mode == 1:
        
        for i in range(data.shape[0]):
            x = data[i,0:size].min()
            y = data[i,size:(2*size)].min()
            data[i,0:size] = data[i,0:size] - x
            data[i,size:(2*size)] = data[i,size:(2*size)] - y
    
    if mode == 2:
        
        for i in range(data.shape[0]):
            pt = int(np.random.randint(0, size, 1))
            x = data[i,pt]
            y = data[i,(size + pt)]
            data[i,0:size] = data[i,0:size] - x
            data[i,size:(2*size)] = data[i,size:(2*size)] - y            
    
    if mode == 3:
        
        for i in range(data.shape[0]):
            pt = int(np.random.randint(0, size, 1))
            x = data[i,pt]
            y = data[i,(size + pt)]
            data[i,0:size] = abs(data[i,0:size] - x)
            data[i,size:(2*size)] = abs(data[i,size:(2*size)] - y)
            
    if mode == 4:
        
        for i in range(data.shape[0]):
            pt = np.argmin(data[i,0:size])
            x = data[i,pt]
            y = data[i,(size + pt)]
            data[i,0:size] = abs(data[i,0:size] - x)
            data[i,size:(2*size)] = abs(data[i,size:(2*size)] - y)
            
    if mode == 5:
        
        for i in range(data.shape[0]):
            pt = 0
            x = data[i,pt]
            y = data[i,(size + pt)]
            data[i,0:size] = abs(data[i,0:size] - x)
            data[i,size:(2*size)] = abs(data[i,size:(2*size)] - y)
            
    
    
    return data




def normalize_coords_parallel(data, mode):
    
    cores=mp.cpu_count()
    
    pool = Pool(cores)
    
    df_split = np.array_split(data, cores, axis=0)
    modes = [mode]*cores
    df_out = np.vstack(pool.map(normalize_coords, df_split, modes))
        
    pool.close()
    pool.join()
    pool.clear()
   
    return df_out

def load_raw(size):
    
    load_path = "TrainingData/Raw/"
    
    mdata = pd.read_csv(load_path + str(size)+ ".csv", header=None).to_numpy()

    return mdata

def load_moats(size):
    
    load_path = "TrainingData/Moats/Moats_"
    
    mdata = pd.read_csv(load_path + str(size)+ ".csv", header=None).to_numpy()

    return mdata

def load_litf(size):
    
    load_path = "TrainingData/LiteratureFeatures/LiteratureFeatures_"
    
    mdata = pd.read_csv(load_path + str(size)+ ".csv", header=None).to_numpy()

    return mdata



def prepare_without_moats(mdata, order=5, mode=4):
    
    size = (mdata.shape[1]-1)//3
    label = mdata[:,-1]
    data = mdata[:,:2*size]
    coords,scales = cast_dataset(data)
    dims = dimensions(coords)
    closestpts = closest_points(coords,order)
    coords = normalize_coords(coords,mode)

    return np.column_stack((dims,coords,closestpts,label/scales,scales))


def prepare_without_moats_parallel(data, order=5, mode=4):
    
    cores = mp.cpu_count()
    pool = Pool(cores)
    
    data = load_raw(data)
    
    orders = [order]*cores
    modes = [mode]*cores
    
    df_split = np.array_split(data, cores, axis=0)
    
    df_out = np.vstack(pool.map(prepare_without_moats, df_split, orders, modes))
        
    pool.close()
    pool.join()
    pool.clear()
    
    df_out = df_out[:,:-1]
    
    return df_out[:,:-1], df_out[:,-1]


def prepare_with_moats_parallel(data, order=5, mode=4):
    
    cores = mp.cpu_count()
    pool = Pool(cores)
    
    moats = load_moats(data)
    data = load_raw(data)
    
    
    orders = [order]*cores
    modes = [mode]*cores
    
    df_split = np.array_split(data, cores, axis=0)
    
    df_out = np.vstack(pool.map(prepare_without_moats, df_split, orders, modes))
        
    pool.close()
    pool.join()
    pool.clear()
    
    scales = df_out[:,-1]
    moats = moats / scales[:,None]
    
    df_out = df_out[:,:-1]
    
    l = df_out[:,-1]
    df_out = df_out[:,:-1]
    
    return np.column_stack((df_out,moats)), l




def prepare_without_moats_litf_parallel(data, order=5, mode=4):
    
    cores = mp.cpu_count()
    pool = Pool(cores)
    
    litf = load_litf(data)
    data = load_raw(data)
    
    orders = [order]*cores
    modes = [mode]*cores
    
    df_split = np.array_split(data, cores, axis=0)
    
    df_out = np.vstack(pool.map(prepare_without_moats, df_split, orders, modes))
        
    pool.close()
    pool.join()
    pool.clear()

    scales = df_out[:,-1]
    
    colms = list(range(36))
    fix = [0,22,23,24,25,26,27,35]
    scl = list(set(colms)-set(fix))
    
    litf[:,scl] = litf[:,scl] / scales[:,None]
    
    df_out = df_out[:,:-1]
    
    l = df_out[:,-1]
    
    df_out = df_out[:,:-1]
 
    
    return np.column_stack((df_out,litf)), l


def prepare_with_moats_litf_parallel(data, order=5, mode=4):
    
    cores = mp.cpu_count()
    pool = Pool(cores)
    
    litf = load_litf(data)
    moats = load_moats(data)
    data = load_raw(data)
    
    
    orders = [order]*cores
    modes = [mode]*cores
    
    df_split = np.array_split(data, cores, axis=0)
    
    df_out = np.vstack(pool.map(prepare_without_moats, df_split, orders, modes))
        
    pool.close()
    pool.join()
    pool.clear()
    
    scales = df_out[:,-1]
    moats = moats / scales[:,None]

    colms = list(range(36))
    fix = [0,22,23,24,25,26,27,35]
    scl = list(set(colms)-set(fix))
    
    litf[:,scl] = litf[:,scl] / scales[:,None]


    
    df_out = df_out[:,:-1]
    
    l = df_out[:,-1]
    df_out = df_out[:,:-1]
    
    return np.column_stack((df_out,moats,litf)), l





