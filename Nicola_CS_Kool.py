import os

import torch

import numpy as np
import pandas as pd

import scipy
import random
import math
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance_matrix
from tqdm import tqdm

import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
import copy

from utils import load_model as torch_load_model

from itertools import repeat

from PrepareData import cast_dataset

def MAPE(predictions, labels):
    
    return abs((predictions/labels)-1).mean()*100


def cavdar_sokol_estimation_single(mdata):
    
    data = copy.deepcopy(mdata)
    
    size = int(data.shape[0]/2) #n in the paper
    
    x = data[:size] #x_coords of nodes
    y = data[size:] #y_coords of nodes
    
    l_x = x.max() - x.min() #l_x in the paper
    l_y = y.max() - y.min() #l_y in the paper 
    
    central_x = (x.max() + x.min())/2 # vertical central line
    central_y = (y.max() + y.min())/2 # horizontal central line
    
    central_x_bar = np.mean(abs(x - central_x)) #is abs correct? #average horizontal distance to vertical central line
    central_y_bar = np.mean(abs(y - central_y)) #average vertical distance to horizontal central line
    
    #central_x_bar = np.mean((x - central_x))
    #central_y_bar = np.mean((y - central_y))
    
    cbar = central_x_bar*central_y_bar
    
    
    std_x = np.std(x) #stdev_x in the paper
    std_y = np.std(y) #stdev_y in the paper 
    std = std_x*std_y
    
    cstd_x = np.std(abs(x - central_x)) #cstdev_x in the paper
    cstd_y = np.std(abs(y - central_y)) #cstdev_y in the paper
    cstd = cstd_x*cstd_y
    
    
    A = l_x*l_y
    
    T = 2.791*(math.sqrt(size*cstd)) + 0.2669*(math.sqrt(size*std*A/cbar))
    
    factor = 0.9325 * math.exp(0.00005298 * size) - 0.2972 * math.exp(-0.01452 * size)
    
    
    E = T/factor
    
    return E
             
def NicolaRCSingle(instance):
    
    size = instance.shape[0]//2
    
    coords_arr = np.column_stack((instance[:size],instance[size:]))
    
    req = size
    
    MinP = distance_matrix(coords_arr,coords_arr).min()#####
    MaxP = distance_matrix(coords_arr,coords_arr).max()#####
    
    DistM = distance_matrix(coords_arr,coords_arr)
    np.fill_diagonal(DistM, 10000000000000000)
    
    SumMinP = DistM.min(axis=1).sum()#####
    
    rec_x = 0.5*(max(instance[:size])+min(instance[:size]))
    rec_y = 0.5*(max(instance[size:])+min(instance[size:]))
    
    cust_centroid = coords_arr.mean(axis=0)
    cust_centroid = cust_centroid.reshape(-1,2)
    
    MinM = distance_matrix(cust_centroid,coords_arr).min()######
    SumM = distance_matrix(cust_centroid,coords_arr).sum()/2######
    VarM = distance_matrix(cust_centroid,coords_arr).var()######
    
    estimation = 0*size - 17.268*MinP + 2.107*MaxP - 6.597*MinM + 0.012*SumM - 0.090*VarM
    
    return estimation
        
    
def cavdar_sokol_batch(test_instance,size):
    
    data = pd.read_csv("BenchmarkInstances/{}_{}.csv".format(test_instance,size),header=None).to_numpy()
    
    size = int((data.shape[1]-1)/3)
    estimates = []
    opts = data[:,-1]
    
    for i in range(data.shape[0]):
        
        est = cavdar_sokol_estimation_single(data[i,:(2*size)])
        
        estimates.append(float(est))
    
    
    estimates = np.array(estimates)
    
    eval_sc = MAPE(estimates,opts)
    print("MAPE is:", eval_sc)
    
    key = "Cavdar and Sokol", size, test_instance
    
    return key, estimates/opts, eval_sc


def NicolaRC_batch(test_instance,size):
    
    data = pd.read_csv("BenchmarkInstances/{}_{}.csv".format(test_instance,size),header=None).to_numpy()
    
    size = int((data.shape[1]-1)/3)
    estimates = []
    opts = data[:,-1]
    
    for i in range(data.shape[0]):
        
        est = NicolaRCSingle(data[i,:(2*size)])
        
        estimates.append(float(est))
    
    
    estimates = np.array(estimates)
    
    eval_sc = MAPE(estimates,opts)
    print("MAPE is:", eval_sc)
    
    key = "Nicola et al.", size, test_instance
    
    return key, estimates/opts, eval_sc
        
        
    
def route_length(data):
    
    length = 0
    for i in range(1,data.shape[0]):
        length = length + np.linalg.norm(data[i,:] - data[i-1, :])

    return length


def make_oracle(model, xy, temperature=1.0):
    
    num_nodes = len(xy)
    
    xyt = torch.tensor(xy).float()[None]  # Add batch dimension
    
    with torch.no_grad():  # Inference only
        embeddings, _ = model.embedder(model._init_embed(xyt))

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = model._precompute(embeddings)
    
    def oracle(tour):
        with torch.no_grad():  # Inference only
            # Input tour with 0 based indices
            # Output vector with probabilities for locations not in tour
            tour = torch.tensor(tour).long()
            if len(tour) == 0:
                step_context = model.W_placeholder
            else:
                step_context = torch.cat((embeddings[0, tour[0]], embeddings[0, tour[-1]]), -1)

            # Compute query = context node embedding, add batch and step dimensions (both 1)
            query = fixed.context_node_projected + model.project_step_context(step_context[None, None, :])

            # Create the mask and convert to bool depending on PyTorch version
            mask = torch.zeros(num_nodes, dtype=torch.uint8) > 0
            mask[tour] = 1
            mask = mask[None, None, :]  # Add batch and step dimension

            log_p, _ = model._one_to_many_logits(query, fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key, mask)
            p = torch.softmax(log_p / temperature, -1)[0, 0]
            assert (p[tour] == 0).all()
            assert (p.sum() - 1).abs() < 1e-5
            #assert np.allclose(p.sum().item(), 1)
        return p.numpy()
    
    return oracle

def estimate_tour_lengths_rl(test_instance, size):
    
    model, _ = torch_load_model('pretrained/tsp_{}/'.format(size))
    model.eval()  # Put in evaluation mode to not track gradients


    
    test_data = pd.read_csv("BenchmarkInstances/{}_{}.csv".format(test_instance,size),header=None).to_numpy()
    
    
    
    
    dim = int((test_data.shape[1]-1)/3)
    
    label = test_data[:,-1]
    
    test_data,scales = cast_dataset(test_data[:,:2*dim])
    label = label/scales
    
    
    tour_lengths = []
    
    
    for i in tqdm(range(test_data.shape[0])):
        
        #if i % 100 == 0:
            #print(i)
        
        xy = np.column_stack((test_data[i,:dim],test_data[i,dim:2*dim]))
    
    
    
        oracle = make_oracle(model, xy)
        sample = False
        tour = []
        tour_p = []
    
        while(len(tour) < len(xy)):
            
            p = oracle(tour)
            
            if sample:
                # Advertising the Gumbel-Max trick
                g = -np.log(-np.log(np.random.rand(*p.shape)))
                i = np.argmax(np.log(p) + g)
                # i = np.random.multinomial(1, p)
            else:
            # Greedy
                i = np.argmax(p)
        
            tour.append(i)
            
        tour.append(tour[0])
        xy = xy[tour,:]
        
        tour_lengths.append(float(route_length(xy)))
      
    tour_lengths = np.array(tour_lengths)
    
    eval_sc = MAPE(tour_lengths,label)
    
    print("MAPE is:", eval_sc)
    
    key = "Kool et al.", size, test_instance
    
    return key, tour_lengths/label, eval_sc

