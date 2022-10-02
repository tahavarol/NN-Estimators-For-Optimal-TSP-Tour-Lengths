from BenchmarkInstances import NR
from concorde.tsp import TSPSolver
import multiprocessing as mp
import time
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool


#solves TSP problems of the raw data and returns the optimal tour and optimal objective value

def route_length(data):
    
    length = 0
    for i in range(1,data.shape[0]):
        length = length + np.linalg.norm(data[i,:] - data[i-1, :])

    return length
    

def concorde_single(data):
    
    size = int(data.shape[0]/2)
    
    
    
    x_s = data[0:size]
    y_s = data[size:(2*size)]
    
    solver = TSPSolver.from_data(x_s*10000, y_s*10000, norm="EUC_2D")
    solution = solver.solve()
    tour = list(solution.tour)
    tour.append(0)
    coords = np.column_stack((x_s, y_s))
    coords = coords[tour]
    obj = route_length(coords)
    tour.pop()
    tour.append(obj)
    
    return np.array(tour)


def concorde_solver(data):
    
    solutions = []
    
    for i in range(data.shape[0]):
        
        solution = concorde_single(data[i,:])
        solutions.append(solution)
    
    
    
    return np.array(solutions)


def concorde_solver_parallel(data):
    
    cores  = mp.cpu_count()
    pool = Pool(cores)
    df_split = np.array_split(data, cores, axis=0)
    df_out = np.vstack(pool.map(concorde_solver, df_split))
        
    pool.close()
    pool.join()
    pool.clear()
    
    df_out = np.column_stack((data, df_out))
    
    return df_out
    

def clear_environment():
    
    import os

    dir_name = os.getcwd()
    test = os.listdir(dir_name)

    for item in test:
        if item.endswith(".res") or item.endswith(".sol") or item.endswith(".pul") or item.endswith(".sav"):
            os.remove(os.path.join(dir_name, item))



instance_sizes = [20,50,100]
sample_sizes = [200000]
seeds = [11]
dict_size = len(instance_sizes) * len(sample_sizes) * len(seeds)

param_list = []

clear_environment()


for instance_size in instance_sizes:
    for sample_size in sample_sizes:
        for seed in seeds:
            param_list.append({"instance_size":instance_size,"sample_size":sample_size,"seed":seed})
                
                
print(param_list)   

dirname = "TrainingData/"

for param in param_list:
    
    print("solving instances with parameter set: ", param)
    
    data = NR(**param)
    s = time.time()
    a = concorde_solver_parallel(data)
    path = dirname + "Raw/" + str(param["instance_size"]) + ".csv"
    np.savetxt(path, a, delimiter=',')
    clear_environment()
    time.sleep(30)

