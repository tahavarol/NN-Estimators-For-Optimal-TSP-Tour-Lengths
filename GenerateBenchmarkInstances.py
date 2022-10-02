from BenchmarkInstances import *
from concorde.tsp import TSPSolver
import time


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


def clear_environment():
    
    import os

    dir_name = os.getcwd()
    test = os.listdir(dir_name)

    for item in test:
        if item.endswith(".res") or item.endswith(".sol") or item.endswith(".pul") or item.endswith(".sav"):
            os.remove(os.path.join(dir_name, item))



instance_sizes = [20,50,100]
sample_sizes = [1000]
seeds = [1]
dict_size = len(instance_sizes) * len(sample_sizes) * len(seeds)

param_list = []

clear_environment()

for instance_size in instance_sizes:
    for sample_size in sample_sizes:
        for seed in seeds:
            param_list.append({"instance_size":instance_size,"sample_size":sample_size,"seed":seed})
                
                
print(param_list)               
runtimes = {}
instance_names = ["G1", "G2_2", "G3_1", "G3_2", "G3_3", "G4", "SG", "US", "UR", "NS", "NR"]

dirname = "BenchmarkInstances/"

for param in param_list:
    
    print("solving instances with parameter set: ", param)
    
    
    print("solving G1")
    data = G1(**param)
    s = time.time()
    a = concorde_solver(data)
    a = np.column_stack((data, a))
    runtimes[(param["instance_size"],"G1")] = time.time()-s
    path = dirname + "G1" + "_" + str(param["instance_size"]) + ".csv"
    np.savetxt(path, a, delimiter=',')
    clear_environment()
    time.sleep(10)
    
    print("solving G2_2")
    data = G2_2(**param)
    s = time.time()
    a = concorde_solver(data)
    a = np.column_stack((data, a))
    runtimes[(param["instance_size"],"G2_2")] = time.time()-s
    path = dirname + "G2_2" + "_" + str(param["instance_size"]) + ".csv"
    np.savetxt(path, a, delimiter=',')
    clear_environment()
    time.sleep(10)
    
    print("solving G3_1")
    data = G3_1(**param)
    s = time.time()
    a = concorde_solver(data)
    a = np.column_stack((data, a))
    runtimes[(param["instance_size"],"G3_1")] = time.time()-s
    path = dirname + "G3_1" + "_" + str(param["instance_size"]) + ".csv"
    np.savetxt(path, a, delimiter=',')
    clear_environment()
    time.sleep(10)
    
    print("solving G3_2")
    data = G3_2(**param)
    s = time.time()
    a = concorde_solver(data)
    a = np.column_stack((data, a))
    runtimes[(param["instance_size"],"G3_2")] = time.time()-s
    path = dirname + "G3_2" + "_" + str(param["instance_size"]) + ".csv"
    np.savetxt(path, a, delimiter=',')
    clear_environment()
    time.sleep(10)
    
    print("solving G3_3")
    data = G3_3(**param)
    s = time.time()
    a = concorde_solver(data)
    a = np.column_stack((data, a))
    runtimes[(param["instance_size"],"G3_3")] = time.time()-s
    path = dirname + "G3_3" + "_" + str(param["instance_size"]) + ".csv"
    np.savetxt(path, a, delimiter=',')
    clear_environment()
    time.sleep(10)
    
    print("solving G4")
    data = G4(**param)
    s = time.time()
    a = concorde_solver(data)
    a = np.column_stack((data, a))
    runtimes[(param["instance_size"],"G4")] = time.time()-s
    path = dirname + "G4" + "_" + str(param["instance_size"]) + ".csv"
    np.savetxt(path, a, delimiter=',')
    clear_environment()
    time.sleep(10)
    
    print("solving SG")
    data = SG(**param)
    s = time.time()
    a = concorde_solver(data)
    a = np.column_stack((data, a))
    runtimes[(param["instance_size"],"SG")] = time.time()-s
    path = dirname + "SG" + "_" + str(param["instance_size"]) + ".csv"
    np.savetxt(path, a, delimiter=',')
    clear_environment()
    time.sleep(10)
    
    print("solving US")
    data = US(**param)
    s = time.time()
    a = concorde_solver(data)
    a = np.column_stack((data, a))
    runtimes[(param["instance_size"],"US")] = time.time()-s
    path = dirname + "US" + "_" + str(param["instance_size"]) + ".csv"
    np.savetxt(path, a, delimiter=',')
    clear_environment()
    time.sleep(10)
    
    print("solving UR")
    data = UR(**param)
    s = time.time()
    a = concorde_solver(data)
    a = np.column_stack((data, a))
    runtimes[(param["instance_size"],"UR")] = time.time()-s
    path = dirname + "UR" + "_" + str(param["instance_size"]) + ".csv"
    np.savetxt(path, a, delimiter=',')
    clear_environment()
    time.sleep(10)
    
    print("solving NS")
    data = NS(**param)
    s = time.time()
    a = concorde_solver(data)
    a = np.column_stack((data, a))
    runtimes[(param["instance_size"],"NS")] = time.time()-s
    path = dirname + "NS" + "_" + str(param["instance_size"]) + ".csv"
    np.savetxt(path, a, delimiter=',')
    clear_environment()
    time.sleep(10)
    
    print("solving NR")
    data = NR(**param)
    s = time.time()
    a = concorde_solver(data)
    a = np.column_stack((data, a))
    runtimes[(param["instance_size"],"NR")] = time.time()-s
    path = dirname + "NR" + "_" + str(param["instance_size"]) + ".csv"
    np.savetxt(path, a, delimiter=',')
    clear_environment()
    time.sleep(10)    
    
print(runtimes)

    