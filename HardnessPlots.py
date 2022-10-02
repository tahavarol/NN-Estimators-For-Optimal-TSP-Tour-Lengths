import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt


sizes = [20,50,100]
types = ["G1","G2_2","G3_1","G3_2","G3_3","G4","SG","US","UR","NS","NR"]

def get_data(size,itype):
    data = itype + "_" + str(size) + ".csv"
    path = "BenchmarkInstances/" + data
    data = pd.read_csv(path,header=None)
    return data.to_numpy()


def CardenasHardness(instance):
    shape = (instance.shape[0]-1)//3
    instance_x = instance[:shape]
    instance_y = instance[shape:2*shape]
    l = instance[-1]
    A = (max(instance_x) - min(instance_x))*(max(instance_y) - min(instance_y))
    N = shape
    return abs((l/(math.sqrt(N*A)))-0.75)


def Experiment(size_list,type_list):
    exp_dict = {}
    for size in size_list:
        exp_dict[size]={}
        for itype in type_list:
            d = get_data(size,itype)
            h_list = []
            for row in tqdm(d):
                h_list.append(CardenasHardness(row))
            exp_dict[size][itype] = h_list
    return exp_dict


def ExportHardnessPlots(experiments, key):
    
    ax = pd.DataFrame(experiments[key]).plot(kind='box',
             color=dict(boxes='r', whiskers='r', medians='r', caps='r'),
             boxprops=dict(linestyle='-', linewidth=1.5),
             flierprops=dict(linestyle='-', linewidth=1.5),
             medianprops=dict(linestyle='-', linewidth=1.5, color='black'),
             whiskerprops=dict(linestyle='-', linewidth=1.5),
             capprops=dict(linestyle='-', linewidth=1.5),
             showfliers=False, grid=True, rot=0, vert=False, fontsize=12, figsize=(10,6))

    ax.set_xlabel('Hardness Score (Smaller the Harder)', fontsize=12)
    ax.set_ylabel('Instance Type, Size={}'.format(key), fontsize=12)
    ax.set_xticks([])
    plt.savefig('CardenasHardnessPlots/Hardness_{}.jpg'.format(key),dpi=250,bbox_inches='tight',transparent=False)
    plt.show()
    


exp_dict = Experiment(sizes,types)

for size in sizes:
    ExportHardnessPlots(exp_dict, key=size)
