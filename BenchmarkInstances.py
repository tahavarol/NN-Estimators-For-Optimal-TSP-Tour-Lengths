import numpy as np
import math
import random 
from tqdm import tqdm
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure 
import time 
from scipy.spatial import Delaunay


def VisualizeInstance(instance, save=False, instance_type="US"):
    
    figure(figsize=(8, 8), dpi=100)
    size = instance.shape[0]//2
    plt.scatter(instance[:size],instance[size:],s=10,c="black")
    
    if save:
        name = str(time.time())
        plt.savefig("{}_{}.jpg".format(instance_type,size),bbox_inches='tight')
        
        

#this is the first step. In this step we generate our instances.

#this function checks if a randomly generated point in the map or not.
def in_map(x_coord,y_coord,upper_left_x,bottom_right_x,bottom_right_y,upper_left_y):
    
    if((x_coord>upper_left_x and x_coord<bottom_right_x) and (y_coord>bottom_right_y and y_coord<upper_left_y)):
        return True
    
    else:
        return False
    
    
#this function generates a non uniform instance having any size and any x*y rectangular shape 

def NR_(size):
    
    upper_left_x = float(np.random.uniform(0,1000,1))
    upper_left_y = float(np.random.uniform(0,1000,1))
    bottom_right_x= float(np.random.uniform(upper_left_x,1000,1))
    bottom_right_y= float(np.random.uniform(0,upper_left_y,1))
    height = upper_left_y - bottom_right_y
    width = bottom_right_x - upper_left_x
    
    n_clust = int(np.random.randint(1, 11, size=1))
    #print(n_clust)
    clust_ratio = float(np.random.uniform(0.5, 0.9,1))
    
    number_of_cluster_points = int(np.round(clust_ratio * size))
    
    number_of_non_cluster_points = size - number_of_cluster_points
    
    cluster_points_coords_x = np.zeros(shape=(number_of_cluster_points,))
    cluster_points_coords_y = np.zeros(shape=(number_of_cluster_points,))
    
    
    
    if(n_clust == 1):
        clust_center_x = float((bottom_right_x + upper_left_x)/2)
        clust_center_y = float((bottom_right_y + upper_left_y)/2)
        
        my_len=0
        scaler = float(np.random.uniform(0.001,0.2,1))
        while my_len < number_of_cluster_points:
            
            clx = clust_center_x + float(np.random.default_rng().normal(0, 1, 1))*width*scaler
            cly = clust_center_y + float(np.random.default_rng().normal(0, 1, 1))*height*scaler
            
            if in_map(clx,cly,upper_left_x,bottom_right_x,bottom_right_y,upper_left_y):
                cluster_points_coords_x[my_len] = clx
                cluster_points_coords_y[my_len] = cly
                my_len = my_len+1
                
    else:
        
        clust_center_x = np.random.uniform(upper_left_x,bottom_right_x,n_clust)
        clust_center_y = np.random.uniform(bottom_right_y, upper_left_y,n_clust)
        
    
        my_len=0
        scaler = float(np.random.uniform(0.001,0.2,1))
        while my_len < number_of_cluster_points:
        
            base = math.ceil(np.random.uniform(0,1,1) * n_clust) - 1 #base is the ID of cluster
            clx = clust_center_x[base] + float(np.random.default_rng().normal(0, 1, 1))*width*scaler
            cly = clust_center_y[base] + float(np.random.default_rng().normal(0, 1, 1))*height*scaler
            if in_map(clx,cly,upper_left_x,bottom_right_x,bottom_right_y,upper_left_y):
                cluster_points_coords_x[my_len] = clx
                cluster_points_coords_y[my_len] = cly
                my_len = my_len+1

    
    
   
    non_cluster_points_coords_x = (np.random.uniform(upper_left_x,bottom_right_x,number_of_non_cluster_points))
    
    non_cluster_points_coords_y = (np.random.uniform(bottom_right_y,upper_left_y,number_of_non_cluster_points))
        
    coords_x = np.concatenate((cluster_points_coords_x,non_cluster_points_coords_x))
    coords_y = np.concatenate((cluster_points_coords_y,non_cluster_points_coords_y))
    
    coords = np.concatenate((coords_x,coords_y))
    
    return coords


#this function creates data set of desired size. First input is the number of instances and the second input is the size of instances.

def NR(instance_size, sample_size, seed=1):
    
    random.seed(seed)
    np.random.seed(seed)
    
    
    my_list=[]
    
    for i in tqdm(range(sample_size)):
        a=NR_(instance_size)
        my_list.append(a)
    
    data = np.array(my_list)
    
    return data



#this function generates a non uniform instance having any size and any 1000*1000 square shape 

def NS_(size):
    
    upper_left_x = 0
    upper_left_y = 1000
    bottom_right_x= 1000
    bottom_right_y= 0
    height = upper_left_y - bottom_right_y
    width = bottom_right_x - upper_left_x
    
    
    n_clust = int(np.random.randint(1, 11, size=1))
    
    clust_ratio = float(np.random.uniform(0.5, 0.9,1))
    
    number_of_cluster_points = int(np.round(clust_ratio * size))
    
    number_of_non_cluster_points = size - number_of_cluster_points
    
    cluster_points_coords_x = np.zeros(shape=(number_of_cluster_points,))
    cluster_points_coords_y = np.zeros(shape=(number_of_cluster_points,))
    
    
    
    if(n_clust == 1):
        clust_center_x = float((bottom_right_x + upper_left_x)/2)
        clust_center_y = float((bottom_right_y + upper_left_y)/2)
        
        my_len=0
        scaler = float(np.random.uniform(0.001,0.2,1))
        while my_len < number_of_cluster_points:
            
            clx = clust_center_x + float(np.random.default_rng().normal(0, 1, 1))*width*scaler
            cly = clust_center_y + float(np.random.default_rng().normal(0, 1, 1))*height*scaler
            
            if in_map(clx,cly,upper_left_x,bottom_right_x,bottom_right_y,upper_left_y):
                cluster_points_coords_x[my_len] = clx
                cluster_points_coords_y[my_len] = cly
                my_len = my_len+1
                
    else:
        
        clust_center_x = np.random.uniform(upper_left_x,bottom_right_x,n_clust)
        clust_center_y = np.random.uniform(bottom_right_y, upper_left_y,n_clust)
        
    
        my_len=0
        scaler = float(np.random.uniform(0.001,0.2,1))
        while my_len < number_of_cluster_points:
        
            base = math.ceil(np.random.uniform(0,1,1) * n_clust) - 1 #base is the ID of cluster
            clx = clust_center_x[base] + float(np.random.default_rng().normal(0, 1, 1))*width*scaler
            cly = clust_center_y[base] + float(np.random.default_rng().normal(0, 1, 1))*height*scaler
            if in_map(clx,cly,upper_left_x,bottom_right_x,bottom_right_y,upper_left_y):
                cluster_points_coords_x[my_len] = clx
                cluster_points_coords_y[my_len] = cly
                my_len = my_len+1

    
    
   
    non_cluster_points_coords_x = (np.random.uniform(upper_left_x,bottom_right_x,number_of_non_cluster_points))
    
    non_cluster_points_coords_y = (np.random.uniform(bottom_right_y,upper_left_y,number_of_non_cluster_points))
        
    coords_x = np.concatenate((cluster_points_coords_x,non_cluster_points_coords_x))
    coords_y = np.concatenate((cluster_points_coords_y,non_cluster_points_coords_y))
    
    coords = np.concatenate((coords_x,coords_y))
    
    return coords


def NS(instance_size, sample_size, seed=1):
    
    random.seed(seed)
    np.random.seed(seed)
    
    
    my_list=[]
    
    for i in range(sample_size):
        a=NS_(instance_size)
        my_list.append(a)
    
    data = np.array(my_list)
    
    return data


def US_(instance_size):
    
    instance_x = np.random.uniform(0,1000,instance_size)
    instance_y = np.random.uniform(0,1000,instance_size)
    
    return np.concatenate((instance_x,instance_y))

def US(instance_size, sample_size, seed=1):
    
    random.seed(seed)
    np.random.seed(seed)
    
    data_set = []
    
    for _ in range(sample_size):
        
        data_set.append(US_(instance_size))
    
    return np.array(data_set)


def UR_(size):
    
        upper_left_x = float(np.random.uniform(0,1000,1))
        upper_left_y = float(np.random.uniform(0,1000,1))
        bottom_right_x= float(np.random.uniform(upper_left_x,1000,1))
        bottom_right_y= float(np.random.uniform(0,upper_left_y,1))
        height = upper_left_y - bottom_right_y
        width = bottom_right_x - upper_left_x
        coords_x = (np.random.uniform(upper_left_x,bottom_right_x,size))
        coords_y = (np.random.uniform(bottom_right_y,upper_left_y,size))
        coords = np.concatenate((coords_x,coords_y))
        
        return coords
    
    
def UR(instance_size, sample_size, seed=1):
    
    random.seed(seed)
    np.random.seed(seed)
    
    data_set = []
    
    for _ in range(sample_size):
        
        data_set.append(UR_(instance_size))
    
    return np.array(data_set)


def Uniform(instance_size, multiplier_x=1, multiplier_y=1):
    
    l_x = [250, 300, 400, 500, 600, 800, 1000, 1200, 1600]
    l_x_ = random.sample(l_x,1)[0]*multiplier_x
    
    l_y = [250000/l_x_, 360000/l_x_, 640000/l_x_]
    l_y_ = random.sample(l_y,1)[0]*multiplier_y
    
    instance_x = np.random.uniform(0,l_x_,instance_size)
    instance_y = np.random.uniform(0,l_y_,instance_size)
    
    return np.concatenate((instance_x,instance_y))


def Triangular(instance_size, multiplier_x=1, multiplier_y=1):
    
    l_x = [250, 300, 400, 500, 600, 800, 1000, 1200, 1600]
    l_x_ = random.sample(l_x,1)[0]*multiplier_x
    l_y = [250000/l_x_, 360000/l_x_, 640000/l_x_]
    l_y_ = random.sample(l_y,1)[0]*multiplier_y
    
    instance_x = np.random.triangular(0,l_x_/2,l_x_,instance_size)
    instance_y = np.random.triangular(0,l_y_/2,l_y_,instance_size)

    
    return np.concatenate((instance_x,instance_y))


def Squeezed(instance_size, multiplier_x=1, multiplier_y=1):

    l_x = [250, 300, 400, 500, 600, 800, 1000, 1200, 1600]
    l_x_ = random.sample(l_x,1)[0]*multiplier_x
    
    l_y = [250000/l_x_, 360000/l_x_, 640000/l_x_]
    l_y_ = random.sample(l_y,1)[0]*multiplier_y


    instance_x = []
    instance_y = []
    
    while len(instance_x)<instance_size:
        
        x = np.random.uniform(0, l_x_, 1)[0]
        y = np.random.uniform(0, l_y_, 1)[0]
        accept = np.random.uniform(0, l_x_*l_y_, 1)[0]
        
        if accept < x*y:
            
            instance_x.append(x)
            instance_y.append(y)
        
    return np.concatenate((np.array(instance_x),np.array(instance_y)))


def X_uniformY_triangular(instance_size, multiplier_x=1, multiplier_y=1):
    
    l_x = [250, 300, 400, 500, 600, 800, 1000, 1200, 1600]
    l_x_ = random.sample(l_x,1)[0]*multiplier_x
    
    l_y = [250000/l_x_, 360000/l_x_, 640000/l_x_]
    l_y_ = random.sample(l_y,1)[0]*multiplier_y
    
    instance_x = np.random.uniform(0,l_x_,instance_size)
    instance_y = np.random.triangular(0,l_y_/2,l_y_,instance_size)

    
    return np.concatenate((instance_x,instance_y))



def X_triangularY_squeezed(instance_size, multiplier_x=1, multiplier_y=1):
    
    l_x = [250, 300, 400, 500, 600, 800, 1000, 1200, 1600]
    l_x_ = random.sample(l_x,1)[0]*multiplier_x
    
    l_y = [250000/l_x_, 360000/l_x_, 640000/l_x_]
    l_y_ = random.sample(l_y,1)[0]*multiplier_y
    
    instance_x = []
    instance_y = []
    
    while len(instance_x)<instance_size:
        
        x = np.random.triangular(0, l_x_/2, l_x_, 1)[0]
        y = np.random.uniform(0, l_y_, 1)[0]
        accept = np.random.uniform(0,l_y_, 1)[0]
        
        if accept < y :
            
            instance_x.append(x)
            instance_y.append(y)
        
    return np.concatenate((np.array(instance_x),np.array(instance_y)))



def X_central(instance_size, multiplier_x=1, multiplier_y=1):
    
    l_x = [250, 300, 400, 500, 600, 800, 1000, 1200, 1600]
    l_x_ = random.sample(l_x,1)[0]*multiplier_x
    l_y = [250000/l_x_, 360000/l_x_, 640000/l_x_]
    l_y_ = random.sample(l_y,1)[0]*multiplier_y
    
    instance_x = []
    instance_y = []
    
    while len(instance_x)<instance_size:
        
        x = np.random.uniform(0, l_x_, 1)[0]
        y = np.random.uniform(0, l_y_, 1)[0]
        accept = np.random.uniform(0,1,1)[0]
        
        if accept < (1-(abs(x-l_x_/2)/(l_x_/2)))*(abs(y-l_y_/2)/(l_y_/2)):
            
            instance_x.append(x)
            instance_y.append(y)
        
    return np.concatenate((np.array(instance_x),np.array(instance_y)))
    
    

def G1_(instance_size):
    
    dispersion_type = np.random.randint(1,7,1)[0]
    if dispersion_type == 1:
        return Uniform(instance_size)

    if dispersion_type == 2:
        return Triangular(instance_size)
    
    if dispersion_type == 3:
        return Squeezed(instance_size)
    
    if dispersion_type == 4:
        return X_uniformY_triangular(instance_size)
    
    if dispersion_type == 5:
        return X_triangularY_squeezed(instance_size)
    
    if dispersion_type == 6:
        return X_central(instance_size)
    
def G1(instance_size, sample_size, seed=1):
    
    random.seed(seed)
    np.random.seed(seed)
    
    data_set = []
    
    for _ in range(sample_size):
        
        data_set.append(G1_(instance_size))
    
    return np.array(data_set)



def G2_2_(instance_size):
    
    dispersion_type = np.random.randint(1,7,1)[0]
    if dispersion_type == 1:
        return Uniform(instance_size, multiplier_x=1.4, multiplier_y=1.2)

    if dispersion_type == 2:
        return Triangular(instance_size, multiplier_x=1.4, multiplier_y=1.2)
    
    if dispersion_type == 3:
        return Squeezed(instance_size, multiplier_x=1.4, multiplier_y=1.2)
    
    if dispersion_type == 4:
        return X_uniformY_triangular(instance_size, multiplier_x=1.4, multiplier_y=1.2)
    
    if dispersion_type == 5:
        return X_triangularY_squeezed(instance_size, multiplier_x=1.4, multiplier_y=1.2)
    
    if dispersion_type == 6:
        return X_central(instance_size, multiplier_x=1.4, multiplier_y=1.2)
    
def G2_2(instance_size, sample_size, seed=1):
    
    random.seed(seed)
    np.random.seed(seed)
    
    data_set = []
    
    for _ in range(sample_size):
        
        data_set.append(G2_2_(instance_size))
    
    return np.array(data_set)



def G3_1_(instance_size, multiplier_x=1, multiplier_y=1):
    
    l_x = [250, 500, 1000]
    l_x_ = random.sample(l_x,1)[0]*multiplier_x
    
    l_y = [100000/l_x_]
    l_y_ = random.sample(l_y,1)[0]*multiplier_y
    
    instance_x = []
    instance_y = []
    
    while len(instance_x)<instance_size:
        
        x = np.random.uniform(0, l_x_, 1)[0]
        y = np.random.uniform(0, l_y_, 1)[0]
        accept = np.random.uniform(0,1,1)[0]
        
        if accept < (abs(x-l_x_/2)/(l_x_/2))*(abs(y-l_y_/2)/(l_y_/2)):
            
            instance_x.append(x)
            instance_y.append(y)
        
    return np.concatenate((np.array(instance_x),np.array(instance_y)))


def G3_1(instance_size, sample_size, seed=1):
    
    random.seed(seed)
    np.random.seed(seed)
    
    data_set = []
    
    for _ in range(sample_size):
        
        data_set.append(G3_1_(instance_size))
    
    return np.array(data_set)



def G3_2_(instance_size, multiplier_x=1, multiplier_y=1):
    
    l_x = [250, 500, 1000]
    l_x_ = random.sample(l_x,1)[0]*multiplier_x
    
    l_y = [100000/l_x_]
    l_y_ = random.sample(l_y,1)[0]*multiplier_y
    
    instance_x = []
    instance_y = []
    rate = 1
    while len(instance_x)<instance_size:
        
        a = -np.log(np.random.uniform(0, 1, 1)[0])/rate
        b = -np.log(np.random.uniform(0, 1, 1)[0])/rate
        
        x = (a-np.floor(a))*l_x_
        y = (b-np.floor(b))*l_y_
        
        instance_x.append(x)
        instance_y.append(y)
        
    return np.concatenate((np.array(instance_x),np.array(instance_y)))


def G3_2(instance_size, sample_size, seed=1):
    
    random.seed(seed)
    np.random.seed(seed)
    
    data_set = []
    
    for _ in range(sample_size):
        
        data_set.append(G3_2_(instance_size))
    
    return np.array(data_set)



def G3_3_(instance_size, multiplier_x=1, multiplier_y=1):
    
    l_x = [250, 500, 1000]
    l_x_ = random.sample(l_x,1)[0]*multiplier_x
    
    l_y = [100000/l_x_]
    l_y_ = random.sample(l_y,1)[0]*multiplier_y
    
    instance_x = []
    instance_y = []
    rate = 2
    while len(instance_x)<instance_size:
        
        a = -np.log(np.random.uniform(0, 1, 1)[0])/rate
        b = -np.log(np.random.uniform(0, 1, 1)[0])/rate
        
        x = (a-np.floor(a))*l_x_
        y = (b-np.floor(b))*l_y_
        
        instance_x.append(x)
        instance_y.append(y)
        
    return np.concatenate((np.array(instance_x),np.array(instance_y)))


def G3_3(instance_size, sample_size, seed=1):
    
    random.seed(seed)
    np.random.seed(seed)
    
    data_set = []
    
    for _ in range(sample_size):
        
        data_set.append(G3_3_(instance_size))
    
    return np.array(data_set)



def SG_(instance_size, multiplier_x=1, multiplier_y=1):
    
    l_x = [500, 1000]
    l_x_ = random.sample(l_x,1)[0]*multiplier_x
    
    l_y = [1000000/l_x_]
    l_y_ = random.sample(l_y,1)[0]*multiplier_y
    
    
    instance_x = np.random.uniform(0,l_x_,instance_size)
    instance_y = np.random.uniform(0,l_y_,instance_size)
    
    return np.concatenate((instance_x,instance_y))



def SG(instance_size, sample_size, seed=1):

    random.seed(seed)
    np.random.seed(seed)
    
    data_set = []
    
    for _ in range(sample_size):
        
        data_set.append(SG_(instance_size))
    
    return np.array(data_set)



def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0



def G4_(instance_size, multiplier_x=1, multiplier_y=1):
    
    m = [5,10,15]
    m_= random.sample(m,1)[0]*multiplier_x
    
    l_x = [250, 500, 1000]
    l_x_ = random.sample(l_x,1)[0]*multiplier_x
    
    l_y = [1000000/l_x_]
    l_y_ = random.sample(l_y,1)[0]*multiplier_y

    hull_x = np.random.uniform(0,l_x_,m_)
    hull_y = np.random.uniform(0,l_y_,m_)
    hull_corners = np.column_stack((hull_x,hull_y))
    
    instance_x = []
    instance_y = []
    
    while len(instance_x) < instance_size - m_:
        
        _x = np.random.uniform(0, l_x_, 1)[0]
        _y = np.random.uniform(0, l_y_, 1)[0]
        
        point = np.array([_x,_y]).reshape(1,2)
        
        if in_hull(point,hull_corners):
            
            instance_x.append(_x)
            instance_y.append(_y)
            
    instance_x = instance_x + hull_x.tolist()
    instance_y = instance_y + hull_y.tolist()
    
    return np.concatenate((np.array(instance_x),np.array(instance_y)))
    
    
def G4(instance_size, sample_size, seed=1):

    random.seed(seed)
    np.random.seed(seed)
    
    data_set = []
    
    for _ in range(sample_size):
        
        data_set.append(G4_(instance_size))
    
    return np.array(data_set)
    
