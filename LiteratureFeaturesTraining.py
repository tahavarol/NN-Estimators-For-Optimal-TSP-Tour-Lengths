import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import distance_matrix
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool


def CircularFeatures(ref_node, coords, M=0.5):
    
    M_dist = distance_matrix(ref_node,coords).max()
    M_dist = M_dist*M
    
    distances = distance_matrix(ref_node,coords).flatten()
    
    cts = (distances < M_dist).sum()
    
    return cts-1

def GenerateRectangleGrid(coords, div=10):
    x_s = np.linspace(coords[:,0].min(), coords[:,0].max(), num=div+1).tolist()
    y_s = np.linspace(coords[:,1].min(), coords[:,1].max(), num=div+1).tolist()
    sp = []
    for i in range(len(x_s)-1):
        for j in range(len(y_s)-1):
            p1 = (x_s[i],y_s[j])
            p2 = (x_s[i+1],y_s[j])
            p3 = (x_s[i],y_s[j+1])
            p4 = (x_s[i+1],y_s[j+1])
            sp.append([p1,p2,p3,p4])
            
    return sp


def IfPointInsideRectangle(rect_corner_quadruple, point):
    x_min = np.array(rect_corner_quadruple)[:,0].min()
    x_max = np.array(rect_corner_quadruple)[:,0].max()
    
    y_min = np.array(rect_corner_quadruple)[:,1].min()
    y_max = np.array(rect_corner_quadruple)[:,1].max()
    
    if (point[0]<=x_max) and (point[0]>=x_min) and (point[1]<=y_max) and (point[1]>=y_min):
        return True
    else:
        return False


def GetBearing(ref,pt):
    m = (ref[0][1]-pt[1])/(ref[0][0]-pt[0])
    return np.arctan(m)


def LiteratureFeatures(instance):
    size = instance.shape[0]//2
    
    #Rasku et al. (2016), Arnold and Sörensen (2019), Nicola et al.(2019)
    F1 = size
    
    #Kwon et al.(1995), Rasku et al.(2016)
    F2 = (max(instance[:size])-min(instance[:size]))*(max(instance[size:])-min(instance[size:])) #area
    F3 = 2*(max(instance[:size])-min(instance[:size]))+(max(instance[size:])-min(instance[size:])) #perimeter
    
    #Rasku et al. (2016)
    coords_arr = np.column_stack((instance[:size],instance[size:]))
    hull = ConvexHull(coords_arr)
    F4 = hull.volume #hull's area
    F5 = hull.area #hull's perimeter
    
    #Rasku et al.(2016)
    F6 = max(instance[:size])-min(instance[:size]) #width
    F7 = max(instance[size:])-min(instance[size:]) #height
    
    #Rasku et al. (2016), Arnold and Sörensen (2019), Nicola et al.(2019)
    F8 = distance_matrix(coords_arr,coords_arr).mean() #avg distance between nodes
    
    #Kwon et al.(1995), Rasku et al. (2016), Arnold and Sörensen (2019), Nicola et al.(2019)
    F9 = distance_matrix(coords_arr[0].reshape(-1,2),coords_arr).mean() #avg distance to depot
    
    #Rasku et al. (2016), Arnold and Sörensen (2019), Nicola et al.(2019)
    depot = coords_arr[0].reshape(-1,2)
    rec_x = 0.5*(max(instance[:size])+min(instance[:size]))
    rec_y = 0.5*(max(instance[size:])+min(instance[size:]))
    rect_centroid = np.array([[rec_x,rec_y]])
    cust_centroid = coords_arr.mean(axis=0)
    cust_centroid = cust_centroid.reshape(-1,2)
    F10 = distance_matrix(depot,rect_centroid)[0][0] #depot to rectangle centroid
    F11 = distance_matrix(depot,cust_centroid)[0][0] #depot to customer centroid
    
    #Rasku et al. (2016), Arnold and Sörensen (2019), Nicola et al.(2019)
    F12 = distance_matrix(rect_centroid,coords_arr).mean() #avg distance of customers to rect centroid
    F13 = distance_matrix(cust_centroid,coords_arr).mean() #avg distance of customers to cust centroid
    
    
    #Arnold and Sörensen (2019)
    F14 = np.array([GetBearing(depot,pt) for pt in coords_arr[1:]]).std()
    F15 = np.array([GetBearing(cust_centroid,pt) for pt in coords_arr]).std()
    F16 = np.array([GetBearing(rect_centroid,pt) for pt in coords_arr]).std()
    
    #Nicola et al. (2016)
    F17 = (coords_arr[:,0].std() + coords_arr[:,1].std())/2
    F18 = coords_arr[:,0].std() * coords_arr[:,1].std()
    F19 = distance_matrix(depot,coords_arr).std()
    F20 = distance_matrix(cust_centroid,coords_arr).std()
    F21 = distance_matrix(rect_centroid,coords_arr).std()
    F22 = distance_matrix(coords_arr,coords_arr).std()
    
    #Akkerman and Mes (2022)
    F23 = CircularFeatures(depot, coords_arr, M=0.5)
    F24 = CircularFeatures(depot, coords_arr, M=0.75)
    F25 = CircularFeatures(cust_centroid, coords_arr, M=0.5)
    F26 = CircularFeatures(cust_centroid, coords_arr, M=0.75)
    F27 = CircularFeatures(rect_centroid, coords_arr, M=0.5)
    F28 = CircularFeatures(rect_centroid, coords_arr, M=0.75)

    #Akkerman and Mes (2022)
    grid = GenerateRectangleGrid(coords_arr,div=10)
    point_chk = [[IfPointInsideRectangle(item, point) for point in coords_arr] for item in grid]
    point_cts = [sum(item) for item in point_chk]
    active_rects = [idx for idx in range(len(point_cts)) if point_cts[idx]!=0]
    most_pts_centroid = np.array(grid[point_cts.index(max(point_cts))]).mean(axis=0)
    most_pts_centroid = most_pts_centroid.reshape(-1,2)
    F29 = distance_matrix(depot,most_pts_centroid)[0][0]
    active_rect_centroids = [np.array(grid[index]).mean(axis=0) for index in range(len(point_cts)) if point_cts[index]!=0]
    F30 = distance_matrix(depot,np.array(active_rect_centroids)).mean()
    F31 = distance_matrix(np.array(active_rect_centroids),np.array(active_rect_centroids)).mean()
    F32_ = [distance_matrix(coords_arr[point_chk[idx]],coords_arr[point_chk[idx]]).mean() for idx in active_rects]
    F32 = sum(F32_) / len(F32_)
    
    
    
    grid = GenerateRectangleGrid(coords_arr,div=15)
    point_chk = [[IfPointInsideRectangle(item, point) for point in coords_arr] for item in grid]
    point_cts = [sum(item) for item in point_chk]
    active_rects = [idx for idx in range(len(point_cts)) if point_cts[idx]!=0]
    most_pts_centroid = np.array(grid[point_cts.index(max(point_cts))]).mean(axis=0)
    most_pts_centroid = most_pts_centroid.reshape(-1,2)
    F33 = distance_matrix(depot,most_pts_centroid)[0][0]
    active_rect_centroids = [np.array(grid[index]).mean(axis=0) for index in range(len(point_cts)) if point_cts[index]!=0]
    F34 = distance_matrix(depot,np.array(active_rect_centroids)).mean()
    F35 = distance_matrix(np.array(active_rect_centroids),np.array(active_rect_centroids)).mean()
    F36_ = [distance_matrix(coords_arr[point_chk[idx]],coords_arr[point_chk[idx]]).mean() for idx in active_rects]
    F36 = sum(F36_) / len(F36_)


    return [F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,
            F12,F13,F14,F15,F16,F17,F18,F19,F20,
            F21,F22,F23,F24,F25,F26,F27,F28,F29,
            F30,F31,F32,F33,F34,F35,F36]



def LiteratureFeaturesBatch(data):
    dataset = []
    size = data.shape[1]//2
    for row in tqdm(data):
        dataset.append(LiteratureFeatures(row))
    return np.array(dataset)


def LiteratureFeaturesBatchParallel(data):
    
    cores=mp.cpu_count()
    
    
    pool = Pool(cores)
    
    df_split = np.array_split(data, cores, axis=0)
    
    df_out = np.vstack(pool.map(LiteratureFeaturesBatch, df_split))
        
    pool.close()
    pool.join()
    pool.clear()
   
    return df_out  

def ComputeTrainingLiteratureFeatures(size):
    
    load_path = "TrainingData/Raw/"
    save_path = "TrainingData/LiteratureFeatures/"
    
    data = pd.read_csv(load_path+str(size)+".csv", header=None).to_numpy()
    coords = data[:,:2*size]
    litfeats = LiteratureFeaturesBatchParallel(coords)
    np.savetxt(save_path + "LiteratureFeatures" + "_" + str(size) + ".csv", litfeats, delimiter=',')
    
    
if __name__ == "__main__":
    
    sizes = [20,50,100]
    for size in sizes:
        ComputeTrainingLiteratureFeatures(size)