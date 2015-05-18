import numpy as np
import random as rand

def CreateArtificialDataset(num_rows,num_cols,sparsity_rate,num_clusters,cluster_overlap_dict):
    A = np.zeros((num_rows,num_cols))
    nonzeros_per_row = np.ceil((float(A.size) * float(sparsity_rate)) / float(num_rows))
    rows_per_cluster = np.floor(float(num_rows) / float(num_clusters))
    
    for k in range(0,num_clusters):        
        from_row = int(k * rows_per_cluster)
        to_row = int(k * rows_per_cluster + rows_per_cluster)
        from_col = int(k * nonzeros_per_row)
        to_col = int(k * nonzeros_per_row + nonzeros_per_row)
        
        if k in cluster_overlap_dict:
            shift = cluster_overlap_dict[k]
            
            from_col = max(0,from_col-shift)
            to_col = max(0,to_col-shift)
                
        for i in range(from_row,to_row):
            for j in range(from_col,to_col):
                A[i,j]=1
        
        if k == num_clusters-1:
            for i in range(to_row,A.shape[0]):
                for j in range(from_col,to_col):
                    A[i,j]=1
                    
            to_row=A.shape[0]
        
    return A
    
def AddNoise(A, noise_level):
    indices_ones = np.where(A==1)
    indices_zeros = np.where(A==0)
    
    num_ones = len(indices_ones[0])
    num_zeros = len(indices_zeros[0])
    
    total_flips = int(np.floor(float(num_ones) * noise_level))
    
    for i in range(0, total_flips):
        one_to_flip = rand.randrange(0,num_ones)
        zero_to_flip = rand.randrange(0,num_zeros)
        
        A[indices_ones[0][one_to_flip],indices_ones[1][one_to_flip]] = 0
        A[indices_zeros[0][zero_to_flip],indices_zeros[1][zero_to_flip]] = 1
        
    return A
    
#----- Main Program -----

#overlap dictionary is simply key = cluster index to overlap with the one before it
#and value = how many columns to shift over.

cluster_overlap_dict = {}
cluster_overlap_dict[3]=2
    
A = CreateArtificialDataset(974,1914,.005,4,cluster_overlap_dict)

idx = AddNoise(A,0.2)

np.savetxt("../Data/Artificial/artificial50pct.csv",A,delimiter=",",fmt="%d")