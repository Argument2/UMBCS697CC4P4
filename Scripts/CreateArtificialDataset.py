import numpy as np

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
#----- Main Program -----

cluster_overlap_dict = {}
cluster_overlap_dict[3]=2
    
A = CreateArtificialDataset(974,1914,.005,4,cluster_overlap_dict)

np.savetxt("../Data/Artificial/artificial.csv",A,delimiter=",",fmt="%d")