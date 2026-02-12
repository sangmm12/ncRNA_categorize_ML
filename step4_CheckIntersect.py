#check the intersection of label global NV
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.optimize import linprog

def intersection(mutset0, mutset1):
    # Input:
    #      mutset0 is m*col dimensional matrix
    #      mutset1 is n*col dimensional matrix
    # Output:
    #      The two point cloud is disjoint if T = 1
    m, l1 = mutset0.shape
    n, l2 = mutset1.shape
    if l1!=l2:
        print("Error!")
        return
    l=l1
    c = np.ones(m + n)
    A0 = np.hstack((mutset0.T, -mutset1.T))
    a1 = np.concatenate((np.ones(m), np.zeros(n)))
    b1 = np.concatenate((np.zeros(m), np.ones(n)))
    Aeq = np.vstack((A0, a1, b1))
    beq = np.hstack((np.zeros(l), 1, 1))
    lb = np.zeros((m + n,1))
    ub = np.ones((m + n,1))
    B = np.hstack((lb,ub))
    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=B)
    if res.success:
        T = 0
        return True #intersect
    else:
        T = 1
        return False #disjoint
    #return res.fun, T

# Load the CSV file
#file_path = 'mammalian_multispecies/multispecies_train_dev_pre_NV.csv'#Sequence,Label,Source,V1,...,V1368
#file_path = 'mammalian_multispecies/mammalian_train_dev_pre_NV.csv'
#file_path = 'mammalian_multispecies/mammalian_train_dev_pre_BPENV1368_limit30.csv'
#file_path = 'mammalian_multispecies/mammalian_train_dev_pre_BPENV344.csv'
file_path = 'mammalian_multispecies/multispecies_train_dev_pre_BPENV1368.csv'
#file_path = 'train_dev_pre_NV.csv'
#file_path = 'rnacentral_active_top20_2w_NV.csv'
df = pd.read_csv(file_path, index_col=0)
print(df.head())
print('Num Of Label: ', df['Label'].unique())
#print('Num Of Family: ', len(df['Family'].unique()))

# split dataset for each label
hulls = {}
for label in df['Label'].unique():
    print(label)
    if pd.isna(label):  # Check if the label is NaN
        print("Found NaN label, skipping...")
        continue
    #points = df[df['Label'] == label].iloc[:, 3:].values
    #points = df[df['Label'] == label].loc[:, 'V1':'V1368'].values
    points = df[df['Label'] == label].loc[:, 'V1':'V344'].values
    hulls[label] = points

    print(f"Num Of Seq: {len(points)}")
    if len(points) > 0:
        print(f"First Point: {len(points[0])}")
    else:
        print("No points found for this label.")

intersections = {}
labels = list(hulls.keys())
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        label1, label2 = labels[i], labels[j]
        res = intersection(hulls[label1], hulls[label2])
        intersections[(label1, label2)] = res
        #print('intersection: ', label1, ' ', label2, ' ', res)

# Print intersection results
intersections_count = 0
non_intersections_count = 0
for (label1, label2), intersects in intersections.items():
    if intersects:
        intersections_count += 1
        print(f"{label1} and {label2}: Intersect")
    else:
        #print(f"{label1} and {label2}: Do not intersect")
        non_intersections_count += 1


print("Intersections between convex hulls:")
# Print the number of non-intersecting pairs
print(f"Number of non-intersecting pairs: {non_intersections_count}")
print(f"Number of intersecting pairs: {intersections_count}")

