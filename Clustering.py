# Clustering    : hierarchical, k-means

############################################################################################################################################
#######################  I> IMPORT THE LIBRARIES  ########################################################################################## 
############################################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as sm


############################################################################################################################################
#######################  II> IMPORT THE DATASET & DATA PREPROCESSING   ##################################################################### 
############################################################################################################################################

dataset = pd.read_csv('E:\\DESK PROJECTS\\MACHINE LEARNING SUMMARY\\ML DATASETS\\Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values


############################################################################################################################################
######################  III> FIND OPTIMAL NO. OF CLUSTERS  #################################################################################
############################################################################################################################################

import scipy.cluster.hierarchy as sch                                      # DENDOGRAM for HIERARCHICAL
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

from sklearn.cluster import KMeans                                         # ELBOW METHOD for K-MEANS
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


############################################################################################################################################
####################### IV> FIT ML MODEL TO TRAINING SET ###################################################################################
############################################################################################################################################

from sklearn.cluster import AgglomerativeClustering                                    # HIERARCHICAL
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)                 # K-MEANS
y_kmeans = kmeans.fit_predict(X)

                                            
############################################################################################################################################
####################### V> VISUALIZE TRAINING SET RESULTS ##################################################################################
############################################################################################################################################

plt.scatter(X[y_hc/y_kmeans == 0, 0], X[y_hc/y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')#y_hc=hierarchical & y_kmeans=k-means
plt.scatter(X[y_hc/y_kmeans == 1, 0], X[y_hc/y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc/y_kmeans == 2, 0], X[y_hc/y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc/y_kmeans == 3, 0], X[y_hc/y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc/y_kmeans == 4, 0], X[y_hc/y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids') # ONLY for k-means

plt.title('Clusters of customers')   # for hierarchical & k-means
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
