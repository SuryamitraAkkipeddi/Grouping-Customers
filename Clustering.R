###########################################################################################
#################################### I> INSTALL PACKAGES ##################################
###########################################################################################

install.packages('caTools') 
install.packages('ggplot2')
install.packages('e1071')
install.packages('rpart')
install.packages('randomForest')
install.packages('ElemStatLearn')
install.packages('class')
install.packages('cluster')
install.packages('arules')
install.packages('caret')
install.packages('kernlab')
install.packages('MASS')
install.packages('xgboost')

###########################################################################################
#################################### II> LOAD THE LIBRARIES ###############################
###########################################################################################

library(caTools)
library(ggplot2)
library(e1071)
library(rpart)
library(randomForest)
library(ElemStatLearn)
library(class)
library(cluster)
library(arules)
library(caret)
library(kernlab)
library(MASS)
library(xgboost)
set.seed(123)

###########################################################################################
#################################### III> IMPORT THE DATASET  #############################
###########################################################################################

dataset = read.csv('E:\\MACHINE LEARNING SUMMARY\\ML DATASETS\\Mall_Customers.csv')  
summary(dataset)

###########################################################################################
################################### IV> OMIT UNNECESSARY DATA #############################
###########################################################################################

dataset = dataset[4:5]

###########################################################################################
################################## V> FIND OPTIMAL NO. OF CLUSTERS ########################
###########################################################################################

wcss = vector()                                                                                       # ELBOW METHOD in K-MEANS CLUSTERING
for (i in 1:10) wcss[i] = sum(kmeans(dataset, i)$withinss)
plot(1:10,wcss,type = 'b',main = paste('The Elbow Method'),xlab = 'Number of clusters',ylab = 'WCSS')

dendrogram = hclust(d = dist(dataset, method = 'euclidean'), method = 'ward.D')                       # DENDOGRAM in HIERARCHICAL CLUSTERING
plot(dendrogram,main = paste('Dendrogram'),xlab = 'Customers',ylab = 'Euclidean distances')

###########################################################################################
################################## VI> FIT/TRAIN ML MODEL TO THE TRAINING SET #############
###########################################################################################

kmeans = kmeans(x = dataset, centers = 5)                               # k-means clustering
y_kmeans = kmeans$cluster

hc = hclust(d = dist(dataset, method = 'euclidean'), method = 'ward.D') # hierarchical clustering
y_hc = cutree(hc, 5)

###########################################################################################
################################### VII> VISUALIZE THE CLUSTERS  ##########################
###########################################################################################

clusplot(dataset,y_kmeans/y_hc,lines = 0,shade = TRUE,color = TRUE,labels = 2,plotchar = FALSE,span = TRUE,main = paste('Clusters of customers'),
         xlab = 'Annual Income',ylab = 'Spending Score') 

