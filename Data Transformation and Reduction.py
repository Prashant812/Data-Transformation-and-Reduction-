# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 21:51:18 2021

@author: Prashant Kumar
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing as prp
from sklearn.decomposition import PCA



df = pd.read_csv("pima-indians-diabetes.csv")
#print(df.head())
data = df.drop(['class'], axis = 1)
#print(data)



#Replacing the outliers with the median of the respective attributes :-
attributesfr = list(data.columns)
def outliers(x):  #Function for outliers
    IQR = np.percentile(data[x],75) - np.percentile(data[x],25)
    minimum= np.percentile(data[x],25) - (1.5*IQR)#conditions for outliers
    maximum= np.percentile(data[x],75) + (1.5*IQR)
    outliers_=pd.concat((data[x][data[x]< minimum],data[x][data[x]> maximum]))
    return outliers_


for i in attributesfr:
    data[i].replace(data[i][list(outliers(i).index)],data[i].median(),inplace = True)

#Min-Max normalization of the outlier corrected data to scale the attribute values in the range 5 to 12 :-
min_max_scaler = prp.MinMaxScaler(feature_range=(5, 12))
x_scaled = min_max_scaler.fit_transform(data)
minMaxNormalized = pd.DataFrame(x_scaled)
minMaxNormalized.rename(columns={i: list(data)[i] for i in range(8)}, inplace=True)

print(minMaxNormalized)

#Standardization each selected attribute using the relation 𝑥̂n= (xn − μ)/σ where μ is mean and σ is standard deviation of that attribute :-


def z_score(x):
    x_sc = (x - x.mean())/x.std()
    return x_sc 

print(z_score(data))
    

#Data Reduction using PCA

from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(data)
#print(X_scaled)

features = X_scaled.T
cov_matrix = np.cov(features)


values, vectors = np.linalg.eig(cov_matrix)
print(values)
print(vectors)
explained_variances = []
for i in range(len(values)):
    explained_variances.append(values[i] / np.sum(values))
 
print(np.sum(explained_variances), '\n', explained_variances)

projected_1 = X_scaled.dot(vectors.T[0])
projected_2 = X_scaled.dot(vectors.T[1])
res = pd.DataFrame(projected_1, columns=['PC1'])
res["PC2"] = projected_2
print(res)

#Using library
pca = PCA(n_components=2)
pca.fit(X_scaled)
prComp = pca.fit_transform(X_scaled)
prDf = pd.DataFrame(data=prComp, columns=['X1', 'X2'])
print(prDf)



