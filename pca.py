# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 07:14:20 2022

@author: End User
"""

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing





p = pd.read_excel("C:/Users/End User/Documents/Machine Learning/Final_dataset.xlsx")

p.pop('Unnamed: 0')

for i in range(len(p["property_type"])):
    if p["property_type"][i]=="Departamento":
        p["property_type"][i]=0
    if p["property_type"][i]=="Casa":
        p["property_type"][i]=1

for i in range(len(p['operation_type'])):
    if p['operation_type'][i]=="Renta":
        p['operation_type'][i]=0
    if p['operation_type'][i]=="venta":
        p['operation_type'][i]=1

for i in range(len(p['due_date'])):
    p['due_date'][i]=int(p['due_date'][i].replace('-',''))

data = p

scaled_data = preprocessing.scale(data, axis=0, with_mean=True, with_std=True, copy=True)

pca = PCA()

pca=pca.fit(scaled_data)

pca_data = pca.transform(scaled_data)

per_var = np.round(pca.explained_variance_ratio_*100,decimals=1)


labels = ['PC'+ str(x) for x in range(1, len(per_var)+1)]

plt.bar(x=range(1, len(per_var)+1), height = per_var, tick_label=labels)
plt.ylabel("Percentage of Explained Variance")
plt.xlabel("Principal component")
plt.title("Scree Plot")
plt.show()

pca_df = pd.DataFrame(pca_data,columns = labels)

plt.scatter(pca_df.PC2, pca_df.PC3)
plt.title("PCA Graph")
plt.xlabel("PC1-{0}%".format(per_var[1]))
plt.ylabel("PC4-{0}%".format(per_var[2]))