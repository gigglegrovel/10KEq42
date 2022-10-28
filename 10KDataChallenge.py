#%%
import pandas as pd
import json
# %%
datos = pd.read_csv("C:/Users/End User/Documents/GitHub/MapatonAgua/files/UnionRegiosDOF.csv")

# %%
df = pd.DataFrame(datos)
# %%
lista = df["USO QUE AMPARA EL TITULO"].unique()
print(lista)
# %%
for i in range(len(lista)):
    for j in range(len(df["USO QUE AMPARA EL TITULO"])):
        if df["USO QUE AMPARA EL TITULO"][j]==lista[i]:
            df["USO QUE AMPARA EL TITULO"][j]=i
# %%
datos_use = [1,5,8,9,16,18,20,22,24,41,42,43,44]
datos_alv=list(range(45))
for i in range(len(datos_use)):
    datos_alv.pop(datos_alv.index(datos_use[i]))
    
#%%
print(df.columns[2])
# %%
counter = 0
for i in range(1,len(datos_alv)):
    df = df.drop(df.columns[datos_alv[i]-counter+1],axis=1)
    counter+=1
# %%
df = df.drop('TITULO',axis=1)
# %%
df = df.drop('Unnamed: 0',axis=1)
# %%
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
#%%
scaled_data = preprocessing.scale(df, axis=0, with_mean=True, with_std=True, copy=True)

# %%
pca = PCA()

pca=pca.fit(scaled_data)

pca_data = pca.transform(scaled_data)

per_var = np.round(pca.explained_variance_ratio_*100,decimals=1)
# %%
labels = ['PC'+ str(x) for x in range(1, len(per_var)+1)]

plt.bar(x=range(1, len(per_var)+1), height = per_var, tick_label=labels)
plt.ylabel("Percentage of Explained Variance")
plt.xlabel("Principal component")
plt.title("Screen Plot")
plt.show()
#%%
pca_df = pd.DataFrame(pca_data,columns = labels)
x_ax = pca_df.PC1
y_ax = pca_df.PC2
#%%
x_ax.pop(6585)
y_ax.pop(6585)
x_ax.pop(1187)
y_ax.pop(1187)
x_ax.pop(712)
y_ax.pop(712)
#%%
plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title("PCA Graph")
plt.xlabel("PC1-{0}%".format(per_var[0]))
plt.ylabel("PC2-{0}%".format(per_var[1]))
# %%
plt.hist(df['USO QUE AMPARA EL TITULO'])
# %%
import statistics
print(statistics.stdev(df['VOLUMEN ANUAL EN m3']))
# %%
from scipy.stats import skewtest
# %%
from scipy.stats import kurtosis
print(kurtosis(df['VOLUMEN ANUAL EN m3']))
# %%
