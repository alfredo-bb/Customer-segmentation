from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

import pandas as pd

#import dataframe

df=pd.read_csv("marketing_cleaned.csv")
#cleaning categorical variables

df.drop(columns=["Education",'Marital_Status'],inplace=True)
#standarize variables
scaler=StandardScaler()
scalex=scaler.fit_transform(df)
#PCA with 2 principal components
pca=PCA(n_components=2)
df_pca=pca.fit_transform(scalex)
#Kmeans
kmeans=KMeans(n_clusters=4)
kmeans.fit(df_pca)
#Agregate labels to DF
df["labels"]=kmeans.labels_
df_pca=pd.DataFrame(df_pca)
df_pca["labels"]=kmeans.labels_

centroides=kmeans.cluster_centers_
cen_x=[i[0] for i in centroides]
cen_y=[i[1] for i in centroides]
df_pca["cen_x"]=df_pca["labels"].map({0:cen_x[0],1:cen_x[1],2:cen_x[2],3:cen_x[3]})
df_pca["cen_y"]=df_pca["labels"].map({0:cen_y[0],1:cen_y[1],2:cen_y[2],3:cen_y[3]})

df.to_csv("marketing_labels.csv",index=False)
df_pca.to_csv("marketing_pca_labels.csv",index=False)