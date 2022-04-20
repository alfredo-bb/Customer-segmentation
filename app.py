import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df_pca=pd.read_csv("marketing_pca_labels.csv")
##titulo
st.title("Customer Personality Analysis")

#descripcion
st.write("""
Customer Personality Analysis is a detailed analysis of a company ideal customers. It helps a business to better understand its customers and makes it easier for them to modify products according to the specific needs, behaviors and concerns of different types of customers.
Customer personality analysis helps a business to modify its product based on its target customers from different types of customer segments. For example, instead of spending money to market a new product to every customer in the company’s database, a company can analyze which 
customer segment is most likely to buy the product and then market the product only on that particular segment.
""")
#exploracion de los datos
#seleccion de variables
#construccion de los clusters

#visualizacion de los clusters
df_pca=df_pca.sample(frac=0.1)
fig,ax=plt.subplots(1,2,figsize=(8,4),sharex=True,sharey=True)

ax[0].scatter(df_pca.loc[df_pca["labels"]==0,"0"],df_pca.loc[df_pca["labels"]==0,"1"],label="cluster 1")
ax[0].scatter(df_pca.loc[df_pca["labels"]==1,"0"],df_pca.loc[df_pca["labels"]==1,"1"],label="cluster 2")
ax[0].scatter(df_pca.loc[df_pca["labels"]==2,"0"],df_pca.loc[df_pca["labels"]==2,"1"],label="cluster 3")
ax[0].scatter(df_pca.loc[df_pca["labels"]==3,"0"],df_pca.loc[df_pca["labels"]==3,"1"],label="cluster 4")
ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)
ax[0].legend()


ax[1].scatter(df_pca["0"],df_pca["1"],c=df_pca["labels"])
for idx,val in df_pca.iterrows():
    x=[val["0"],val["cen_x"]]
    y=[val["1"],val["cen_y"]]
    ax[1].plot(x,y)

st.pyplot(fig)
#modelo de clasificacion