import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import mahalanobis 
from scipy.stats import chi2
from joblib import load

df=pd.read_csv("marketing_labels.csv")
df_pca=pd.read_csv("marketing_pca_labels.csv")
##titulo
st.title("Customer Personality Analysis")

#descripcion
st.write("""
Customer Personality Analysis is a detailed analysis of a company ideal customers. It helps a business to better understand its customers and makes it easier for them to modify products according to the specific needs, behaviors and concerns of different types of customers.
Customer personality analysis helps a business to modify its product based on its target customers from different types of customer segments. For example, instead of spending money to market a new product to every customer in the company’s database, a company can analyze which 
customer segment is most likely to buy the product and then market the product only on that particular segment.

https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis
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

df["mahal"]=mahalanobis(df.drop(columns="labels"),df.drop(columns="labels"))  #we create a distance vector. so we can find values that are far from the average(outliers)
df["p_valor"]=1-chi2.cdf(df["mahal"],df.shape[1]-2) #we calculate p_valor to make an hyphotesis test. where the null hyphotesis , its that no value is an outlier
df=(df[df["p_valor"]>=0.01]).drop(columns=["mahal","p_valor"])

@st.cache(suppress_st_warning=True)
def function():
    for i in df.columns[:-1]:
        fig,ax=plt.subplots(1,4,figsize=(10,4),sharex=True,sharey=True)
        for j in df["labels"].unique():
            fil=df[df["labels"]==j]
            sns.histplot(fil[i],ax=ax[j],stat="probability")
        st.pyplot(fig)
function()

st.write("""

Conclusion

In the first graphic "Income",we can see people on clusters 0 and 2 have less income and people on clusters 1 and 3 have more income.

Comparing these data with the next cluster we can see that most people with higher income also have one or no kids at home (clusters 1 and 3) and also no teenager kids.

We can see that clusters 0 and 2 almost don't buy by catalog and cluster 1 and 3 prefer to buy in the store.

Clusters 0 and 2 are the ones that less campaigns accept.
""")

#modelo de clasificacion: every time a new custoemr comes, the model classifies it in one or another cluster

#introduccion de datos de nuevos clientes

st.header("Model")
income=st.slider("Introduce your income:",0,200000)
kid_home=st.slider("Introduce kids at home:",0,5)
teen_home=st.slider("Introduce teens at home:",0,5)
Recency=st.slider("Number of days since customer's last purchase:",0,100)
Mnt_Total=st.slider("Introduce the global amount of purchases:",0,5000)
age=st.slider("Introduce your age:",0,100)
Total_Camp_Accepted=st.slider("Introduce the total of marketing campaigns accepted:",0,5)
NumWebPurchases=st.slider("Introduce the number of purchases made through the company’s website:",0,10)
NumCatalogPurchases=st.slider("Introduce the number of purchases made using a catalogue:",0,10)
NumStorePurchases=st.slider("Introduce the number of purchases made directly in stores:",0,10) 
NumWebVisitsMonth=st.slider("ntroduce the number of visits to company’s website in the last month:",0,10)
NumDealsPurchases=st.slider("Introduce the number of purchases made with a discount:",0,10)
Customer_Years=st.slider("Introduce thee number of years being a customer:",0,10)

data=np.array([[income,
        kid_home,
        teen_home,
        Recency,
        NumDealsPurchases,
        NumWebPurchases,
        NumCatalogPurchases,
        NumStorePurchases,
        NumWebVisitsMonth,
        age,
        Mnt_Total,
        Total_Camp_Accepted,
        Customer_Years]])

xgb=load("model_xgb.joblib")
if st.button("Predict"):
    pred=xgb.predict(data)
    st.write(f"This customer belongs to cluster number:{pred[0]}")
else:
    st.write("Without prediction")


