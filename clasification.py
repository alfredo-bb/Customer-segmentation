import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from joblib import dump

df=pd.read_csv("marketing_labels.csv")

#we create the matrix X,Y whre we are gonna run the model
X=df.copy()
y=X.pop("labels")

#we introduce the most common XGBOOST parameters to calibrate the XGBOOST model
params={"min_child_weight":[1,5,10],
        "max_depth":[3,4,5]}

#Grid search combinates all the parameters above and searchs for the best combination 
xgb=XGBClassifier(learning_rate=0.02,n_estimators=600,objective="multi:softmax")
xgb_gs=GridSearchCV(xgb,params,cv=5)
xgb_gs.fit(X,y)

 
xgb=XGBClassifier(**xgb_gs.best_params_)

xgb.fit(X,y)
dump(xgb,"model_xgb.joblib")