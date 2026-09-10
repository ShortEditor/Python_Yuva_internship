"""
Week 6 Integrative Capstone: Titanic End-to-End Data Science Pipeline
Stages: acquisition, cleaning, EDA, supervised Random Forest, K-Means, evaluation.
"""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

SEED=42
df=pd.read_csv("titanic_cleaned.csv")
df["FarePerPerson"]=df["Fare"]/df["FamilySize"].replace(0,1)
features=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Title","FamilySize","IsAlone","FarePerPerson"]
X,y=df[features],df["Survived"].astype(int)
numeric=["Pclass","Age","SibSp","Parch","Fare","FamilySize","IsAlone","FarePerPerson"]
categorical=["Sex","Embarked","Title"]

prep=ColumnTransformer([
 ("num",SimpleImputer(strategy="median"),numeric),
 ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                  ("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical)
])
model=Pipeline([
 ("prep",prep),
 ("rf",RandomForestClassifier(n_estimators=350,max_depth=7,min_samples_leaf=3,
                              class_weight="balanced",random_state=SEED,n_jobs=-1))
])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,stratify=y,random_state=SEED)
model.fit(Xtr,ytr)
p=model.predict(Xte); pr=model.predict_proba(Xte)[:,1]
print("Accuracy:",accuracy_score(yte,p))
print("Precision:",precision_score(yte,p))
print("Recall:",recall_score(yte,p))
print("F1:",f1_score(yte,p))
print("ROC-AUC:",roc_auc_score(yte,pr))

cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
scores=cross_validate(model,Xtr,ytr,cv=cv,scoring=["accuracy","precision","recall","f1","roc_auc"])
print("CV means:",{m:scores["test_"+m].mean() for m in ["accuracy","precision","recall","f1","roc_auc"]})

cluster_cols=["Age","Fare","FamilySize","Pclass","SibSp","Parch"]
Z=StandardScaler().fit_transform(df[cluster_cols])
sil={}
for k in range(2,7):
    labels=KMeans(n_clusters=k,n_init=20,random_state=SEED).fit_predict(Z)
    sil[k]=silhouette_score(Z,labels)
best_k=max(sil,key=sil.get)
df["Cluster"]=KMeans(n_clusters=best_k,n_init=20,random_state=SEED).fit_predict(Z)
print("Best K:",best_k,"Silhouette:",sil[best_k])
print(df.groupby("Cluster")[cluster_cols+["Survived"]].mean())
PCA(n_components=2,random_state=SEED).fit_transform(Z)
