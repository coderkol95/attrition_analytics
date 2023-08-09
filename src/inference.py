import pickle
import shap
import pandas as pd
import numpy as np
from data import get_data
import pickle
import json

with open("config.json","r") as f:
    CONFIG=json.load(f)

SPLITS=CONFIG["SPLITS"]
NUMERICAL_FEATS=CONFIG["numerical_feats"]
CATEGORICAL_FEATS=CONFIG["categorical_feats"]

lrs=[]
thresholds=[]

for i in np.arange(SPLITS):
    with open(f'assets/lr_{i}.pkl','rb') as f:
        lr=pickle.load(f)
        lrs.append(lr)
        
with open('assets/thresholds.txt','r') as f:
    thresholds=f.read().split(',')
thresholds = [float(x) for x in thresholds]

with open(f'assets/data_pipe.pkl','rb') as f:
    data_pipe=pickle.load(f)

X_trains,_ = get_data('train')

def predict(data, flag='live'):
    
    if flag=='live':
        if data.ndim==1:
            data = data.reshape(1,-1)
        data = pd.DataFrame(data=data, columns=[NUMERICAL_FEATS+CATEGORICAL_FEATS])
        data=data_pipe.transform(data)
        data = pd.DataFrame(data=data, columns=[NUMERICAL_FEATS+CATEGORICAL_FEATS])

    probs=[]
    for i in np.arange(5):
        pred=lrs[i].predict_proba(data)[:,1]
        probs.append(pred)

    if (probs[0]>thresholds[0]) | (probs[1]>thresholds[1]) or (probs[2]>thresholds[2]) or (probs[3]>thresholds[3]) or (probs[4]>thresholds[4]):
        idx = probs.index(max(probs))
        flag='positive'
        prediction=1
    else:
        idx = probs.index(min(probs))
        flag='negative'
        prediction=0
    model = lrs[idx]
    train = X_trains[idx]
    explanation=explain_causes(model, train, data, flag)

    return prediction, explanation

def explain_causes(model,train,data, flag):

    cols= [NUMERICAL_FEATS+CATEGORICAL_FEATS]
    explainer = shap.Explainer(model, pd.DataFrame(train,columns=cols), feature_names=cols)
    shap_values = explainer(data)
    print(shap_values.values)
    causes=pd.Series(shap_values.values[0], index=cols)
    if flag=='positive':
        causes=causes[causes>0]/sum(causes[causes>0]) *100
    else:
        causes=causes[causes<0]/sum(causes[causes<0]) *100

    return causes.sort_values(ascending=False)[:5]

if __name__=="__main__":

    sample=np.array([1, 0, 14, 3, 3, 2028, 3, 6, 4, 3, 24, 103, 0, 4, 12947, 50, 2, 3,
       28, 3, 2, 5, 2, 'Travel_Rarely', 'Research & Development',
       'Life Sciences', 'Laboratory Technician', 'Single', 'Yes', 'Male',
       'India'])

    pr,exp=predict(sample)
    print(pr,exp)