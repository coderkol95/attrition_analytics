import pickle
import shap
import pandas as pd
import numpy as np
from train import get_data
import pickle
SPLITS=5

lrs=[]
thresholds=[]

for i in np.arange(SPLITS):
    with open(f'assets/lr_{i}.pkl','r') as f:
        lr=pickle.load(f)
        lrs.append(lr)

with open('assets/thresholds.txt','r') as f:
    thresholds=f.read().split(',')

X_trains,_ = get_data('train')

def predict(data):

    if df.ndim==1:
        df = df.reshape(1,-1)

    probs=[]
    for i in np.arange(5):
        pred=lrs[i].predict_proba(df)[:,1]
        probs.append(pred)

    if (probs[0]>thresholds[0]) | (probs[1]>thresholds[1]) or (probs[2]>thresholds[2]) or (probs[3]>thresholds[3]) or (probs[4]>thresholds[4]):
        idx = probs.index(max(probs))
        flag='positive'
        [prediction]=1
    else:
        idx = probs.index(min(probs))
        flag='negative'
        prediction=0
    model = lrs[idx]
    train = X_trains[idx]
    explanation=explain_causes(model, train, df, flag)
    explanation

    return prediction, explanation

def explain_causes(model,train,data, flag):

    cols= [numerical_feats+categorical_feats]
    explainer = shap.Explainer(model, pd.DataFrame(train,columns=cols), feature_names=cols)
    shap_values = explainer(data)
    print(shap_values.values)
    causes=pd.Series(shap_values.values[0], index=[numerical_feats+categorical_feats])
    if flag=='positive':
        causes=causes[causes>0]/sum(causes[causes>0]) *100
    else:
        causes=causes[causes<0]/sum(causes[causes<0]) *100

    return causes.sort_values(ascending=False)[:5]