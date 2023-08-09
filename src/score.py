import pickle
import pandas as pd
import numpy as np
from inference import predict
from sklearn.metrics import roc_auc_score, classification_report
from data import get_data
SPLITS=5
lrs=[]
thresholds=[]

for i in np.arange(SPLITS):
    with open(f'assets/lr_{i}.pkl','r') as f:
        lr=pickle.load(f)
        lrs.append(lr)

with open('assets/thresholds.txt','r') as f:
    thresholds=f.read().split(',')

def score(lrs):

    X_test, y_test = get_data('test')
    preds=[]

    for row in X_test.iterrows():
        pred,_=predict(row[1].vaues)
        preds.append(pred)

    print(roc_auc_score(y_test, preds))
    print(classification_report(y_test, preds))

    with open("score.txt","w") as f:
        f.write(classification_report(y_test,preds))
        f.write(roc_auc_score(y_test, preds))