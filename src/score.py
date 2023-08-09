import pickle
import pandas as pd
import numpy as np
from inference import predict
from sklearn.metrics import roc_auc_score, classification_report, recall_score, accuracy_score
from data import get_data
import json
with open("config.json","r") as f:
    CONFIG=json.load(f)

SPLITS=CONFIG["SPLITS"]


lrs=[]
thresholds=[]

for i in np.arange(SPLITS):
    with open(f'assets/lr_{i}.pkl','rb') as f:
        lr=pickle.load(f)
        lrs.append(lr)

with open('assets/thresholds.txt','r') as f:
    thresholds=f.read().split(',')
thresholds = [float(x) for x in thresholds]

def score(lrs):

    X_test, y_test = get_data('test')
    preds=[]

    for row in X_test.iterrows():
        pred,_=predict(row[1].values, flag="test")
        preds.append(pred)

    print(roc_auc_score(y_test, preds))
    print(classification_report(y_test, preds))

    with open("score.txt","w") as f:
        f.write(classification_report(y_test,preds))
        f.write("\n\nROC_AUC score="+str(roc_auc_score(y_test, preds)))
        f.write("\n\nRecall score="+str(recall_score(y_test, preds)))
        f.write("\n\nAccuracy score="+str(accuracy_score(y_test, preds)))

if __name__=="__main__":

    score(lrs)