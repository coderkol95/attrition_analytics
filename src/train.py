import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, classification_report, roc_auc_score
import pickle
from data import get_data
import json
with open("config.json","r") as f:
    CONFIG=json.load(f)

RANDOM_STATE=CONFIG["RANDOM_STATE"]
SPLITS=CONFIG["SPLITS"]


def train_LR_models():

    lrs=[]
    thresholds=[]
    X_trains, y_trains=get_data('train')
    X_val, y_val = get_data('val')

    for i in np.arange(SPLITS):
        model=LogisticRegression(random_state=RANDOM_STATE).fit(X_trains[i],y_trains[i])
        fpr,tpr,thresh=roc_curve(y_val, model.predict_proba(X_val)[:,1], )
        pos=np.argmax(tpr-fpr)
        thr=thresh[pos]

        lrs.append(model)
        thresholds.append(thr)

    return lrs, thresholds

if __name__=="__main__":

    lrs,thresholds=train_LR_models()
    thresholds = [str(x) for x in thresholds]

    for i,lr in enumerate(lrs):
        with open(f'assets/lr_{i}.pkl','wb') as f:
            pickle.dump(lr,f)

    with open('assets/thresholds.txt','w') as f:
        f.write(",".join(thresholds))

