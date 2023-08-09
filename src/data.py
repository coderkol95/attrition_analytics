import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,StandardScaler
from sklearn.model_selection import train_test_split
RANDOM_STATE=42
SPLITS=5

def get_split_data(data, test_ratio=0.1, val_ratio=0.1):

    X,y=data[numerical_feats+categorical_feats],data['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_ratio, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=val_ratio, random_state=RANDOM_STATE)

    return X_train, X_val, X_test, y_train, y_val, y_test

def transform_data(X_train, X_val, X_test, y_train, y_val, y_test):

    data_pipeline = ColumnTransformer([('encoder',OrdinalEncoder(),categorical_feats), ('scaler',StandardScaler(),numerical_feats)])
    label_enc = LabelEncoder()

    data_pipeline.fit(X_train)
    label_enc.fit(y_train)

    X_train = data_pipeline.transform(X_train)
    X_val = data_pipeline.transform(X_val)
    X_test = data_pipeline.transform(X_test)

    y_train = label_enc.transform(y_train)
    y_val = label_enc.transform(y_val)
    y_test = label_enc.transform(y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def _size_for_undersampled_class(size_of_minority_class, size_of_majority_class):

    mul=len(size_of_majority_class) // len(size_of_minority_class)
    # mul=5, hence 5 datasets and models will be required
    rem=len(size_of_majority_class) - len(size_of_minority_class)*mul
    size=len(size_of_minority_class)+rem//mul

    return size

def sample_data_for_models(X_train, y_train):

    nos=np.where(y_train==0)[0]
    yes=np.where(y_train==1)[0]
    size=_size_for_undersampled_class(yes,nos)

    indices1 = np.random.choice(nos, size=size, replace=False).tolist()
    indices2=np.random.choice(list(set(nos).difference(set(indices1))), size=size, replace=False).tolist()
    indices3=np.random.choice(list(set(nos).difference(set(indices1).union(set(indices2)))), size=size, replace=False).tolist()
    indices4=np.random.choice(list(set(nos).difference(set(indices1).union(set(indices2).union(set(indices3))))), size=size, replace=False).tolist()
    indices5=list(set(nos).difference(set(indices1).union(set(indices2).union(set(indices3).union(set(indices4))))))

    X_train1,y_train1=X_train[list(indices1)+list(yes)], y_train[list(indices1)+list(yes)]
    X_train2,y_train2=X_train[list(indices2)+list(yes)], y_train[list(indices2)+list(yes)]
    X_train3,y_train3=X_train[list(indices3)+list(yes)], y_train[list(indices3)+list(yes)]
    X_train4,y_train4=X_train[list(indices4)+list(yes)], y_train[list(indices4)+list(yes)]
    X_train5,y_train5=X_train[list(indices5)+list(yes)], y_train[list(indices5)+list(yes)]

    return X_train1,y_train1, X_train2, y_train2, X_train3, y_train3, X_train4, y_train4, X_train5, y_train5

def name_the_columns(X,Y):

    XX = [pd.DataFrame(x, columns=[numerical_feats+categorical_feats]) for x in X]

    YY = [pd.DataFrame(y) for y in Y]

    return XX, YY

def prepare_data(data, numerical_feats, categorical_feats):

    X_train, X_val, X_test, y_train, y_val, y_test = get_split_data(data)
    X_train, X_val, X_test, y_train, y_val, y_test = transform_data(X_train, X_val, X_test, y_train, y_val, y_test)
    X_train1,y_train1, X_train2, y_train2, X_train3, y_train3, X_train4, y_train4, X_train5, y_train5=sample_data_for_models(X_train, y_train)

    X,Y=name_the_columns(X=[X_train1, X_train2, X_train3, X_train4, X_train5, X_val, X_test],Y=[y_train1, y_train2, y_train3, y_train4, y_train5, y_val, y_test])

    for i, x in enumerate(X):
        if i<5:
            x.to_csv(f"../data/X_train{i}.csv", index=False)
        if i==5:
            x.to_csv("../data/X_val.csv", index=False)
        if i==6:
            x.to_csv("../data/X_test.csv", index=False)
   
    for i, y in enumerate(Y):
        if i<5:
            y.to_csv(f"../data/y_train{i}.csv", index=False)
        if i==5:
            y.to_csv("../data/y_val.csv", index=False)
        if i==6:
            y.to_csv("../data/y_test.csv", index=False)

def get_data(mode='test'):

    if mode=='train':
        X_trains=[]
        y_trains=[]

        for i in np.arange(SPLITS):
            X_trains.append(pd.read_csv(f'data/X_train{i}.csv'))
            y_trains.append(pd.read_csv(f'data/y_train{i}.csv'))

        return X_trains, y_trains
    
    elif mode=='val':
        X_val = pd.read_csv('data/X_val.csv')
        y_val = pd.read_csv('data/y_val.csv')
        return X_val, y_val

    elif mode=='test':
        X_test = pd.read_csv('data/X_test.csv')
        y_test = pd.read_csv('data/y_test.csv')
        return X_test, y_test

if __name__=="__main__":

    data = pd.read_excel("attrition_analytics/Attrition Data Exercise.xlsx")

    categorical_feats = ["BusinessTravel","Department","EducationField","JobRole","MaritalStatus","OverTime","Gender","Location"]
    numerical_feats = ['JobLevel', 'StockOptionLevel', 'PercentSalaryHike', 'EnvironmentSatisfaction',
    'PerformanceRating', 'MonthlyIncome', 'JobSatisfaction', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsWithCurrManager',
    'DistanceFromHome', 'DailyRate', 'YearsSinceLastPromotion', 'Company Tenure (yrs)', 'MonthlyRate', 'HourlyRate', 'YearsInCurrentRole',
    'WorkLifeBalance', 'Age', 'Education', 'JobInvolvement', 'NumCompaniesWorked', 'RelationshipSatisfaction']

    prepare_data(data, numerical_feats, categorical_feats)