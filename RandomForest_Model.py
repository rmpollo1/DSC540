import numpy as np 
import pandas as pd 
import csv

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler

##########################################
# Global Parameters
rand_st = 132
#
#
##########################################

##########################################
# Data Import
#
target_idx = 0 
feat_start = 1

f = csv.reader(open("HDMA_Loan_Data_Clean.csv"), delimiter = ',', quotechar = '"')

header = next(f)

#Read data
data=[]
target=[]
for i,row in enumerate(f):
    # For Testing Scipt read first 10k samples
    #if i == 150_000:
    #    break 

    #Load Target
    if row[target_idx]=='':                         #If target is blank, skip row                       
        continue
    else:
        target.append(float(row[target_idx]))       #If pre-binned class, change float to int

    #Load row into temp array, cast columns  
    temp=[]
                 
    for j in range(feat_start,len(header)):
        if row[j]=='':
            temp.append(float())
        else:
            temp.append(float(row[j]))

    #Load temp into Data array
    data.append(temp)
  
#Test Print
print(header)
print(len(target),len(data))
print('\n')

data_np=np.asarray(data)
target_np=np.asarray(target)
#
#
##########################################
data_train, data_validate, target_train, target_validate = train_test_split(data_np,target_np,train_size=1_000,random_state=rand_st)
##########################################
# Grid Search
params = {"n_estimators":[5,10,20,50,100]}
scorers = {"Accuracy":'accuracy',"AUC":'roc_auc'}
clf = RandomForestClassifier(random_state=rand_st)

rus = RandomUnderSampler(random_state=rand_st)
data_res, target_res = rus.fit_resample(data_train, target_train)

cross_val = StratifiedKFold(n_splits=5,random_state=rand_st)

grid = GridSearchCV(clf,param_grid=params,scoring=scorers,cv=cross_val,refit='Accuracy')
grid.fit(data_res,target_res)
rf_results = pd.DataFrame(grid.cv_results_)
rf_results.to_csv("./RandomForestResults.csv",index=False)

for _,row in rf_results.iterrows():
    print("Random Forest Metrics (Number of Trees {})".format(row['param_n_estimators']))
    print("Accuracy: {:0.2f} ( +/- {:0.2f})".format(row['mean_test_Accuracy'],2*row['std_test_Accuracy']))
    print("AUC: {:0.2f} ( +/- {:0.2f})".format(row['mean_test_AUC'],2*row['std_test_AUC']))
    print("Runtime: {:0.4f}".format(row['mean_fit_time']))
