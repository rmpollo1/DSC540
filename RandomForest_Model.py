import numpy as np 
import csv

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler

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
data_train, data_validate, target_train, target_validate = train_test_split(data_np,target_np,train_size=10_000)
##########################################
# Grid Search
params = {"n_estimators":[5,10,20,50,100]}
scorers = {"Accuracy":'accuracy',"AUC":'roc_auc'}
clf = RandomForestClassifier()

rus = RandomUnderSampler()
data_res, target_res = rus.fit_resample(data_train, target_train)

grid = GridSearchCV(clf,param_grid=params,scoring=scorers,cv=5,refit='Accuracy')
grid.fit(data_res,target_res)
print(grid.cv_results_)

print(grid.score(data_validate,target_validate))