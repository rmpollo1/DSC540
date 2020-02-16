import numpy as np 
import pandas as pd 
import csv

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, SelectKBest


from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

##########################################
# Global Parameters
rand_st = 132
resampling = 0                  # Resampling Switch
resampling_method = 1           # Undersampling = 1; Oversampling = 2
fs = 0                          # Feature Selection Switch
fs_method = 1                   # Feature selection Method
params = {}                     # Grid Search Parameters
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


##########################################
# Feature Selection
if fs == 1:
    pass
#
##########################################

##########################################
#  Train / Test Split
data_train, data_test, target_train, target_test = train_test_split(data_np,target_np,train_size=0.2,random_state=rand_st)
##########################################

##########################################
# Resampling 
if resampling == 1:
    print("--- Resampling ON ---")
    if resampling_method == 1:
        print("--- Undersampling ON ---")
        rs = RandomUnderSampler(random_state=rand_st)
    if resampling_method == 2:
        print("--- Oversampling ON ---")
        rs = RandomOverSampler(random_state=rand_st)
    data_train, target_train = rs.fit_resample(data_train,target_train)
    print("Resampled Training Data Samples: {}".format(len(target_train)))

##########################################

##########################################
# Grid Search
params['strategy'] = ['stratified','most_frequent','uniform']
scorers = {"Accuracy":'accuracy',"AUC":'roc_auc'}
cross_val = StratifiedKFold(n_splits=5,random_state=rand_st)
clf = DummyClassifier(random_state=rand_st)

grid = GridSearchCV(clf,param_grid=params,scoring=scorers,cv=cross_val,refit='Accuracy')
grid.fit(data_train,target_train)
rf_results = pd.DataFrame(grid.cv_results_)
rf_results.to_csv("./DummyResults.csv",index=False)

for _,row in rf_results.iterrows():
    print("Dummy Metrics (Strategy {})".format(row['param_strategy']))
    print("\tAccuracy: {:0.3f} (+/- {:0.3f})".format(row['mean_test_Accuracy'],2*row['std_test_Accuracy']))
    print("\tAUC: {:0.3f} (+/- {:0.3f})".format(row['mean_test_AUC'],2*row['std_test_AUC']))
    print("\tRuntime: {:0.4f}".format(row['mean_fit_time']))
##########################################

##########################################
# Test Performance
print("--- Test Set Performance ---")
target_test_pred = grid.predict(data_test)
acc = accuracy_score(target_test,target_test_pred)
auc = roc_auc_score(target_test,target_test_pred)
print("Random Forest Test Accuracy: {}".format(acc))
print("Random Forest Test AUC: {}".format(auc))
##########################################

