import sys
import numpy as np 
import pandas as pd 
import csv

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, balanced_accuracy_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

##########################################
# Global Parameters
resampling = 1                  # Resampling Switch
resampling_method = 1           # Undersampling = 1; Oversampling = 2
fs = 1                          # Feature Selection Switch
fs_method = 3                   # Feature Selection Method
fs_scorer = mutual_info_classif                # Filter Selection Scorer
#
#
##########################################

##########################################
# Cache Results
try:
    res_file = sys.argv[1]
except IndexError as e:
    res_file = "RandomForest_Results.csv"
    print("No Result File Supplied saving results in RandomForest_Results.csv")

#
#
##########################################

##########################################
# Data Import
#
target_idx = 0 
feat_start = 1

f = csv.reader(open("./Data/Training_Data.csv"), delimiter = ',', quotechar = '"')

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
# Base Estimator (Random Forest)
clf = RandomForestClassifier(n_estimators=20,min_samples_split=3,n_jobs=-2)

##########################################

##########################################
# Feature Scaling
data_np = MinMaxScaler().fit_transform(data_np)

##########################################

##########################################
# Feature Selection
if fs == 1: 
    # Apply Feature Selection 

    if fs_method == 1:
        print("--- LV Filter ON ---")
        flt = VarianceThreshold(threshold=0.5)
        data_np = flt.fit_transform(data_np,target_np)
    
    if fs_method == 2:
        print("--- Feature Selection ON ---")
        flt = SelectKBest(score_func=fs_scorer,k=30)
        data_np = flt.fit_transform(data_np,target_np)
    
    if fs_method == 3:
        print("--- Embedded Feature Selection ON ---")
        flt = SelectFromModel(clf, prefit=False, threshold='mean', max_features=None)
        data_np = flt.fit_transform(data_np,target_np)

    selected = []
    removed = []
    for i,c in zip(flt.get_support(),header[1:]):
        if i:
            selected.append(c)
        else:
            removed.append(c)
    print("Features Selected: ",selected)
    print("Features Removed: ",removed)
    print("Selected/Removed/Total: (",len(selected),len(removed),len(header)-1,")")
#
##########################################

##########################################
# Resampling 
if resampling == 1:
    print("--- Resampling ON ---")
    if resampling_method == 1:
        print("--- Undersampling ON ---")
        rs = RandomUnderSampler()
    if resampling_method == 2:
        print("--- Oversampling ON ---")
        rs = RandomOverSampler()
    data_np, target_np = rs.fit_resample(data_np,target_np)
    print("Resampled Training Data Samples: {}".format(len(target_np)))

##########################################

##########################################
# Grid Search
params = {"n_estimators":[5,10,20,50,100]}
scorers = {
    "Accuracy":'accuracy',"AUC":'roc_auc',
    'F1':'f1','bAcc':'balanced_accuracy',
    'precision':'precision'
}
cross_val = StratifiedKFold(n_splits=5)

grid = GridSearchCV(clf,param_grid=params,scoring=scorers,cv=cross_val,refit='precision')
grid.fit(data_np,target_np)
rf_results = pd.DataFrame(grid.cv_results_)
rf_results.to_csv(res_file,index=False)

for _,row in rf_results.iterrows():
    print("Random Forest Metrics (Number of Trees {})".format(row['param_n_estimators']))
    print("\tAccuracy: {:0.3f} (+/- {:0.3f})".format(row['mean_test_Accuracy'],2*row['std_test_Accuracy']))
    print("\tBalanced Acc: {:0.3f} (+/- {:0.3f})".format(row['mean_test_bAcc'],2*row['std_test_bAcc']))
    print("\tAUC: {:0.3f} (+/- {:0.3f})".format(row['mean_test_AUC'],2*row['std_test_AUC']))
    print("\tF1: {:0.3f} (+/- {:0.3f})".format(row['mean_test_F1'],2*row['std_test_F1']))
    print("\tPrecision: {:0.3f} (+/- {:0.3f})".format(row['mean_test_precision'],2*row['std_test_precision']))
    print("\tMean Fit Time: {:0.4f}".format(row['mean_fit_time']))
##########################################


