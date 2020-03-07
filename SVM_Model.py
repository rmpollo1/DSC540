import sys
import numpy as np 
import pandas as pd 
import csv

from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
    res_file_2 = sys.argv[2]
except IndexError as e:
    res_file = "./Final_Results/Linear_SVM_Results.csv"
    res_file = "./Final_Results/Kernel_SVM_Results.csv"
    print("No Result File Supplied saving results in ",res_file)

#
#
##########################################

##########################################
# Import Dataset
#
# Read In Training Data
df = pd.read_csv("./Data/Training_Data.csv")

# Seperate Target from Feature
# Convert to Numpy Array
target = np.array(df['approve'])
df = df.drop('approve',axis=1)
feature_names = df.columns 
df = np.array(df)
##########################################

##########################################
# Base Estimator (Linear SVC)
clf = LinearSVC(C=1.0)
clf_2 = SVC(C=1.0,kernel='rbf')
##########################################

##########################################
# Feature Scaling
df = MinMaxScaler().fit_transform(df)

##########################################

##########################################
# Feature Selection
if fs == 1: 
    # Apply Feature Selection 

    if fs_method == 1:
        print("--- LV Filter ON ---")
        flt = VarianceThreshold(threshold=0.2)
        df = flt.fit_transform(df,target)
    
    if fs_method == 2:
        print("--- Feature Selection ON ---")
        flt = SelectKBest(score_func=fs_scorer,k=30)
        df = flt.fit_transform(df,target)
    
    if fs_method == 3:
        print("--- Embedded Feature Selection ON ---")
        flt = SelectFromModel(clf, prefit=False, threshold='mean', max_features=None)
        df = flt.fit_transform(df,target)

    selected = []
    removed = []
    for i,c in zip(flt.get_support(),feature_names):
        if i:
            selected.append(c)
        else:
            removed.append(c)
    print("Features Selected: ",selected)
    print("Features Removed: ",removed)
    print("Selected/Removed/Total: (",len(selected),len(removed),len(feature_names),")")
#
##########################################

##########################################
# Feature Selection
if fs == 1: 
    # Apply Feature Selection 

    if fs_method == 1:
        print("--- LV Filter ON ---")
        flt = VarianceThreshold(threshold=0.16)
        df = flt.fit_transform(df,target)
    
    if fs_method == 2:
        print("--- Feature Selection ON ---")
        flt = SelectKBest(score_func=fs_scorer,k=15)
        df = flt.fit_transform(df,target)
    
    if fs_method == 3:
        print("--- Embedded Feature Selection ON ---")
        flt = SelectFromModel(clf, prefit=False, threshold='mean', max_features=None)
        df = flt.fit_transform(df,target)

    selected = []
    removed = []
    for i,c in zip(flt.get_support(),feature_names):
        if i:
            selected.append(c)
        else:
            removed.append(c)
    print("Features Selected: ",selected)
    print("Features Removed: ",removed)
    print("Selected/Removed/Total: (",len(selected),len(removed),len(feature_names),")")
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
    df, target = rs.fit_resample(df,target)
    print("Resampled Training Data Samples: {}".format(len(target)))

##########################################

##########################################
# Grid Search
params = {'C':np.logspace(-1,1,3)}
scorers = {
    "Accuracy":'accuracy',"AUC":'roc_auc',
    'F1':'f1','bAcc':'balanced_accuracy',
    'precision':'precision'
}
cross_val = StratifiedKFold(n_splits=5)

grid = GridSearchCV(clf,param_grid=params,scoring=scorers,cv=cross_val,refit='precision')
grid.fit(df,target)
rf_results = pd.DataFrame(grid.cv_results_)
rf_results.to_csv(res_file,index=False)

for _,row in rf_results.iterrows():
    print("Linear SVM Metrics (C {})".format(row['param_C']))
    print("\tAccuracy: {:0.3f} (+/- {:0.3f})".format(row['mean_test_Accuracy'],2*row['std_test_Accuracy']))
    print("\tBalanced Acc: {:0.3f} (+/- {:0.3f})".format(row['mean_test_bAcc'],2*row['std_test_bAcc']))
    print("\tAUC: {:0.3f} (+/- {:0.3f})".format(row['mean_test_AUC'],2*row['std_test_AUC']))
    print("\tF1: {:0.3f} (+/- {:0.3f})".format(row['mean_test_F1'],2*row['std_test_F1']))
    print("\tPrecision: {:0.3f} (+/- {:0.3f})".format(row['mean_test_precision'],2*row['std_test_precision']))
    print("\tMean Fit Time: {:0.4f}".format(row['mean_fit_time']))
##########################################


