# Data Util Libs
import pandas as pd 
import numpy as np 
# ML Libs
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline


##########################################
# Globals 

# Scoring Metrics
scorers = {
        "Accuracy":'accuracy',"AUC":'roc_auc',
        'F1':'f1','bAcc':'balanced_accuracy',
        'precision':'precision'
    }

# Results Collector
full_results = pd.DataFrame()

# Resampling Switch
# 0 = Undersampling
# 1 = Oversampling
resamping = 1 

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
# Feature Scaling
scaler = MinMaxScaler()
df = scaler.fit_transform(df)
##########################################

##########################################
# Base Classifiers
clf_names = ["Random Forest","Linear SVM","MLP"]
rf_clf = RandomForestClassifier(n_estimators=50,criterion='entropy',min_samples_split=3)
svm_clf = LinearSVC(C=1.0)
mlp_clf = MLPClassifier(hidden_layer_sizes=(8,),activation='relu',solver='adam')
clfs = [rf_clf,svm_clf,mlp_clf]
##########################################

##########################################
# Wrapper Feature Selection
wrp_names = ["Linear SVM","Random Forest"]
svm_wrapper = SelectFromModel(svm_clf,threshold='mean').fit(df,target)
rf_wrapper = SelectFromModel(rf_clf,threshold='mean').fit(df,target)
wrps = [svm_wrapper,rf_wrapper]
##########################################

##########################################
# Resampling 

if resamping == 0:
    rs = RandomUnderSampler()    
if resamping == 1:
    rs = RandomOverSampler()

df,target = rs.fit_resample(df,target)
##########################################

##########################################
# Cross Validation Stategy 
cv = StratifiedKFold(n_splits=5)
##########################################

##########################################
# 5-Fold Cross Validation 
for wrp_name, wrp in zip(wrp_names,wrps):
    # Transform Dataset
    df_sub = wrp.transform(df)
    k = len([x for x in wrp.get_support() if x])
    # Cross Validate Classifiers
    # Random Forest ( n_trees=50 )
    # Linear SVM ( C=1.0 )
    # MLP ( Hidden Layer = (8,) )
    for clf_name,clf in zip(clf_names,clfs):
        grid = GridSearchCV(clf,param_grid={},scoring=scorers,refit='precision',cv=cv)
        grid.fit(df_sub,target)

        cv_res = pd.DataFrame(grid.cv_results_)
        cv_res['Classifier'] = clf_name
        cv_res['Selection Method'] = wrp_name 
        cv_res['Num Features'] = k
        cv_res['Features Selected'] = '|'.join([f for f,u in zip(feature_names,wrp.get_support()) if u])
        full_results = full_results.append(cv_res,ignore_index=True)
        for _,row in cv_res.iterrows():
            print("Number of Features: {}".format(k))
            print("{} Metrics".format(clf_name))
            print("\tAccuracy: {:0.3f} (+/- {:0.3f})".format(row['mean_test_Accuracy'],2*row['std_test_Accuracy']))
            print("\tBalanced Acc: {:0.3f} (+/- {:0.3f})".format(row['mean_test_bAcc'],2*row['std_test_bAcc']))
            print("\tAUC: {:0.3f} (+/- {:0.3f})".format(row['mean_test_AUC'],2*row['std_test_AUC']))
            print("\tF1: {:0.3f} (+/- {:0.3f})".format(row['mean_test_F1'],2*row['std_test_F1']))
            print("\tPrecision: {:0.3f} (+/- {:0.3f})".format(row['mean_test_precision'],2*row['std_test_precision']))
            print("\tMean Fit Time: {:0.4f}".format(row['mean_fit_time']))

full_results.to_csv("./Final_Results/Wrapper_Selection_Results_Oversample.csv",index=False)
##########################################