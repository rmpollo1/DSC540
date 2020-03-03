# Data Util Libs
import pandas as pd 
import numpy as np 
# ML Libs
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline

from imblearn.under_sampling import RandomUnderSampler


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
##########################################

# Read In Training Data
df = pd.read_csv("./Data/Training_Data.csv")

# Seperate Target from Feature
# Convert to Numpy Array
target = np.array(df['approve'])
df = df.drop('approve',axis=1)
feature_names = df.columns 
df = np.array(df)

# Feature Scaling
scaler = MinMaxScaler()
#df = scaler.fit_transform(df)

# Filter Based Feature Selectors 
chi2_flt = SelectKBest(score_func=chi2)
chi2_flt.fit(df,target)
f_flt = SelectKBest(score_func=f_classif)
f_flt.fit(df,target)
mi_flt = SelectKBest(score_func=mutual_info_classif)
mi_flt.fit(df,target)

filters = [chi2_flt,f_flt,mi_flt]
filter_names = ["Chi2","F_Classif","Mutual Info"]

# Resampling 
df, target = RandomUnderSampler().fit_resample(df,target)

# Base Classifiers
clf_names = ["Random Forest","Linear SVM","MLP"]
rf_clf = RandomForestClassifier(n_estimators=20,criterion='gini',min_samples_split=3)
svm_clf = make_pipeline(scaler,LinearSVC(C=1.0)) 
mlp_clf = make_pipeline(scaler,MLPClassifier(hidden_layer_sizes=(8,),activation='relu',solver='adam'))
clfs = [rf_clf,svm_clf,mlp_clf]

# Cross Validation Stategy 
cv = StratifiedKFold(n_splits=5)

# Number of Features to Select
for k in [5,8,10,12,15,20,30,40,len(feature_names)]:
    # Feature Selection Methods 
    # Chi2, F_Classif, & Mutual Info
    for flt_name, flt in zip(filter_names,filters):
        # Set Number of Features to Select
        flt.set_params(**{'k':k})
        # Transform Dataset
        df_sub = flt.transform(df)
    
        # Cross Validate Classifiers
        # Random Forest ( n_trees=20 )
        # Linear SVM ( C=1.0 )
        # MLP ( Hidden Layer = (8,) )
        for clf_name,clf in zip(clf_names,clfs):
            grid = GridSearchCV(clf,param_grid={},scoring=scorers,refit='precision',cv=cv)
            grid.fit(df_sub,target)

            cv_res = pd.DataFrame(grid.cv_results_)
            cv_res['Classifier'] = clf_name
            cv_res['Filter Method'] = flt_name 
            cv_res['Num Features'] = k
            cv_res['Features Selected'] = '|'.join([f for f,u in zip(feature_names,flt.get_support()) if u])
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

full_results.to_csv("./Results/Feature_Selection_Results_3.csv",index=False)