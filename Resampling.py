import numpy as np 
import pandas as pd 

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from imblearn.ensemble import BalancedRandomForestClassifier
# Utilities
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
# Resampling
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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
##########################################

##########################################
# Resamplers
over_rs = RandomOverSampler()
under_rs = RandomUnderSampler()
##########################################

##########################################
# Models
rf_clf = RandomForestClassifier(n_estimators=20,criterion='entropy',min_samples_split=3)
svm_clf = LinearSVC(C=1.0)
mlp_clf = MLPClassifier(hidden_layer_sizes=(8,))
brf_clf = BalancedRandomForestClassifier(n_estimators=20,criterion='entropy',min_samples_split=3)
##########################################

##########################################
# Pipelines
pipe = Pipeline(steps=[
    ('rs',over_rs),
    ('scaling',scaler),
    ('clf',rf_clf)
])
#print(pipe.get_params().keys())
##########################################

##########################################
# Cross Validation (Resampling)
params = {
    'rs':[over_rs,under_rs],
    'clf':[rf_clf,mlp_clf,svm_clf]
}
spliter = StratifiedKFold(n_splits=5)
grid = GridSearchCV(estimator=pipe,param_grid=params,scoring=scorers,cv=spliter,refit=False,n_jobs=-2)
grid.fit(df,target)
res = pd.DataFrame(grid.cv_results_)
res.to_csv('./Results/Resample_Results.csv',index=False)
##########################################

##########################################
# Balanced Random Forest
grid = GridSearchCV(estimator=brf_clf,cv=spliter,refit=False,param_grid={},scoring=scorers,n_jobs=-2)
grid.fit(df,target)
res = pd.DataFrame(grid.cv_results_)
res.to_csv('./Results/Balanced_RF_Results.csv',index=False)
##########################################