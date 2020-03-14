import numpy as np 
import pandas as pd 

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
# Utilities
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
# Resampling
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
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
n_obs, n_var = df.shape
print("Number of Samples: {} \nNumber of Features: {}".format(n_obs, n_var))
##########################################

##########################################
# Feature Scaling
scaler = MinMaxScaler()
##########################################

##########################################
# Resamplers
over_rs = RandomOverSampler()
under_rs = RandomUnderSampler()
sm = SMOTE()
##########################################

##########################################
# Models
rf_clf = RandomForestClassifier(n_estimators=50,criterion='entropy',min_samples_split=3)
svm_clf = LinearSVC(C=1.0)
mlp_clf = MLPClassifier(hidden_layer_sizes=(8,))
##########################################

##########################################
# Pipelines
pipe = Pipeline(steps=[
    ('scaling',scaler),
    ('rs',over_rs),
    ('clf',rf_clf)
])
#print(pipe.get_params().keys())
##########################################

##########################################
# Cross Validation (Resampling)
params = {
    'rs':[over_rs,under_rs, sm],
    'clf':[rf_clf,mlp_clf,svm_clf]
}
spliter = StratifiedKFold(n_splits=5)
grid = GridSearchCV(estimator=pipe,param_grid=params,scoring=scorers,cv=spliter,refit=False,n_jobs=-2)
grid.fit(df,target)
res = pd.DataFrame(grid.cv_results_)
res.to_csv('./Final_Results/SMOTE_Results.csv',index=False)
##########################################

