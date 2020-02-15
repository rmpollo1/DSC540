import numpy as np
import pandas as pd


####################
# Globals
#
action_dict = {
    'Loan originated': 1,
    'Application denied by financial institution':0,
    'Loan purchased by the institution': 1,
    'Application approved but not accepted': 1,
    'Preapproval request approved but not accepted': 1,
    'Preapproval request denied by financial institution':0
}
#
#
#####################

# Read in Raw Data
data = pd.read_csv("./Washington_State_HDMA-2016.csv",low_memory=False,dtype={'census_tract_number':'object'})

# Column Remover for missing values
def missing_value_threshold(df,threshold=0.9):
    '''
    Removes columns from dataframe with 
    missing values above threshold.

    Inputs 
        df: Pandas Dataframe

        threshold: Float
    Outputs
        Pandas Dataframe
    '''
    print("--- Removing Missing Colunms ---")
    removed = []
    for c in df.columns:
        if df[c].isna().mean() > threshold:
            removed.append(c)

    print("Columns Removed: {}".format(removed))
    return df.drop(removed,axis=1)

# Remove Columns with Many Missing Values (>90% Missing)
data = missing_value_threshold(data)
# Fill Missing Values for some columns
data = data.fillna({'edit_status_name':'blank','msamd_name':'Rural - WA'})
# Drop Non-informative Columns
data = data.drop(
    [
        'agency_name','as_of_year','state_name',
        'state_abbr','sequence_number','respondent_id',
        'application_date_indicator','msamd_name',
        'county_name','census_tract_number'
    ]
    ,1
) 

# Drop Samples based on action taken
data = data[data.action_taken_name != 'File closed for incompleteness'] # Can be caught by other automation
data = data[data.action_taken_name != 'Application withdrawn by applicant'] # No Application Decision Reached

# Drop Observations with missing Values
data = data.dropna()

# Turn Action Taken into Binary Target Value (approve/reject)
data['approve'] = data.action_taken_name.apply(lambda x: action_dict[x])
data = data.drop('action_taken_name',1)

# Write Clean Data to File
data.to_csv("./HDMA_Loan_Data_Clean.csv")
