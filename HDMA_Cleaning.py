import pandas as pd 

data = pd.read_csv("./Washington_State_HDMA-2016.csv",low_memory=False)

# Column Remover for missing values
def missing_value_threshold(df,threshold=0.9):
    '''
    Removes 
    '''
    print("--- Removing Missing Colunms ---")
    removed = []
    for c in df.columns:
        if df[c].isna().mean() > threshold:
            removed.append(c)

    print("Columns Removed: {}".format(removed))
    return df.drop(removed,axis=1)

data = missing_value_threshold(data) # Remove Columns with Many Missing Values (>90% Missing)

data = data.fillna({'edit_status_name':'blank','msamd_name':'Rural - WA'}) # Fill Missing Values for some columns

data = data.drop(['agency_name','as_of_year','state_name','state_abbr','sequence_number','respondent_id'],1) # Drop Non-informative Columns

data = data.dropna()
for c in data.columns:
    print(c,len(data[c].unique()))
