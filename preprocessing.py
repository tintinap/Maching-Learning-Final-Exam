import pandas as pd
import numpy as np

def preprocess_input_mental(df):
    df = df.copy()

    # Drop Employee ID column
    df = df.drop('Employee ID', axis=1)

    # Drop rows with missing target values
    missing_value_target = df.loc[df['Burn Rate'].isna(), :].index
    df = df.drop(missing_value_target, axis=0).reset_index(drop=True)

    # Fill remaining missing values with column means
    for column in ['Resource Allocation', 'Mental Fatigue Score']:
        df[column] = df[column].fillna(df[column].mean())

    # Extract date features
     ## no year becuz there's only a year(2008) in this dataset
    df['Date of Joining'] = pd.to_datetime(df['Date of Joining'])
    df['Join Month'] = df['Date of Joining'].apply(lambda x: x.month)
    df['Join Day'] = df['Date of Joining'].apply(lambda x: x.day)
    df = df.drop('Date of Joining', axis=1)

    # Binary encoding
    df['Gender'] = df['Gender'].replace({'Female': 0, 'Male': 1})
    df['Company Type'] = df['Company Type'].replace({'Product': 0, 'Service': 1})
    df['WFH Setup Available'] = df['WFH Setup Available'].replace({'No': 0, 'Yes': 1})

    # Split df into X and y
    X = df.drop('Burn Rate', axis=1)

    # Scale X
    for column in X.columns:
        df[column] = df[column].apply(lambda x: (x -df[column].min()) / (df[column].max()-df[column].min()))
    
    # reorder columns
    reorder_col = X.columns.tolist()
    reorder_col.append('Burn Rate')
    df = df[reorder_col]

    del X

    return df

def check_available():
    print('good')