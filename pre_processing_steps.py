import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Pre-processing Introduction

# Pre-processing is a crucial step in data analysis and machine learning.
# It involves transforming raw data into a format that is more suitable for analysis.
# This step can include tasks such as data cleaning, normalization, transformation, 
# feature extraction, and selection.

# Below are some common pre-processing techniques:


# Load dataset
def preprocess_dataframe(df, model_run:str):
    null_pacentages = {}
    for col in df.columns:
        null_percentage = df[col].isnull().sum() / df.shape[0] * 100
        null_pacentages[col] = null_percentage

    # Create a dataframe to display the percentage of null values for each column.
    null_values_percentage = pd.DataFrame.from_dict(null_pacentages, orient='index', columns=['Null Percentage'])
    null_values_percentage

    # Create a series from the null_pacentages dictionary.
    null_pacentages_series = pd.Series(null_pacentages)

    # Drop columns with more than 50% null values from the dataframe.
    df = df.drop(columns=null_pacentages_series[null_pacentages_series >= 50].index)

    # Identify more columns in the dataframe that contain null values.
    df.columns[df.isnull().any()]

    if(model_run == 'best_tuner'):
    
    # Remove dead patients from the dataframe
        df = df[(df['death.within.3.months'] == 0) | (df['death.within.6.months'] == 0)]

        # Remove irrelevant columns from the dataframe
        df = df.drop(['Unnamed: 0', 'inpatient.number', 'DestinationDischarge', 'admission.ward', 'admission.way', 'occupation', 'discharge.department', 'visit.times', 'death.within.3.months', 're.admission.within.3.months', 'death.within.6.months', 're.admission.within.6.months', 'return.to.emergency.department.within.6.months'], axis=1)

    for col in df.columns:
        if df[col].dtype == 'object':
            crosstab = pd.crosstab(df[col], df['re.admission.within.28.days'], normalize='index') * 100

    # Create a dictionary to store categorical columns and their corresponding values.
    categorical_cols = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            categorical_cols[col] = df[col]

    # Encode categorical features using LabelEncoder.
    from sklearn.preprocessing import LabelEncoder

    # Create a LabelEncoder object
    encoder = LabelEncoder()

    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

    return df

