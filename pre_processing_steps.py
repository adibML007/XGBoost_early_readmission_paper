import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Function to preprocess the dataframe based on the model run type
def preprocess_dataframe(df, model_run: str):
    # Calculate the percentage of null values for each column
    null_percentages = {col: df[col].isnull().sum() / df.shape[0] * 100 for col in df.columns}

    # Create a dataframe to display the percentage of null values for each column
    null_values_percentage = pd.DataFrame.from_dict(null_percentages, orient='index', columns=['Null Percentage'])

    # Plot the null percentages as a bar plot
    null_percentages_series = pd.Series(null_percentages).sort_values()
    plt.figure(figsize=(12, 20))
    null_percentages_series.plot(kind='barh', fontsize=8)
    plt.title('Percentage of Null Values per Column')
    plt.xlabel('Columns')
    plt.ylabel('Percentage of Null Values')
    plt.savefig('null_values_percentage.png')

    # Drop columns with more than 50% null values
    df = df.drop(columns=null_percentages_series[null_percentages_series >= 50].index)

    # Remove dead patients and irrelevant columns if model_run is 'best_tuner'
    if model_run == 'best_tuner':
        df = df[(df['death.within.3.months'] == 0) | (df['death.within.6.months'] == 0)]
        df = df.drop(['Unnamed: 0', 'inpatient.number', 'DestinationDischarge', 'admission.ward', 'admission.way', 
                      'occupation', 'discharge.department', 'visit.times', 'death.within.3.months', 
                      're.admission.within.3.months', 'death.within.6.months', 
                      're.admission.within.6.months', 'return.to.emergency.department.within.6.months'], axis=1)

    # Encode categorical features using LabelEncoder
    encoder = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = encoder.fit_transform(df[col])

    return df
