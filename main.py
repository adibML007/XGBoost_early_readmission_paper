"""
Author: Adib Zaman
Date: December 2024
Description: This script performs K-Fold Cross Validation using XGBoost for predicting early readmission within 28 days. 
It includes data preprocessing, feature selection using Recursive Feature Elimination (RFE), and handling class imbalance with SMOTE.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings

# Import custom functions
from pre_processing_steps import preprocess_dataframe

# Ignore future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load and preprocess the dataframe
df = pd.read_csv("dat.csv")
df = preprocess_dataframe(df, 'best_tuner')
df.fillna(df.median(), inplace=True)

# Plot feature correlation with respect to early readmission
plt.figure(figsize=(10, 50))
df.drop(columns="re.admission.within.28.days").corrwith(df['re.admission.within.28.days']).plot(kind='line', color='green')
plt.title("Feature correlation with respect to early readmission")
plt.ylim(-0.15, 0.15)
plt.xlabel("Correlation score with early readmission")
plt.ylabel("Features")
plt.xticks(range(len(df.columns[:-1])), df.columns[:-1], rotation=90)
plt.savefig("feature_correlation.png")

# Separate features (X) and target variable (y)
X = df.drop('re.admission.within.28.days', axis=1)
y = df['re.admission.within.28.days']

# Standardize features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Initialize K-Fold Cross Validation
kf = KFold(n_splits=10, random_state=4, shuffle=True)

# Initialize results DataFrame
results_df = pd.DataFrame(columns=['Fold', 'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1', 'Train AUC', 'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1', 'Test AUC'])

# Perform K-Fold Cross Validation
for fold, (train_index, test_index) in enumerate(kf.split(X, y), start=1):
    # Split data into training and testing sets for the current fold
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

    # Initialize and fit the model with Recursive Feature Elimination (RFE)
    model = XGBClassifier(max_depth=2, n_estimators=30, random_state=4)
    model_rfe = RFE(estimator=model, n_features_to_select=17)
    model_rfe.fit(X_train_fold, y_train_fold)
    selected_features = X_train_fold.columns[model_rfe.support_]

    # Select features based on RFE
    X_train_fold = X_train_fold[selected_features]
    X_test_fold = X_test_fold[selected_features]

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_fold, y_train_fold = smote.fit_resample(X_train_fold, y_train_fold)

    # Fit the model on the training data
    model.fit(X_train_fold, y_train_fold)

    # Evaluate the model on the training set
    y_train_pred_fold = model.predict(X_train_fold)
    train_accuracy = round(accuracy_score(y_train_fold, y_train_pred_fold), 2)
    train_precision = round(precision_score(y_train_fold, y_train_pred_fold), 2)
    train_recall = round(recall_score(y_train_fold, y_train_pred_fold), 2)
    train_f1 = round(f1_score(y_train_fold, y_train_pred_fold), 2)
    fpr_train, tpr_train, _ = roc_curve(y_train_fold, y_train_pred_fold)
    train_roc_auc = round(auc(fpr_train, tpr_train), 2)

    # Evaluate the model on the test set
    y_test_pred_fold = model.predict(X_test_fold)
    test_accuracy = round(accuracy_score(y_test_fold, y_test_pred_fold), 2)
    test_precision = round(precision_score(y_test_fold, y_test_pred_fold), 2)
    test_recall = round(recall_score(y_test_fold, y_test_pred_fold), 2)
    test_f1 = round(f1_score(y_test_fold, y_test_pred_fold), 2)
    fpr_test, tpr_test, _ = roc_curve(y_test_fold, y_test_pred_fold)
    test_roc_auc = round(auc(fpr_test, tpr_test), 2)

    # Check if the current fold's test recall is higher than the previous maximum
    if fold == 1 or test_recall > results_df['Test Recall'].max():
        fold_max = fold
        train_index_max = train_index
        test_index_max = test_index
        selected_features_max = selected_features
        model_max = model

    # Append scores to the results DataFrame
    results_df = pd.concat([results_df, pd.DataFrame({
        'Fold': [fold],
        'Train Accuracy': [train_accuracy],
        'Train Precision': [train_precision],
        'Train Recall': [train_recall],
        'Train F1': [train_f1],
        'Train AUC': [train_roc_auc],
        'Test Accuracy': [test_accuracy],
        'Test Precision': [test_precision],
        'Test Recall': [test_recall],
        'Test F1': [test_f1],
        'Test AUC': [test_roc_auc]
    })], ignore_index=True)

# Save the maximum scores to a DataFrame
max_scores_df = pd.DataFrame({
    'Fold': [fold_max],
    'Train Accuracy': [results_df.loc[fold_max - 1, 'Train Accuracy']],
    'Train Precision': [results_df.loc[fold_max - 1, 'Train Precision']],
    'Train Recall': [results_df.loc[fold_max - 1, 'Train Recall']],
    'Train F1': [results_df.loc[fold_max - 1, 'Train F1']],
    'Train AUC': [results_df.loc[fold_max - 1, 'Train AUC']],
    'Test Accuracy': [results_df.loc[fold_max - 1, 'Test Accuracy']],
    'Test Precision': [results_df.loc[fold_max - 1, 'Test Precision']],
    'Test Recall': [results_df.loc[fold_max - 1, 'Test Recall']],
    'Test F1': [results_df.loc[fold_max - 1, 'Test F1']],
    'Test AUC': [results_df.loc[fold_max - 1, 'Test AUC']],
    'Selected_Features': [list(selected_features_max)],
    'Train_Index': [list(train_index_max)],
    'Test_Index': [list(test_index_max)]
})

# Save results to CSV and JSON files
results_df.to_csv("Prelim_results.csv", index=False)
max_scores_df.to_json("Max_scores.json", orient='records', lines=True)

# Print the results DataFrame
print(results_df)
