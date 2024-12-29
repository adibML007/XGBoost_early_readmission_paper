"""
This script performs data preprocessing, model training, and evaluation for predicting early readmission using XGBoost.
It includes steps for handling imbalanced data with SMOTE, evaluating model performance, and visualizing results.
Additionally, it generates SHAP plots for feature importance and plots for ROC and Precision-Recall curves.
"""

import pandas as pd
import json
from pre_processing_steps import preprocess_dataframe
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, f1_score, roc_curve, auc, average_precision_score, classification_report, confusion_matrix, precision_recall_curve
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import shap

# Load the JSON data
with open('Max_scores.json') as f:
    data = json.load(f)

# Load and preprocess the data
df = pd.read_csv("dat.csv")
df = df[(df['death.within.3.months'] == 0) | (df['death.within.6.months'] == 0)]
y = df['re.admission.within.28.days']
df = pd.concat([df[data["Selected_Features"]], y], axis=1)
df = preprocess_dataframe(df, 'best_tuners')
df.fillna(df.median(), inplace=True)
X = df.drop('re.admission.within.28.days', axis=1)

# Initialize and train the model
model = XGBClassifier(max_depth=2, n_estimators=30, random_state=4, scale_pos_weight=1)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X.iloc[data["Train_Index"]], y.iloc[data["Train_Index"]])
model.fit(X_train, y_train)

# Evaluate on training data
y_train_pred_fold = model.predict(X_train)
train_recall = round(recall_score(y_train, y_train_pred_fold), 2)
train_f1 = round(f1_score(y_train, y_train_pred_fold), 2)
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_fold)
train_roc_auc = round(auc(fpr_train, tpr_train), 2)

print(f"Train Recall: {train_recall}")
print(f"Train F1 Score: {train_f1}")
print(f"Train ROC AUC: {train_roc_auc}")

# Evaluate on test data
y_test = y.iloc[data["Test_Index"]]
y_test_pred_fold = model.predict(X.iloc[data["Test_Index"]])
y_test_pred_prob = model.predict_proba(X.iloc[data["Test_Index"]])[:, 1]
test_recall = round(recall_score(y_test, y_test_pred_fold), 2)
test_f1 = round(f1_score(y_test, y_test_pred_fold), 2)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred_prob)
test_roc_auc = round(auc(fpr_test, tpr_test), 2)

print(f"Test Recall: {test_recall}")
print(f"Test F1 Score: {test_f1}")
print(f"Test ROC AUC: {test_roc_auc}")
print(classification_report(y_test, y_test_pred_fold))

# Confusion matrix
cm_selected = confusion_matrix(y_test, y_test_pred_fold, labels=model.classes_)
sns.heatmap(cm_selected, annot=True, fmt='g', xticklabels=model.classes_, yticklabels=model.classes_)
plt.ylabel('Prediction', fontsize=13)
plt.xlabel('Actual', fontsize=13)
plt.title('XGBoost Confusion Matrix (Selected Features)', fontsize=14)
plt.savefig('xgb-cm.png')

# SHAP plot
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X.iloc[data["Test_Index"]])
plt.figure(figsize=(8, 8))
shap.summary_plot(shap_values, X.iloc[data["Test_Index"]])
plt.savefig('shap-summary.png')

# ROC curve
plt.figure(figsize=(4, 4))
plt.plot(fpr_test, tpr_test, color='blue', label=f'ROC curve (AUC = {test_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='green', linestyle='--', label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc-auc.png')

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_test_pred_fold)
pr_auc = average_precision_score(y_test, y_test_pred_prob)
print(f"PR AUC: {pr_auc}")
plt.figure(figsize=(4, 4))
plt.plot(recall, precision, color='blue', label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.savefig('pr-auc.png')
