import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from joblib import dump
import pickle
from sklearn.metrics import (
    classification_report,
    make_scorer,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

# Limit GPU usage to a specific device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only use GPU 2

# Display options
#pd.set_option('display.max_columns', 15)

data_set = "multispecies"#"mammalian" #multispecies
# Data loading and initial setup
df = pd.read_csv(f'mammalian_multispecies/{data_set}_train_dev_pre_NV.csv') # Sequence, Label, Source, V1,...,V1368
print(df.head())
print(df.shape)
print(df.columns)

# Encode labels to consecutive integers
label_encoder = LabelEncoder()
df['Encoded_Label'] = label_encoder.fit_transform(df['Label'])
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping:", label_mapping)

dim = 'V1368'
X_start = 3  # Adjust this based on your feature column start index
X_end   = 3 + 1368
# Separate features and target
#X = df.loc[:, 'V1':f'{dim}'].values
X = df.iloc[:, X_start:X_end]
y = df['Encoded_Label']
print("Feature shape:", X.shape)

# Split the data based on the 'Source' column
X_train = X[df['Source'] == 'train']
y_train = y[df['Source'] == 'train']

X_val = X[df['Source'] == 'pre']
y_val = y[df['Source'] == 'pre']

X_test = X[df['Source'] == 'dev']
y_test = y[df['Source'] == 'dev']

# Ensure alignment
assert all(X_train.index == y_train.index), "Mismatch between X_train and y_train indices!"
assert all(X_val.index == y_val.index), "Mismatch between X_val and y_val indices!"
assert all(X_test.index == y_test.index), "Mismatch between X_test and y_test indices!"

# Print dataset shapes
print("Train set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Test set shape:", X_test.shape)

# Custom scorer for multi-class ROC AUC
#roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr')
# 原来的
# roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr')

# 修改后（兼容新版本）
roc_auc_scorer = make_scorer(
    roc_auc_score,
    response_method="predict_proba",
    multi_class="ovr",
)
'''
# Random Forest Classifier Hyperparameter Tuning
print("\n========== Starting hyperparameter tuning for Random Forest Classifier ==========")
rf_model = RandomForestClassifier(random_state=42)
rf_param_grid = {
    'n_estimators': [500, 1000, 2000, 3000, 4000, 5000],
    'max_depth': [20, 50, 100],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [2, 4, 6, 8]
}
rf_grid_search = GridSearchCV(
    estimator=rf_model, param_grid=rf_param_grid, cv=5,
    scoring=roc_auc_scorer, n_jobs=-1, verbose=0
)
rf_grid_search.fit(X_train, y_train)
print("Random Forest hyperparameter tuning completed.")
best_rf_model = rf_grid_search.best_estimator_
print("Best Random Forest parameters:", rf_grid_search.best_params_)
os.makedirs('best_models', exist_ok=True)
dump(best_rf_model, f'best_models/{data_set}_best_rf_{dim}.pkl')

# Gradient Boosting Classifier Hyperparameter Tuning
print("\n========== Starting hyperparameter tuning for Gradient Boosting Classifier ==========")
gb_model = GradientBoostingClassifier(random_state=42)
gb_param_grid = {
    'n_estimators': [50, 200, 500, 1000, 2000, 3000],
    'learning_rate': [0.1, 0.15],
    'max_depth': [5, 10]
}
gb_grid_search = GridSearchCV(
    estimator=gb_model, param_grid=gb_param_grid, cv=5,
    scoring=roc_auc_scorer, n_jobs=-1, verbose=0
)
gb_grid_search.fit(X_train, y_train)
print("Gradient Boosting hyperparameter tuning completed.")
best_gb_model = gb_grid_search.best_estimator_
print("Best Gradient Boosting parameters:", gb_grid_search.best_params_)
dump(best_gb_model, f'best_models/{data_set}_best_gb_{dim}.pkl')
'''
# XGBoost Classifier Hyperparameter Tuning
print("\n========== Starting hyperparameter tuning for XGBoost ==========")
xgb_model = xgb.XGBClassifier(
    eval_metric='mlogloss',  # Use appropriate evaluation metric
    random_state=42,
    tree_method='gpu_hist',  # Use GPU for training if available
    objective='multi:softprob'  # Specify the objective for multi-class classification
)
xgb_param_grid = {
    'n_estimators': [250, 500, 1000, 3000, 5000, 10000],
    'max_depth': [5, 20, 40, 60],
    'learning_rate': [0.05, 0.1, 0.15]
}

try:
    grid_search = GridSearchCV(
        estimator=xgb_model, param_grid=xgb_param_grid, cv=5,
        scoring=roc_auc_scorer, n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    print("XGBoost hyperparameter tuning completed.")
    best_xgb_model = grid_search.best_estimator_
    print("Best XGBoost parameters:", grid_search.best_params_)
except Exception as e:
    print("Error during XGBoost tuning:", e)
    # Fallback: Fit the default XGBoost model
    print("Fitting default XGBoost model...")
    xgb_model.fit(X_train, y_train)
    best_xgb_model = xgb_model

# Save the best XGBoost model
dump(best_xgb_model, f'best_models/{data_set}_best_xgb_{dim}.pkl')

# Model Evaluation Function
def evaluate_model(model, model_name, X_val, y_val, X_test, y_test, summary_list):
    print(f"\nEvaluating model: {model_name} ...")

    # ---------- Validation ----------
    print("Validation Set Evaluation:")
    y_val_pred = model.predict(X_val)
    # 分类报告 4 位小数
    print(classification_report(y_val, y_val_pred, digits=4))

    # AUC
    if len(np.unique(y_val)) > 2:
        auc_roc_val = roc_auc_score(y_val, model.predict_proba(X_val), multi_class='ovr')
    else:
        auc_roc_val = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    # 其他指标（加权平均，适合类别不均衡）
    val_acc = accuracy_score(y_val, y_val_pred)
    val_prec = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
    val_rec = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
    val_mcc = matthews_corrcoef(y_val, y_val_pred)

    print(f"Validation AUC-ROC: {auc_roc_val:.4f}")

    # ---------- Test ----------
    print("\nTest Set Evaluation:")
    y_test_pred = model.predict(X_test)
    print(classification_report(y_test, y_test_pred, digits=4))

    if len(np.unique(y_test)) > 2:
        auc_roc_test = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
    else:
        auc_roc_test = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_rec = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_mcc = matthews_corrcoef(y_test, y_test_pred)

    print(f"Test AUC-ROC: {auc_roc_test:.4f}")

    # 把本模型的指标放进 summary_list，后面汇总成表
    summary_list.append({
        "Model": model_name,
        "Val_Accuracy":  val_acc,
        "Val_Precision": val_prec,
        "Val_Recall":    val_rec,
        "Val_F1":        val_f1,
        "Val_MCC":       val_mcc,
        "Val_AUC":       auc_roc_val,
        "Test_Accuracy":  test_acc,
        "Test_Precision": test_prec,
        "Test_Recall":    test_rec,
        "Test_F1":        test_f1,
        "Test_MCC":       test_mcc,
        "Test_AUC":       auc_roc_test,
    })

# 用来存所有模型的性能指标
performance_summary = []
'''
print("\nEvaluating Random Forest...")
evaluate_model(best_rf_model, "Random Forest", X_val, y_val, X_test, y_test, performance_summary)

print("\nEvaluating Gradient Boosting...")
evaluate_model(best_gb_model, "Gradient Boosting", X_val, y_val, X_test, y_test, performance_summary)
'''
print("\nEvaluating Best XGBoost Model...")
evaluate_model(best_xgb_model, "XGBoost", X_val, y_val, X_test, y_test, performance_summary)

# ---------- 最终总表 ----------
summary_df = pd.DataFrame(performance_summary)

# 按你给的列顺序排一下
summary_df = summary_df[
    [
        "Model",
        "Val_Accuracy", "Val_Precision", "Val_Recall", "Val_F1", "Val_MCC", "Val_AUC",
        "Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1", "Test_MCC", "Test_AUC",
    ]
]

print("\nPerformance Summary:")
print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"   # 所有浮点数 4 位小数
    )
)


