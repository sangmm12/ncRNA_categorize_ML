#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support, matthews_corrcoef, accuracy_score

import xgboost as xgb
from joblib import dump

# 仅用 GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
pd.set_option('display.max_columns', 15)

# ===== 自定义 AUC 计算（兼容二/多分类） =====
def _ovr_auc_from_proba(y_true, proba):
    if proba is None:
        raise ValueError("This estimator cannot provide probabilities or decision scores.")
    # (n,) => 二分类阳性概率；(n,1) => 退化情形；(n,2) => 二分类；(n,K) => 多分类
    if proba.ndim == 1:
        return roc_auc_score(y_true, proba)
    if proba.shape[1] == 1:
        return roc_auc_score(y_true, proba.ravel())
    n_classes = len(np.unique(y_true))
    if n_classes <= 2:
        pos = proba[:, 1]
        return roc_auc_score(y_true, pos)
    else:
        return roc_auc_score(y_true, proba, multi_class='ovr')

# ===== 关键修复：自定义 scorer（不使用 make_scorer） =====
def auc_ovr_scorer(estimator, X, y):
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
    elif hasattr(estimator, "decision_function"):
        dec = estimator.decision_function(X)
        # 将 decision_function 映射为概率
        if dec.ndim == 1:
            from scipy.special import expit
            proba = expit(dec)
        else:
            from scipy.special import softmax
            proba = softmax(dec, axis=1)
    else:
        proba = None
    return _ovr_auc_from_proba(y, proba)

# ===== 数据加载 =====
data_set = "mammalian"#"multispecies"  # or "mammalian"
#df = pd.read_csv(f'mammalian_multispecies/{data_set}_train_dev_pre_NV.csv')#square
#df = pd.read_csv(f'mammalian_multispecies/{data_set}_train_dev_pre_NV1368.csv')#abs
#df = pd.read_csv('/home/hgq/BioData/hgq/ncRNA/old/mammalian_train_dev_pre_NV1368_abs.csv')
#df = pd.read_csv('mammalian_multispecies/multispecies_train_dev_pre_NV1368_abs.csv')
df = pd.read_csv('mammalian_multispecies/multispecies_train_dev_pre_NV1368_absfalse.csv')
print(df.head())
print(df.shape)
print(df.columns)

# 标签编码
le = LabelEncoder()
df["Encoded_Label"] = le.fit_transform(df["Label"])
print("Label Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# 特征选择
dim = 'V1368'
X_start, X_end = 3, 3 + 1368
X = df.iloc[:, X_start:X_end].to_numpy()
y = df["Encoded_Label"].to_numpy()

train_mask = (df["Source"] == "train").to_numpy()
val_mask   = (df["Source"] == "pre").to_numpy()
test_mask  = (df["Source"] == "dev").to_numpy()

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"Train set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")

# ===== 打印最佳参数和所有候选参数 =====
def print_best_and_candidates(gs, model_name):
    print(f"\nBest {model_name} parameters found: {gs.best_params_}")
    print("All candidate parameters and scores:")
    for i, params in enumerate(gs.cv_results_['params']):
        mean_score = gs.cv_results_['mean_test_score'][i]
        print(f"ROC AUC Score: {mean_score:.4f}, Parameters: {params}")

# ===== Random Forest =====
print("\n========== Starting hyperparameter tuning for Random Forest Classifier ==========")
rf = RandomForestClassifier(random_state=42)
rf_grid = {
    "n_estimators": [500, 1000, 1200, 1400, 1600, 1800, 2000],
    "max_depth": [20, 50, 100],
    "min_samples_split": [10, 20],
    "min_samples_leaf": [2, 4, 6, 8],
}
rf_gs = GridSearchCV(rf, rf_grid, cv=3, scoring=auc_ovr_scorer, n_jobs=-1, verbose=2)
rf_gs.fit(X_train, y_train)
best_rf = rf_gs.best_estimator_
print("Random Forest hyperparameter tuning completed.")
print_best_and_candidates(rf_gs, "Random Forest")
os.makedirs("best_models", exist_ok=True)
dump(best_rf, f"best_models/{data_set}_best_rf_{dim}.pkl")

# ===== Gradient Boosting =====
print("\n========== Starting hyperparameter tuning for Gradient Boosting Classifier ==========")
gb = GradientBoostingClassifier(random_state=42)
gb_grid = {
    "n_estimators": [50, 100, 150, 200],
    "learning_rate": [0.1, 0.15],
    "max_depth": [5, 10],
}
gb_gs = GridSearchCV(gb, gb_grid, cv=3, scoring=auc_ovr_scorer, n_jobs=-1, verbose=2)
gb_gs.fit(X_train, y_train)
best_gb = gb_gs.best_estimator_
print("Gradient Boosting hyperparameter tuning completed.")
print_best_and_candidates(gb_gs, "Gradient Boosting")
dump(best_gb, f"best_models/{data_set}_best_gb_{dim}.pkl")

# ===== XGBoost =====
print("\n========== Starting hyperparameter tuning for XGBoost ==========")
n_classes = len(np.unique(y_train))

# GPU 可用性探测并自动回退
tree_method = "hist"
device = "cpu"  # 默认使用 CPU

try:
    # 尝试使用 GPU
    _ = xgb.Booster(params={"tree_method": "hist", "device": "cuda"})
    device = "cuda"
    print("GPU 可用，使用 GPU 训练")
except Exception:
    device = "cpu"
    print("GPU 不可用，回退到 CPU 训练")

xgb_kwargs = dict(
    eval_metric="mlogloss",
    random_state=42,
    tree_method=tree_method,
)
if n_classes > 2:
    xgb_kwargs.update(objective="multi:softprob", num_class=n_classes)
else:
    xgb_kwargs.update(objective="binary:logistic")

xgb_clf = xgb.XGBClassifier(**xgb_kwargs)

xgb_grid = {
    "n_estimators": [250, 350, 500, 550],
    "max_depth": [5, 20, 40, 60],
    "learning_rate": [0.05, 0.1, 0.15],
}

try:
    xgb_gs = GridSearchCV(xgb_clf, xgb_grid, cv=3, scoring=auc_ovr_scorer, n_jobs=-1, verbose=2)
    xgb_gs.fit(X_train, y_train)
    best_xgb = xgb_gs.best_estimator_
    print("XGBoost hyperparameter tuning completed.")
    print_best_and_candidates(xgb_gs, "XGBoost")
except Exception as e:
    print("Error during XGBoost tuning:", e)
    print("Fitting default XGBoost model...")
    xgb_clf.fit(X_train, y_train)
    best_xgb = xgb_clf

dump(best_xgb, f"best_models/{data_set}_best_xgb_{dim}.pkl")

# ===== 统一评估（四位小数） =====
def evaluate_model(name, model, Xv, yv, Xt, yt):
    print(f"\n{'='*60}")
    print(f"Evaluating {name}...")
    print(f"{'='*60}")

    # 验证集评估
    print("Validation Set Evaluation:")
    yv_pred = model.predict(Xv)
    
    # 计算accuracy, precision, recall, f1-score, MCC
    accuracy = accuracy_score(yv, yv_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(yv, yv_pred, average='weighted')
    mcc = matthews_corrcoef(yv, yv_pred)
    
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, MCC: {mcc:.4f}")
    
    # 详细分类报告
    print("\nDetailed Classification Report:")
    print(classification_report(yv, yv_pred, digits=4))

    # AUC-ROC
    if hasattr(model, "predict_proba") or hasattr(model, "decision_function"):
        if hasattr(model, "predict_proba"):
            v_proba = model.predict_proba(Xv)
        else:
            dec = model.decision_function(Xv)
            if dec.ndim == 1:
                from scipy.special import expit
                v_proba = expit(dec)
            else:
                from scipy.special import softmax
                v_proba = softmax(dec, axis=1)
        auc_v = _ovr_auc_from_proba(yv, v_proba)
        print(f"Validation AUC-ROC: {auc_v:.4f}")
    else:
        print("Validation AUC-ROC: N/A (no proba/decision_function)")

    # 测试集评估
    print(f"\n{'='*60}")
    print("Test Set Evaluation:")
    yt_pred = model.predict(Xt)
    
    # 计算accuracy, precision, recall, f1-score, MCC
    accuracy = accuracy_score(yt, yt_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(yt, yt_pred, average='weighted')
    mcc = matthews_corrcoef(yt, yt_pred)
    
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, MCC: {mcc:.4f}")
    
    # 详细分类报告
    print("\nDetailed Classification Report:")
    print(classification_report(yt, yt_pred, digits=4))

    if hasattr(model, "predict_proba") or hasattr(model, "decision_function"):
        if hasattr(model, "predict_proba"):
            t_proba = model.predict_proba(Xt)
        else:
            dec = model.decision_function(Xt)
            if dec.ndim == 1:
                from scipy.special import expit
                t_proba = expit(dec)
            else:
                from scipy.special import softmax
                t_proba = softmax(dec, axis=1)
        auc_t = _ovr_auc_from_proba(yt, t_proba)
        print(f"Test AUC-ROC: {auc_t:.4f}")
    else:
        print("Test AUC-ROC: N/A (no proba/decision_function)")

# 评估三个模型
evaluate_model("Random Forest", best_rf, X_val, y_val, X_test, y_test)
evaluate_model("Gradient Boosting", best_gb, X_val, y_val, X_test, y_test)
evaluate_model("XGBoost", best_xgb, X_val, y_val, X_test, y_test)

print("\n" + "="*80)
print("SUMMARY: All models have been evaluated with accuracy, precision, recall, f1-score, MCC and AUC-ROC")
print("="*80)

# ===== 最终结果汇总表格 =====
def create_summary_table(models_dict, X_val, y_val, X_test, y_test):
    """创建所有模型的性能汇总表格"""
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY TABLE")
    print(f"{'='*80}")
    
    results = []
    
    for name, model in models_dict.items():
        # 验证集性能
        y_val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(y_val, y_val_pred, average='weighted')
        val_mcc = matthews_corrcoef(y_val, y_val_pred)
        
        # 测试集性能
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='weighted')
        test_mcc = matthews_corrcoef(y_test, y_test_pred)
        
        # AUC计算
        if hasattr(model, "predict_proba"):
            val_proba = model.predict_proba(X_val)
            test_proba = model.predict_proba(X_test)
            val_auc = _ovr_auc_from_proba(y_val, val_proba)
            test_auc = _ovr_auc_from_proba(y_test, test_proba)
        else:
            val_auc = test_auc = "N/A"
        
        results.append({
            'Model': name,
            'Val_Accuracy': f"{val_accuracy:.4f}",
            'Val_Precision': f"{val_precision:.4f}",
            'Val_Recall': f"{val_recall:.4f}", 
            'Val_F1': f"{val_f1:.4f}",
            'Val_MCC': f"{val_mcc:.4f}",
            'Val_AUC': f"{val_auc:.4f}" if val_auc != "N/A" else "N/A",
            'Test_Accuracy': f"{test_accuracy:.4f}",
            'Test_Precision': f"{test_precision:.4f}",
            'Test_Recall': f"{test_recall:.4f}",
            'Test_F1': f"{test_f1:.4f}",
            'Test_MCC': f"{test_mcc:.4f}",
            'Test_AUC': f"{test_auc:.4f}" if test_auc != "N/A" else "N/A"
        })
    
    # 创建并显示表格
    summary_df = pd.DataFrame(results)
    print("\nPerformance Summary:")
    print(summary_df.to_string(index=False))
    
    return summary_df

# 创建汇总表格
models_dict = {
    "Random Forest": best_rf,
    "Gradient Boosting": best_gb, 
    "XGBoost": best_xgb
}
summary_df = create_summary_table(models_dict, X_val, y_val, X_test, y_test)

# ===== 保存结果到CSV文件 =====
os.makedirs("results", exist_ok=True)
summary_df.to_csv(f"results/{data_set}_performance_summary.csv", index=False)
print(f"\nResults saved to: results/{data_set}_performance_summary.csv")
