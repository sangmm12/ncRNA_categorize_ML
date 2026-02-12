import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
from joblib import dump

# Limit GPU usage to a specific device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 3

# Display options
pd.set_option('display.max_columns', 15)

# Data loading and initial setup
#X_start = 4 # Adjust this based on your feature column start index
#df = pd.read_csv('rnacentral_active_top20_2w_5_NV.csv')  # Sequence, Label, Species, Length, Source, V1,...,V1368
#df = pd.read_csv('ENA_NV.csv')#GeneName,Label,Length,Sequence,V1,...,V1368
#df_ = pd.read_csv('results_optimized/mRNA_2_nodup_NV.csv') #GeneName,Label,Length,Sequence,V1,...,V1368

#df_ = pd.read_csv('mRNA_nodup_NV.csv')
#df_ = pd.read_csv('mRNA_nodup_3class_NV.csv')
#df_ = pd.read_csv('all_mrna_nodup_labeled_NV.csv') #4 class

X_start = 5
#df_ = pd.read_csv('rnacentral/rnacentral_active_top20_2w_seed42_NV1368.csv')
#df_ = pd.read_csv('rnacentral/rnacentral_active_top20_2w_seed43_NV1368.csv')
#df_ = pd.read_csv('rnacentral/rnacentral_active_top20_2w_seed44_NV1368.csv')
df_ = pd.read_csv('rnacentral/rnacentral_active_top20_2w_seed45_NV1368.csv')
#df_ = pd.read_csv('rnacentral/rnacentral_active_top20_2w_seed46_NV1368.csv')
#df_ = pd.read_csv('rnacentral/rnacentral_active_top20_2w_seed47_NV1368.csv')
#df_ = pd.read_csv('rnacentral/rnacentral_active_top20_2w_seed48_NV1368.csv')
#df_ = pd.read_csv('rnacentral/rnacentral_active_top20_2w_seed49_NV1368.csv')
#df_ = pd.read_csv('rnacentral/rnacentral_active_top20_2w_seed50_NV1368.csv')
#df_ = pd.read_csv('rnacentral/rnacentral_active_top20_2w_seed51_NV1368.csv')


#df_ = pd.read_csv('rnacentral/rnacentral_active_top20_2w_seed42_BPE1500.csv')
#df_ = pd.read_csv('rnacentral/rnacentral_active_top20_2w_seed42_BPE1368.csv')
#df_ = pd.read_csv('rnacentral/rnacentral_active_top20_2w_seed42_BPE1496.csv')
#df_ = pd.read_csv('rnacentral/rnacentral_active_top20_2w_seed42_6mer5464.csv')

print(df_.head())
print(df_.shape)
print(df_.columns)
print(df_['Label'].value_counts())

# Step 2: Determine the minimum count among all unique labels
min_count = df_['Label'].value_counts().min()
print("Minimum count among labels:", min_count)

# Step 3: For each label group, randomly sample min_count records
df = df_.groupby('Label', group_keys=False).apply(lambda x: x.sample(n=min_count, random_state=42))
print(df.head())
print(df.shape)
print(df['Label'].value_counts())

# Filter data based on label set
#label_set = {"miRNA", "siRNA", "lncRNA", "rRNA", "tRNA", "snRNA", "snoRNA", "Y_RNA", "SRP_RNA", "pre_miRNA"}
#df = df[df['Label'].isin(label_set)]
#print("Filtered data:")
#print(df.head())
#print(df.shape)

# Separate features and target
X_balanced = df.iloc[:, X_start:]
#X_balanced = df.iloc[:, -1368:]
y_balanced = df['Label']
print("Feature shape:", X_balanced.shape)

# Encode the target labels
label_encoder = LabelEncoder()
y_balanced_encoded = label_encoder.fit_transform(y_balanced)

# Train-test split (80% train, 10% validation, 10% test)
X_train, X_temp, y_train, y_temp = train_test_split(X_balanced, y_balanced_encoded, test_size=0.2, random_state=42, stratify=y_balanced_encoded)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Print label mapping
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label mapping:", label_mapping)

# Save the label encoder for later use
#dump(label_encoder, f'best_models/mRNA_label_encoder_4class.pkl')
dump(label_encoder, f'best_models/rnacentral_label_encoder.pkl')
print("Label encoder saved.")

# Print dataset shapes
print("Train set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_val.shape, y_val.shape)
print("Test set shape:", X_test.shape, y_test.shape)

# Model training and evaluation function
def train_and_save_model(model, X_train, y_train, model_name, file_path):
    print(f"\n========== Training {model_name} ==========")
    model.fit(X_train, y_train)
    dump(model, file_path)
    print(f"{model_name} training completed and model saved to {file_path}.")
    return model

def evaluate_model_with_labels(model, X_val, y_val, X_test, y_test, label_encoder):
    print("\nEvaluating model...")
    try:
        # Predictions
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        y_val_pred_labels = label_encoder.inverse_transform(y_val_pred)
        y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)

        # Classification report
        print("Validation Set Evaluation:")
        print(classification_report(label_encoder.inverse_transform(y_val), y_val_pred_labels))
        print("Test Set Evaluation:")
        print(classification_report(label_encoder.inverse_transform(y_test), y_test_pred_labels))

        # AUC-ROC calculation
        if hasattr(model, "predict_proba"):
            if len(np.unique(y_val)) > 2:  # Multi-class
                auc_roc_val = roc_auc_score(y_val, model.predict_proba(X_val), multi_class='ovr')
                auc_roc_test = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
            else:  # Binary
                auc_roc_val = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
                auc_roc_test = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

            print(f"Validation AUC-ROC: {auc_roc_val:.4f}")
            print(f"Test AUC-ROC: {auc_roc_test:.4f}")
        else:
            print("AUC-ROC calculation skipped: Model does not support predict_proba.")
    except Exception as e:
        print(f"Error during model evaluation: {e}")

#暂时不用Random/GB，如果用###去掉
# Random Forest Classifier
#rf_model = RandomForestClassifier(n_estimators=1000, max_depth=50, min_samples_split=10, min_samples_leaf=2, random_state=42)
#Best Random Forest parameters: {'max_depth': 50, 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 2000}

###rf_model = RandomForestClassifier(n_estimators=2000, max_depth=50, min_samples_split=10, min_samples_leaf=2, random_state=42)
###rf_model = train_and_save_model(rf_model, X_train, y_train, "Random Forest", 'best_models/rnacentral_best_rf.pkl')
#rf_model = train_and_save_model(rf_model, X_train, y_train, "Random Forest", 'best_models/mRNA_best_rf_4class.pkl')

# Gradient Boosting Classifier
#gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
#Best Gradient Boosting parameters: {'learning_rate': 0.15, 'max_depth': 10, 'n_estimators': 200}

###gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.15, max_depth=10, random_state=42)
###gb_model = train_and_save_model(gb_model, X_train, y_train, "Gradient Boosting", 'best_models/rnacentral_best_gb.pkl')
#gb_model = train_and_save_model(gb_model, X_train, y_train, "Gradient Boosting", 'best_models/mRNA_best_gb_4class.pkl')

# XGBoost Classifier
# XGBoost Classifier（单卡省显存参数）
xgb_model = xgb.XGBClassifier(
    n_estimators=250,
    max_depth=20,                  # ↓ 从 20 降到 10
    learning_rate=0.01,
    eval_metric='mlogloss',
    tree_method='hist',            # 2.x 推荐；配合 device='cuda' 走 GPU
    device='cuda',
    objective='multi:softprob',
    random_state=42,
    # ↓ 关键的显存优化项
    max_bin=256,
    subsample=0.8,
    colsample_bytree=0.8,
    single_precision_histogram=True
)
'''
xgb_model = xgb.XGBClassifier(
    n_estimators=250, 
    max_depth=20, 
    learning_rate=0.01,
    eval_metric='mlogloss', 
    tree_method='hist',
    device='cuda',
    objective='multi:softprob', 
    random_state=42
)

#Best XGBoost parameters: {'learning_rate': 0.05, 'max_depth': 20, 'n_estimators': 250}
xgb_model = xgb.XGBClassifier(
    n_estimators=250, 
    max_depth=20, 
    learning_rate=0.05,
    eval_metric='logloss',  # Use 'logloss' for binary classification
    tree_method='gpu_hist', 
    objective='binary:logistic',  # Correct objective for binary tasks
    random_state=42
)
'''
xgb_model = train_and_save_model(xgb_model, X_train, y_train, "XGBoost", 'best_models/rnacentral_best_xgb.pkl')
#xgb_model = train_and_save_model(xgb_model, X_train, y_train, "XGBoost", 'best_models/mRNA_best_xgb_4class.pkl')

# Evaluate Models
###print("\nEvaluating Random Forest...")
###evaluate_model_with_labels(rf_model, X_val, y_val, X_test, y_test, label_encoder)

###print("\nEvaluating Gradient Boosting...")
###evaluate_model_with_labels(gb_model, X_val, y_val, X_test, y_test, label_encoder)

print("\nEvaluating XGBoost Model...")
evaluate_model_with_labels(xgb_model, X_val, y_val, X_test, y_test, label_encoder)


