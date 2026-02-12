import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
from joblib import dump
import argparse
import sys
from datetime import datetime
from sklearn.metrics import classification_report, roc_auc_score, matthews_corrcoef

def setup_argparse():
    parser = argparse.ArgumentParser(description='RNA Classification Model Training')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--out', required=True, help='Output file for print statements')
    parser.add_argument('--x_start', type=int, default=5, help='Starting column index for features')
    parser.add_argument('--model_dir', default='best_models', help='Directory to save models')
    return parser.parse_args()

def main():
    args = setup_argparse()
    
    # Redirect all print output to file
    original_stdout = sys.stdout
    with open(args.out, 'w', encoding='utf-8') as f:
        sys.stdout = f
        
        print(f"=== RNA Classification Training Started at {datetime.now()} ===")
        print(f"Input file: {args.input}")
        print(f"Output log: {args.out}")
        print(f"Model directory: {args.model_dir}")
        print(f"Feature start column: {args.x_start}")
        
        # Create model directory if it doesn't exist
        os.makedirs(args.model_dir, exist_ok=True)
        
        # Limit GPU usage to a specific device
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0

        # Display options
        pd.set_option('display.max_columns', 15)
        pd.set_option('display.float_format', '{:.4f}'.format)

        # Data loading
        try:
            df_ = pd.read_csv(args.input)
            print(f"Data loaded successfully from {args.input}")
        except Exception as e:
            print(f"Error loading data: {e}")
            return

        print("\n=== Data Overview ===")
        print(df_.head())
        print(f"Data shape: {df_.shape}")
        print(f"Columns: {list(df_.columns)}")
        print("\nLabel distribution:")
        print(df_['Label'].value_counts())

        # Step 2: Determine the minimum count among all unique labels
        min_count = df_['Label'].value_counts().min()
        print(f"\nMinimum count among labels: {min_count}")

        # Step 3: For each label group, randomly sample min_count records
        # Fix the FutureWarning by explicitly selecting non-grouping columns
        df = df_.groupby('Label', group_keys=False).apply(
            lambda x: x.sample(n=min_count, random_state=42)
        ).reset_index(drop=True)
        print("\n=== Balanced Data ===")
        print(df.head())
        print(f"Balanced data shape: {df.shape}")
        print("\nBalanced label distribution:")
        print(df['Label'].value_counts())

        # === 新增的过滤代码放在这里 ===
        print("\n=== Filtering out 'ncRNA' and 'other' labels ===")
        print("Before filtering label distribution:")
        print(df['Label'].value_counts())

        # 剔除'ncRNA'和'other'标签的数据
        df = df[~df['Label'].isin(['ncRNA', 'other'])]

        print("After filtering label distribution:")
        print(df['Label'].value_counts())
        print(f"Filtered data shape: {df.shape}")
        # === 新增代码结束 ===

        # Separate features and target
        X_balanced = df.iloc[:, args.x_start:]
        y_balanced = df['Label']
        print(f"\nFeature shape: {X_balanced.shape}")

        # Encode the target labels
        label_encoder = LabelEncoder()
        y_balanced_encoded = label_encoder.fit_transform(y_balanced)

        # Train-test split (80% train, 10% validation, 10% test)
        X_train, X_temp, y_train, y_temp = train_test_split(X_balanced, y_balanced_encoded, test_size=0.2, random_state=42, stratify=y_balanced_encoded)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

        # Print label mapping
        label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        print("\n=== Label Mapping ===")
        for label, code in label_mapping.items():
            print(f"  {label}: {code}")

        # Save the label encoder for later use
        encoder_path = os.path.join(args.model_dir, 'rnacentral_label_encoder.pkl')
        dump(label_encoder, encoder_path)
        print(f"\nLabel encoder saved to: {encoder_path}")

        # Print dataset shapes
        print("\n=== Dataset Shapes ===")
        print(f"Train set shape: {X_train.shape}, {y_train.shape}")
        print(f"Validation set shape: {X_val.shape}, {y_val.shape}")
        print(f"Test set shape: {X_test.shape}, {y_test.shape}")

        # Model training and evaluation function
        def train_and_save_model(model, X_train, y_train, model_name, file_path, params=None):
            print(f"\n========== Training {model_name} ==========")
            if params:
                print(f"Parameters: {params}")
            model.fit(X_train, y_train)
            dump(model, file_path)
            print(f"{model_name} training completed and model saved to {file_path}.")
            return model

        def evaluate_model_with_labels(model, X_val, y_val, X_test, y_test, label_encoder, dataset_name=""):
            print(f"\n=== Evaluating {dataset_name} Model ===")
            try:
                # 预测（数值编码）
                y_val_pred = model.predict(X_val)
                y_test_pred = model.predict(X_test)
        
                # 转回原始字符串标签，便于看报告
                y_val_true_labels = label_encoder.inverse_transform(y_val)
                y_test_true_labels = label_encoder.inverse_transform(y_test)
                y_val_pred_labels = label_encoder.inverse_transform(y_val_pred)
                y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)
        
                # 分类报告（含 precision / recall / f1-score）
                print("\nValidation Set Evaluation:")
                print(classification_report(y_val_true_labels, y_val_pred_labels, digits=4))
        
                print("\nTest Set Evaluation:")
                print(classification_report(y_test_true_labels, y_test_pred_labels, digits=4))
        
                # ===== 新增：MCC 指标 =====
                mcc_val = matthews_corrcoef(y_val, y_val_pred)
                mcc_test = matthews_corrcoef(y_test, y_test_pred)
                print(f"\nValidation MCC: {mcc_val:.4f}")
                print(f"Test MCC: {mcc_test:.4f}")
                # ===== 新增结束 =====
        
                # AUC-ROC 计算
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
        '''        
        def evaluate_model_with_labels(model, X_val, y_val, X_test, y_test, label_encoder, dataset_name=""):
            print(f"\n=== Evaluating {dataset_name} Model ===")
            try:
                # Predictions
                y_val_pred = model.predict(X_val)
                y_test_pred = model.predict(X_test)
                y_val_pred_labels = label_encoder.inverse_transform(y_val_pred)
                y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)

                # Classification report with 4 decimal places
                print("\nValidation Set Evaluation:")
                print(classification_report(label_encoder.inverse_transform(y_val), y_val_pred_labels, digits=4))
                print("\nTest Set Evaluation:")
                print(classification_report(label_encoder.inverse_transform(y_test), y_test_pred_labels, digits=4))

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
        '''
        # XGBoost parameter combinations to try
        xgb_param_combinations = [
            {
                'n_estimators': 250,
                'max_depth': 10,
                'learning_rate': 0.01,
                'name_suffix': '250_10_0.01'
            },
            {
                'n_estimators': 250,
                'max_depth': 10,
                'learning_rate': 0.05,
                'name_suffix': '250_10_0.05'
            },
            {
                'n_estimators': 250,
                'max_depth': 20,
                'learning_rate': 0.01,
                'name_suffix': '250_20_0.01'
            },
            {
                'n_estimators': 250,
                'max_depth': 20,
                'learning_rate': 0.05,
                'name_suffix': '250_20_0.05'
            },
            {
                'n_estimators': 500,
                'max_depth': 10,
                'learning_rate': 0.01,
                'name_suffix': '500_10_0.01'
            },
            {
                'n_estimators': 500,
                'max_depth': 10,
                'learning_rate': 0.05,
                'name_suffix': '500_10_0.05'
            },
            {
                'n_estimators': 500,
                'max_depth': 20,
                'learning_rate': 0.01,
                'name_suffix': '500_20_0.01'
            },
            {
                'n_estimators': 500,
                'max_depth': 20,
                'learning_rate': 0.05,
                'name_suffix': '500_20_0.05'
            }
        ]

        best_auc = 0
        best_model = None
        best_params = None
        best_model_name = ""

        for i, params in enumerate(xgb_param_combinations):
            print(f"\n{'='*60}")
            print(f"XGBoost Parameter Combination {i+1}/{len(xgb_param_combinations)}")
            print(f"{'='*60}")

            xgb_model = xgb.XGBClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                eval_metric='mlogloss',
                tree_method='hist',
                device='cuda',
                objective='multi:softprob',
                random_state=42,
                max_bin=256,
                subsample=0.8,
                colsample_bytree=0.8
            )

            model_path = os.path.join(args.model_dir, f'rnacentral_xgb_{params["name_suffix"]}.pkl')
            xgb_model = train_and_save_model(
                xgb_model, X_train, y_train,
                f"XGBoost ({params['name_suffix']})",
                model_path, params
            )

            # ===== 新增：每个组合的精度 / 召回 / F1 / MCC =====
            # 使用验证集
            y_val_pred = xgb_model.predict(X_val)

            # 打印带原始标签名的分类报告
            y_val_true_labels = label_encoder.inverse_transform(y_val)
            y_val_pred_labels = label_encoder.inverse_transform(y_val_pred)

            print("\nValidation classification report for this combination:")
            print(classification_report(y_val_true_labels, y_val_pred_labels, digits=4))

            # MCC（使用数值编码）
            mcc_val = matthews_corrcoef(y_val, y_val_pred)
            print(f"Validation MCC for this combination: {mcc_val:.4f}")
            # ===== 新增结束 =====

            # Evaluate and track best model（仍然用 AUC-ROC 选最优）
            if hasattr(xgb_model, "predict_proba"):
                try:
                    if len(np.unique(y_val)) > 2:
                        auc_roc_val = roc_auc_score(y_val, xgb_model.predict_proba(X_val), multi_class='ovr')
                    else:
                        auc_roc_val = roc_auc_score(y_val, xgb_model.predict_proba(X_val)[:, 1])

                    print(f"Validation AUC-ROC for this combination: {auc_roc_val:.4f}")

                    if auc_roc_val > best_auc:
                        best_auc = auc_roc_val
                        best_model = xgb_model
                        best_params = params
                        best_model_name = model_path
                except Exception as e:
                    print(f"Error calculating AUC-ROC: {e}")
            else:
                print("Cannot calculate AUC-ROC for model comparison")
        '''        
        # Train and evaluate XGBoost with different parameter combinations
        for i, params in enumerate(xgb_param_combinations):
            print(f"\n{'='*60}")
            print(f"XGBoost Parameter Combination {i+1}/{len(xgb_param_combinations)}")
            print(f"{'='*60}")
            
            # Remove single_precision_histogram parameter to avoid warning
            xgb_model = xgb.XGBClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                eval_metric='mlogloss',
                tree_method='hist',
                device='cuda',
                objective='multi:softprob',
                random_state=42,
                max_bin=256,
                subsample=0.8,
                colsample_bytree=0.8
                # single_precision_histogram removed to avoid warning
            )
            
            model_path = os.path.join(args.model_dir, f'rnacentral_xgb_{params["name_suffix"]}.pkl')
            xgb_model = train_and_save_model(xgb_model, X_train, y_train, 
                                           f"XGBoost ({params['name_suffix']})", 
                                           model_path, params)
            
            # Evaluate and track best model
            if hasattr(xgb_model, "predict_proba"):
                try:
                    if len(np.unique(y_val)) > 2:
                        auc_roc_val = roc_auc_score(y_val, xgb_model.predict_proba(X_val), multi_class='ovr')
                    else:
                        auc_roc_val = roc_auc_score(y_val, xgb_model.predict_proba(X_val)[:, 1])
                    
                    print(f"Validation AUC-ROC for this combination: {auc_roc_val:.4f}")
                    
                    if auc_roc_val > best_auc:
                        best_auc = auc_roc_val
                        best_model = xgb_model
                        best_params = params
                        best_model_name = model_path
                except Exception as e:
                    print(f"Error calculating AUC-ROC: {e}")
            else:
                print("Cannot calculate AUC-ROC for model comparison")
        '''
        # Final evaluation of best model
        if best_model is not None:
            print(f"\n{'='*60}")
            print("BEST MODEL SUMMARY")
            print(f"{'='*60}")
            print(f"Best parameters: {best_params}")
            print(f"Best validation AUC-ROC: {best_auc:.4f}")
            print(f"Best model saved as: {best_model_name}")
            
            # Save the best model with a special name
            best_model_final_path = os.path.join(args.model_dir, 'rnacentral_best_xgb.pkl')
            dump(best_model, best_model_final_path)
            print(f"Best model also saved as: {best_model_final_path}")
            
            # Final comprehensive evaluation
            evaluate_model_with_labels(best_model, X_val, y_val, X_test, y_test, label_encoder, "Best XGBoost")
        else:
            print("No suitable model found!")

        print(f"\n=== Training Completed at {datetime.now()} ===")
        
        # Restore stdout
        sys.stdout = original_stdout
    
    print(f"Training completed! Results saved to {args.out}")
    print(f"Best model saved in {args.model_dir}")

if __name__ == "__main__":
    main()
