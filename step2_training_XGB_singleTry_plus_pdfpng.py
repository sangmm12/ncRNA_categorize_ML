import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, auc
import xgboost as xgb
from joblib import dump, load
import argparse
import sys
from datetime import datetime
from sklearn.metrics import classification_report, roc_auc_score, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

# 设置matplotlib后端为PDF兼容
plt.rcParams['pdf.fonttype'] = 42  # 确保PDF中的文字可编辑
plt.rcParams['ps.fonttype'] = 42   # 确保PostScript中的文字可编辑
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

def setup_argparse():
    parser = argparse.ArgumentParser(description='RNA Classification Model Training')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--out', required=True, help='Output file for print statements')
    parser.add_argument('--x_start', type=int, default=5, help='Starting column index for features')
    parser.add_argument('--model_dir', default='best_models', help='Directory to save models')
    return parser.parse_args()

# 可视化函数
def plot_training_summary(best_model, X_val, y_val, X_test, y_test, label_encoder, 
                         best_auc, best_params, save_dir='best_models'):
    """绘制训练结果总结图表"""
    
    # 1. 混淆矩阵热图
    def plot_confusion_matrices():
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # 验证集混淆矩阵
        y_val_pred = best_model.predict(X_val)
        cm_val = confusion_matrix(y_val, y_val_pred)
        
        # 测试集混淆矩阵
        y_test_pred = best_model.predict(X_test)
        cm_test = confusion_matrix(y_test, y_test_pred)
        
        # 验证集混淆矩阵
        sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_,
                   ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_title('Validation Set - Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        axes[0].set_ylabel('True Label', fontsize=12)
        
        # 测试集混淆矩阵
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Oranges', 
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_,
                   ax=axes[1], cbar_kws={'label': 'Count'})
        axes[1].set_title('Test Set - Confusion Matrix', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        axes[1].set_ylabel('True Label', fontsize=12)
        
        plt.suptitle('Confusion Matrices', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # 保存双版本
        save_path = os.path.join(save_dir, 'confusion_matrices')
        plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}.pdf', bbox_inches='tight', transparent=False)
        plt.close()
        print(f"Confusion matrices saved to {save_path}.png and {save_path}.pdf")
    
    # 2. 特征重要性图
    def plot_feature_importance():
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = best_model.feature_importances_
            indices = np.argsort(feature_importance)[-20:]  # 取最重要的20个特征
            
            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(indices)), feature_importance[indices], align='center', 
                           color='steelblue', alpha=0.8, edgecolor='black')
            
            # 添加特征标签
            if X_val.shape[1] <= 50:  # 如果特征不多，显示特征名
                # 这里简化处理，实际使用时需要特征名
                feature_names = [f'Feature_{i}' for i in range(X_val.shape[1])]
                plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            else:
                plt.yticks(range(len(indices)), [f'Feature {i}' for i in indices])
            
            plt.xlabel('Feature Importance', fontsize=12)
            plt.title('Top 20 Feature Importance', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='x')
            
            # 添加数值标签
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.0005, bar.get_y() + bar.get_height()/2,
                        f'{width:.4f}', ha='left', va='center', fontsize=9)
            
            plt.tight_layout()
            
            # 保存双版本
            save_path = os.path.join(save_dir, 'feature_importance')
            plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{save_path}.pdf', bbox_inches='tight', transparent=False)
            plt.close()
            print(f"Feature importance plot saved to {save_path}.png and {save_path}.pdf")
    
    # 3. ROC曲线（多分类或二分类）
    def plot_roc_curves():
        if hasattr(best_model, 'predict_proba'):
            n_classes = len(label_encoder.classes_)
            
            if n_classes == 2:  # 二分类
                plt.figure(figsize=(10, 8))
                
                # 验证集ROC曲线
                y_val_proba = best_model.predict_proba(X_val)[:, 1]
                fpr_val, tpr_val, _ = roc_curve(y_val, y_val_proba)
                roc_auc_val = auc(fpr_val, tpr_val)
                
                # 测试集ROC曲线
                y_test_proba = best_model.predict_proba(X_test)[:, 1]
                fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
                roc_auc_test = auc(fpr_test, tpr_test)
                
                plt.plot(fpr_val, tpr_val, color='blue', lw=2, 
                        label=f'Validation (AUC = {roc_auc_val:.4f})')
                plt.plot(fpr_test, tpr_test, color='red', lw=2, 
                        label=f'Test (AUC = {roc_auc_test:.4f})')
                plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
                
                plt.xlim([-0.02, 1.02])
                plt.ylim([-0.02, 1.02])
                plt.xlabel('False Positive Rate', fontsize=12)
                plt.ylabel('True Positive Rate', fontsize=12)
                plt.title('ROC Curves (Binary Classification)', fontsize=14, fontweight='bold')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                
            else:  # 多分类，绘制每个类别的ROC曲线
                fig, axes = plt.subplots(2, 2, figsize=(14, 12))
                axes = axes.flatten()
                
                y_val_proba = best_model.predict_proba(X_val)
                y_test_proba = best_model.predict_proba(X_test)
                
                for i, class_name in enumerate(label_encoder.classes_[:4]):  # 最多显示4个类别
                    if i < len(axes):
                        # 验证集
                        fpr_val, tpr_val, _ = roc_curve((y_val == i).astype(int), y_val_proba[:, i])
                        roc_auc_val = auc(fpr_val, tpr_val)
                        
                        # 测试集
                        fpr_test, tpr_test, _ = roc_curve((y_test == i).astype(int), y_test_proba[:, i])
                        roc_auc_test = auc(fpr_test, tpr_test)
                        
                        axes[i].plot(fpr_val, tpr_val, color='blue', lw=2, 
                                   label=f'Val AUC = {roc_auc_val:.4f}')
                        axes[i].plot(fpr_test, tpr_test, color='red', lw=2, 
                                   label=f'Test AUC = {roc_auc_test:.4f}')
                        axes[i].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
                        
                        axes[i].set_xlim([-0.02, 1.02])
                        axes[i].set_ylim([-0.02, 1.02])
                        axes[i].set_xlabel('False Positive Rate', fontsize=10)
                        axes[i].set_ylabel('True Positive Rate', fontsize=10)
                        axes[i].set_title(f'Class: {class_name}', fontsize=12, fontweight='bold')
                        axes[i].legend(loc="lower right", fontsize=9)
                        axes[i].grid(True, alpha=0.3)
                
                if len(label_encoder.classes_) > 4:
                    # 在第4个子图中添加说明
                    axes[3].text(0.5, 0.5, f'Showing first 4 of {len(label_encoder.classes_)} classes\n'
                                         f'Overall best AUC: {best_auc:.4f}\n'
                                         f'Best params: {best_params}',
                               ha='center', va='center', fontsize=11,
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                    axes[3].axis('off')
                
                plt.suptitle('ROC Curves by Class (Multiclass)', fontsize=16, fontweight='bold', y=0.98)
            
            plt.tight_layout()
            
            # 保存双版本
            save_path = os.path.join(save_dir, 'roc_curves')
            plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{save_path}.pdf', bbox_inches='tight', transparent=False)
            plt.close()
            print(f"ROC curves saved to {save_path}.png and {save_path}.pdf")
    
    # 4. 性能指标对比图
    def plot_performance_comparison():
        # 计算各个性能指标
        y_val_pred = best_model.predict(X_val)
        y_test_pred = best_model.predict(X_test)
        
        # MCC
        mcc_val = matthews_corrcoef(y_val, y_val_pred)
        mcc_test = matthews_corrcoef(y_test, y_test_pred)
        
        # 准确率
        accuracy_val = np.mean(y_val == y_val_pred)
        accuracy_test = np.mean(y_test == y_test_pred)
        
        # AUC（如果可用）
        auc_val = best_auc
        if hasattr(best_model, 'predict_proba'):
            if len(np.unique(y_test)) > 2:
                auc_test = roc_auc_score(y_test, best_model.predict_proba(X_test), multi_class='ovr')
            else:
                auc_test = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
        else:
            auc_val = auc_test = 0
        
        # 创建对比图
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        
        metrics_data = {
            'Validation': [accuracy_val, auc_val, mcc_val],
            'Test': [accuracy_test, auc_test, mcc_test]
        }
        
        metric_names = ['Accuracy', 'AUC', 'MCC']
        colors = ['skyblue', 'lightcoral']
        
        for idx, metric in enumerate(metric_names):
            values = [metrics_data['Validation'][idx], metrics_data['Test'][idx]]
            bars = axes[idx].bar(['Validation', 'Test'], values, 
                               color=colors, edgecolor='black', alpha=0.8)
            
            axes[idx].set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
            axes[idx].set_ylabel(metric, fontsize=11)
            axes[idx].set_ylim([0, 1.1])
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # 在柱子上添加数值
            for bar, value in zip(bars, values):
                axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                             f'{value:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()
        
        # 保存双版本
        save_path = os.path.join(save_dir, 'performance_comparison')
        plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}.pdf', bbox_inches='tight', transparent=False)
        plt.close()
        print(f"Performance comparison plot saved to {save_path}.png and {save_path}.pdf")
        
        return {
            'validation': {'accuracy': accuracy_val, 'auc': auc_val, 'mcc': mcc_val},
            'test': {'accuracy': accuracy_test, 'auc': auc_test, 'mcc': mcc_test}
        }
    
    # 5. 预测概率分布图
    def plot_prediction_distribution():
        if hasattr(best_model, 'predict_proba'):
            n_classes = len(label_encoder.classes_)
            
            if n_classes == 2:  # 二分类
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # 验证集
                y_val_proba = best_model.predict_proba(X_val)[:, 1]
                axes[0].hist(y_val_proba[y_val == 0], bins=30, alpha=0.6, color='blue', 
                           label=f'{label_encoder.inverse_transform([0])[0]}', density=True)
                axes[0].hist(y_val_proba[y_val == 1], bins=30, alpha=0.6, color='red', 
                           label=f'{label_encoder.inverse_transform([1])[0]}', density=True)
                axes[0].set_xlabel('Predicted Probability', fontsize=11)
                axes[0].set_ylabel('Density', fontsize=11)
                axes[0].set_title('Validation Set - Probability Distribution', fontsize=13, fontweight='bold')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # 测试集
                y_test_proba = best_model.predict_proba(X_test)[:, 1]
                axes[1].hist(y_test_proba[y_test == 0], bins=30, alpha=0.6, color='blue', 
                           label=f'{label_encoder.inverse_transform([0])[0]}', density=True)
                axes[1].hist(y_test_proba[y_test == 1], bins=30, alpha=0.6, color='red', 
                           label=f'{label_encoder.inverse_transform([1])[0]}', density=True)
                axes[1].set_xlabel('Predicted Probability', fontsize=11)
                axes[1].set_ylabel('Density', fontsize=11)
                axes[1].set_title('Test Set - Probability Distribution', fontsize=13, fontweight='bold')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                plt.suptitle('Prediction Probability Distributions', fontsize=15, fontweight='bold', y=1.02)
            
            else:  # 多分类，显示每个类别的预测概率
                fig, axes = plt.subplots(2, min(2, n_classes), figsize=(14, 10))
                axes = axes.flatten()
                
                y_val_proba = best_model.predict_proba(X_val)
                
                for i, class_name in enumerate(label_encoder.classes_[:len(axes)]):
                    # 该类别的预测概率
                    class_probs = y_val_proba[:, i]
                    
                    # 按真实标签分组
                    true_indices = (y_val == i)
                    axes[i].hist(class_probs[true_indices], bins=30, alpha=0.6, color='green',
                               label=f'True {class_name}', density=True)
                    axes[i].hist(class_probs[~true_indices], bins=30, alpha=0.6, color='gray',
                               label='Other classes', density=True)
                    
                    axes[i].set_xlabel(f'P({class_name})', fontsize=10)
                    axes[i].set_ylabel('Density', fontsize=10)
                    axes[i].set_title(f'Class: {class_name}', fontsize=12, fontweight='bold')
                    axes[i].legend(fontsize=9)
                    axes[i].grid(True, alpha=0.3)
                
                plt.suptitle('Prediction Probability by Class (Validation Set)', 
                           fontsize=16, fontweight='bold', y=0.98)
            
            plt.tight_layout()
            
            # 保存双版本
            save_path = os.path.join(save_dir, 'prediction_distribution')
            plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{save_path}.pdf', bbox_inches='tight', transparent=False)
            plt.close()
            print(f"Prediction distribution plot saved to {save_path}.png and {save_path}.pdf")
    
    # 6. 参数影响分析图（针对XGBoost的不同参数组合）
    def plot_parameter_analysis(param_results):
        if param_results:
            # 提取参数和对应的AUC值
            param_names = [f"Comb_{i+1}" for i in range(len(param_results))]
            auc_values = [result['auc'] for result in param_results]
            
            # 创建参数分析图
            plt.figure(figsize=(12, 6))
            bars = plt.bar(param_names, auc_values, color='lightgreen', 
                          edgecolor='darkgreen', alpha=0.7)
            
            # 标记最佳参数组合
            best_idx = np.argmax(auc_values)
            bars[best_idx].set_color('gold')
            bars[best_idx].set_edgecolor('darkorange')
            
            plt.xlabel('Parameter Combination', fontsize=12)
            plt.ylabel('Validation AUC', fontsize=12)
            plt.title('Parameter Tuning Results for XGBoost', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            
            # 在柱子上添加AUC值
            for bar, auc_val in zip(bars, auc_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{auc_val:.4f}', ha='center', va='bottom', fontsize=9, rotation=90)
            
            plt.tight_layout()
            
            # 保存双版本
            save_path = os.path.join(save_dir, 'parameter_analysis')
            plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{save_path}.pdf', bbox_inches='tight', transparent=False)
            plt.close()
            print(f"Parameter analysis plot saved to {save_path}.png and {save_path}.pdf")
    
    # 执行所有绘图函数
    print("\n" + "="*60)
    print("Generating Visualization Plots...")
    print("="*60)
    
    # 用于存储参数分析结果
    param_results = []
    
    try:
        # 绘制各种图表
        plot_confusion_matrices()
        
        # 特征重要性图（只适用于树模型）
        if hasattr(best_model, 'feature_importances_'):
            plot_feature_importance()
        
        plot_roc_curves()
        
        performance_metrics = plot_performance_comparison()
        
        plot_prediction_distribution()
        
        print("\nAll visualization plots generated successfully!")
        print(f"PNG and PDF versions saved to: {save_dir}")
        
        return performance_metrics
        
    except Exception as e:
        print(f"Error generating plots: {e}")
        return None

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
        
        # 创建专门的目录用于保存图表
        plots_dir = os.path.join(args.model_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
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
        X_train, X_temp, y_train, y_temp = train_test_split(X_balanced, y_balanced_encoded, 
                                                           test_size=0.2, random_state=42, 
                                                           stratify=y_balanced_encoded)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, 
                                                       random_state=42, stratify=y_temp)

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
        
        # 存储参数组合结果用于可视化
        param_results = []

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
                    
                    # 存储参数组合结果
                    param_results.append({
                        'params': params,
                        'auc': auc_roc_val,
                        'mcc': mcc_val
                    })

                    if auc_roc_val > best_auc:
                        best_auc = auc_roc_val
                        best_model = xgb_model
                        best_params = params
                        best_model_name = model_path
                except Exception as e:
                    print(f"Error calculating AUC-ROC: {e}")
            else:
                print("Cannot calculate AUC-ROC for model comparison")

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
            
            # ===== 新增：生成可视化图表 =====
            print(f"\n{'='*60}")
            print("GENERATING VISUALIZATION PLOTS")
            print(f"{'='*60}")
            
            # 调用可视化函数
            performance_metrics = plot_training_summary(
                best_model, X_val, y_val, X_test, y_test, label_encoder,
                best_auc, best_params, save_dir=plots_dir
            )
            
            # 创建参数分析图
            if param_results:
                # 提取参数和对应的AUC值用于绘图
                param_auc_data = []
                for i, result in enumerate(param_results):
                    param_auc_data.append({
                        'name': f"Comb_{i+1}",
                        'params': result['params']['name_suffix'],
                        'auc': result['auc'],
                        'mcc': result['mcc']
                    })
                
                # 绘制参数分析图
                plt.figure(figsize=(14, 7))
                
                # AUC对比
                plt.subplot(1, 2, 1)
                auc_values = [d['auc'] for d in param_auc_data]
                bar_colors = ['lightgreen' if i != best_idx else 'gold' 
                             for i in range(len(auc_values))]
                best_idx = np.argmax(auc_values)
                
                bars1 = plt.bar(range(len(auc_values)), auc_values, 
                               color=bar_colors, edgecolor='darkgreen', alpha=0.7)
                plt.xlabel('Parameter Combination', fontsize=11)
                plt.ylabel('Validation AUC', fontsize=11)
                plt.title('AUC by Parameter Combination', fontsize=13, fontweight='bold')
                plt.xticks(range(len(auc_values)), [f"C{i+1}" for i in range(len(auc_values))])
                plt.grid(True, alpha=0.3, axis='y')
                
                for bar, auc_val in zip(bars1, auc_values):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                            f'{auc_val:.4f}', ha='center', va='bottom', fontsize=9, rotation=90)
                
                # MCC对比
                plt.subplot(1, 2, 2)
                mcc_values = [d['mcc'] for d in param_auc_data]
                bars2 = plt.bar(range(len(mcc_values)), mcc_values, 
                               color=bar_colors, edgecolor='darkblue', alpha=0.7)
                plt.xlabel('Parameter Combination', fontsize=11)
                plt.ylabel('Validation MCC', fontsize=11)
                plt.title('MCC by Parameter Combination', fontsize=13, fontweight='bold')
                plt.xticks(range(len(mcc_values)), [f"C{i+1}" for i in range(len(mcc_values))])
                plt.grid(True, alpha=0.3, axis='y')
                
                for bar, mcc_val in zip(bars2, mcc_values):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                            f'{mcc_val:.4f}', ha='center', va='bottom', fontsize=9, rotation=90)
                
                plt.suptitle('Parameter Tuning Results Analysis', fontsize=15, fontweight='bold', y=1.02)
                plt.tight_layout()
                
                # 保存参数分析图
                save_path = os.path.join(plots_dir, 'parameter_tuning_analysis')
                plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
                plt.savefig(f'{save_path}.pdf', bbox_inches='tight', transparent=False)
                plt.close()
                print(f"Parameter tuning analysis plot saved to {save_path}.png and {save_path}.pdf")
            
            # 创建最终报告
            report_path = os.path.join(args.model_dir, 'training_summary.txt')
            with open(report_path, 'w') as report_file:
                report_file.write("="*60 + "\n")
                report_file.write("RNA CLASSIFICATION TRAINING SUMMARY\n")
                report_file.write("="*60 + "\n\n")
                report_file.write(f"Training completed at: {datetime.now()}\n")
                report_file.write(f"Input file: {args.input}\n")
                report_file.write(f"Number of parameter combinations tested: {len(xgb_param_combinations)}\n")
                report_file.write(f"Best parameters: {best_params}\n")
                report_file.write(f"Best validation AUC: {best_auc:.4f}\n")
                
                if performance_metrics:
                    report_file.write("\nPerformance Metrics:\n")
                    report_file.write(f"  Validation Accuracy: {performance_metrics['validation']['accuracy']:.4f}\n")
                    report_file.write(f"  Validation MCC: {performance_metrics['validation']['mcc']:.4f}\n")
                    report_file.write(f"  Test Accuracy: {performance_metrics['test']['accuracy']:.4f}\n")
                    report_file.write(f"  Test MCC: {performance_metrics['test']['mcc']:.4f}\n")
                
                report_file.write("\nGenerated Plots (PNG and PDF):\n")
                report_file.write("  1. confusion_matrices\n")
                report_file.write("  2. feature_importance (if applicable)\n")
                report_file.write("  3. roc_curves\n")
                report_file.write("  4. performance_comparison\n")
                report_file.write("  5. prediction_distribution\n")
                report_file.write("  6. parameter_tuning_analysis\n")
                report_file.write("\nAll plots are saved in: " + plots_dir + "\n")
            
            print(f"\nTraining summary saved to: {report_path}")
            # ===== 新增结束 =====
            
        else:
            print("No suitable model found!")

        print(f"\n=== Training Completed at {datetime.now()} ===")
        
        # Restore stdout
        sys.stdout = original_stdout
    
    print(f"Training completed! Results saved to {args.out}")
    print(f"Best model saved in {args.model_dir}")
    print(f"Visualization plots saved in {os.path.join(args.model_dir, 'plots')}")

if __name__ == "__main__":
    main()
