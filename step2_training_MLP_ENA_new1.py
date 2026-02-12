import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, matthews_corrcoef, roc_auc_score, roc_curve, auc, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置matplotlib后端为PDF兼容
plt.rcParams['pdf.fonttype'] = 42  # 确保PDF中的文字可编辑
plt.rcParams['ps.fonttype'] = 42   # 确保PostScript中的文字可编辑

# GPU Selection (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1 if available

# Display options
pd.set_option('display.max_columns', 15)

# Load dataset
df_ = pd.read_csv('ENA/ENA_deduped_NV1368.csv')
print("Original Dataset Shape:", df_.shape)
print(df_['Label'].value_counts())

min_count = df_['Label'].value_counts().min()
print("Minimum count among labels:", min_count)

# Step 3: For each label group, randomly sample min_count records
df = df_.groupby('Label', group_keys=False).apply(lambda x: x.sample(n=min_count, random_state=42))
print(df['Label'].value_counts())
df = shuffle(df, random_state=42)

# Separate features and target
X = df.iloc[:, -1368:]  # Features
y = df['Label']  # Target
print("Feature Shape:", X.shape, "Target Shape:", y.shape)

# Encode the target labels using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("Label Mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# Save the label encoder to disk
with open('best_models/ENA_label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("LabelEncoder saved.")

# Train-test split (90% train, 9% validation, 1% test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Print dataset shapes
print("Train Set Shape:", X_train.shape, "Train Target Shape:", y_train.shape)
print("Validation Set Shape:", X_val.shape, "Validation Target Shape:", y_val.shape)
print("Test Set Shape:", X_test.shape, "Test Target Shape:", y_test.shape)

# Define the Neural Network model for binary classification
def create_binary_mlp(input_dim, learning_rate):
    model = Sequential([
        Dense(512, input_dim=input_dim, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Sigmoid for binary classification
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',  # Binary classification loss
        metrics=['accuracy', 'AUC']  # 添加AUC作为监控指标
    )
    return model

# Model parameters
input_dim = X_train.shape[1]  # Number of features
learning_rate = 0.000001  # Initial learning rate

# Check if the model already exists
model_path = 'best_models/ENA_binary_classification_model.h5'

if os.path.exists(model_path):
    print(f"Loading existing model from {model_path}...")
    # Load the existing model and continue training
    model = load_model(model_path)
    learning_rate = 0.000000001  # Adjust learning rate for continued training
else:
    # Create and compile a new model
    model = create_binary_mlp(input_dim, learning_rate)

# Add EarlyStopping for better convergence
early_stopping = EarlyStopping(
    monitor='val_auc',  # 监控val_auc
    patience=50,
    restore_best_weights=True,
    mode='max'  # AUC越大越好
)

# Add ReduceLROnPlateau to dynamically adjust the learning rate
reduce_lr = ReduceLROnPlateau(
    monitor='val_auc',  # 监控val_auc
    factor=0.5,
    patience=10,
    min_lr=1e-10,
    verbose=2,
    mode='max'
)

# Add ModelCheckpoint to save the best model during training
model_checkpoint = ModelCheckpoint(
    filepath='best_models/ENA_binary_classification_model.h5',
    monitor='val_auc',  # 监控val_auc
    save_best_only=True,
    verbose=2,
    mode='max'
)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=2
)

# Load the best model
best_model = load_model(model_path)

# Evaluate the best model
val_loss, val_accuracy, val_auc_keras = best_model.evaluate(X_val, y_val, verbose=0)
test_loss, test_accuracy, test_auc_keras = best_model.evaluate(X_test, y_test, verbose=0)

print(f"Validation Accuracy (Best Model): {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation AUC: {val_auc_keras:.4f}")
print(f"Test Accuracy (Best Model): {test_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test AUC: {test_auc_keras:.4f}")

# Predict and evaluate using the best model
y_test_pred_prob = best_model.predict(X_test)
y_test_pred_labels = (y_test_pred_prob > 0.5).astype(int)  # Convert probabilities to binary labels

# Also get predictions for validation set for evaluation
y_val_pred_prob = best_model.predict(X_val)
y_val_pred_labels = (y_val_pred_prob > 0.5).astype(int)

# Calculate MCC for both validation and test sets
val_mcc = matthews_corrcoef(y_val, y_val_pred_labels)
test_mcc = matthews_corrcoef(y_test, y_test_pred_labels)

# Calculate ROC AUC using sklearn for verification
val_auc_sklearn = roc_auc_score(y_val, y_val_pred_prob)
test_auc_sklearn = roc_auc_score(y_test, y_test_pred_prob)

print(f"\nMCC Scores:")
print(f"Validation MCC: {val_mcc:.4f}")
print(f"Test MCC: {test_mcc:.4f}")

print(f"\nAUC Scores:")
print(f"Validation AUC (Keras): {val_auc_keras:.4f}, AUC (sklearn): {val_auc_sklearn:.4f}")
print(f"Test AUC (Keras): {test_auc_keras:.4f}, AUC (sklearn): {test_auc_sklearn:.4f}")

# Generate a classification report
print("\nTest Set Evaluation (Best Model):")
print(classification_report(y_test, y_test_pred_labels, 
                           target_names=label_encoder.classes_, 
                           digits=4))

# ==============================================
# 可视化图表部分 - 修改为同时保存PNG和PDF
# ==============================================

# 1. 训练历史曲线（类似文章中Figure 1）
def plot_training_history(history, save_path='best_models/training_history'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss曲线
    axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy曲线
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC曲线
    axes[1, 0].plot(history.history['auc'], label='Training AUC', linewidth=2)
    axes[1, 0].plot(history.history['val_auc'], label='Validation AUC', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('AUC', fontsize=12)
    axes[1, 0].set_title('Training and Validation AUC', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 学习率变化（如果有的话）
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], label='Learning Rate', linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # 显示最终指标总结
        axes[1, 1].text(0.1, 0.5, 
                       f'Final Metrics:\n\n'
                       f'Best Val AUC: {max(history.history["val_auc"]):.4f}\n'
                       f'Best Val Acc: {max(history.history["val_accuracy"]):.4f}\n'
                       f'Final Train Loss: {history.history["loss"][-1]:.4f}\n'
                       f'Final Val Loss: {history.history["val_loss"][-1]:.4f}',
                       fontsize=12, verticalalignment='center')
        axes[1, 1].axis('off')
    
    plt.suptitle('Model Training History', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 保存PNG版本
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    # 保存PDF版本
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight', transparent=False)
    plt.close()
    print(f"Training history plot saved to {save_path}.png and {save_path}.pdf")

# 2. ROC曲线（类似文章中Figure 2）
def plot_roc_curves(y_true_val, y_pred_val, y_true_test, y_pred_test, save_path='best_models/roc_curves'):
    plt.figure(figsize=(10, 8))
    
    # 计算ROC曲线
    fpr_val, tpr_val, _ = roc_curve(y_true_val, y_pred_val)
    roc_auc_val = auc(fpr_val, tpr_val)
    
    fpr_test, tpr_test, _ = roc_curve(y_true_test, y_pred_test)
    roc_auc_test = auc(fpr_test, tpr_test)
    
    # 绘制ROC曲线
    plt.plot(fpr_val, tpr_val, color='blue', lw=3, alpha=0.8, 
             label=f'Validation (AUC = {roc_auc_val:.4f})')
    plt.plot(fpr_test, tpr_test, color='red', lw=3, alpha=0.8, 
             label=f'Test (AUC = {roc_auc_test:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.8)
    
    # 添加一些关键点
    thresholds = [0.2, 0.5, 0.8]
    for threshold in thresholds:
        idx_val = np.argmin(np.abs(y_pred_val - threshold))
        idx_test = np.argmin(np.abs(y_pred_test - threshold))
        
        if idx_val < len(fpr_val) and idx_test < len(fpr_test):
            plt.scatter(fpr_val[idx_val], tpr_val[idx_val], s=100, 
                       color='blue', alpha=0.6, zorder=5)
            plt.scatter(fpr_test[idx_test], tpr_test[idx_test], s=100, 
                       color='red', alpha=0.6, zorder=5)
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curves', 
              fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 添加性能总结
    plt.text(0.6, 0.3, 
             f'Validation AUC: {roc_auc_val:.4f}\n'
             f'Test AUC: {roc_auc_test:.4f}\n'
             f'ΔAUC: {abs(roc_auc_test - roc_auc_val):.4f}',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=12)
    
    plt.tight_layout()
    
    # 保存PNG版本
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    # 保存PDF版本
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight', transparent=False)
    plt.close()
    print(f"ROC curves plot saved to {save_path}.png and {save_path}.pdf")

# 3. 混淆矩阵热图（类似文章中Figure 3）
def plot_confusion_matrix(y_true, y_pred, labels, dataset_name="Test", save_path='best_models/confusion_matrix'):
    cm_matrix = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title(f'Confusion Matrix - {dataset_name} Set', fontsize=16, fontweight='bold')
    
    # 计算并显示一些指标
    tn, fp, fn, tp = cm_matrix.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    plt.text(0.5, -0.15, 
             f'Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}',
             ha='center', transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    
    # 为不同数据集生成不同的文件名
    dataset_suffix = f'_{dataset_name.lower()}'
    
    # 保存PNG版本
    plt.savefig(f'{save_path}{dataset_suffix}.png', dpi=300, bbox_inches='tight')
    # 保存PDF版本
    plt.savefig(f'{save_path}{dataset_suffix}.pdf', bbox_inches='tight', transparent=False)
    plt.close()
    print(f"Confusion matrix for {dataset_name} set saved to {save_path}{dataset_suffix}.png and {save_path}{dataset_suffix}.pdf")

# 4. 性能指标对比图
def plot_performance_comparison(metrics_dict, save_path='best_models/performance_comparison'):
    datasets = ['Validation', 'Test']
    metrics = ['Accuracy', 'AUC', 'MCC']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(metrics):
        values = [metrics_dict[f'{dataset.lower()}_{metric.lower()}'] 
                  for dataset in datasets]
        
        colors = ['skyblue', 'lightcoral']
        bars = axes[idx].bar(datasets, values, color=colors, edgecolor='black', alpha=0.8)
        
        # 在柱子上显示数值
        for bar, value in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{value:.4f}', ha='center', va='bottom', fontsize=11)
        
        axes[idx].set_ylabel(metric, fontsize=12)
        axes[idx].set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        axes[idx].set_ylim([0, 1.1 if metric != 'MCC' else 1.1])
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存PNG版本
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    # 保存PDF版本
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight', transparent=False)
    plt.close()
    print(f"Performance comparison plot saved to {save_path}.png and {save_path}.pdf")

# 5. 预测概率分布图（类似文章中Figure 4）
def plot_prediction_distribution(y_true, y_pred_prob, dataset_name="Test", save_path='best_models/prediction_distribution'):
    # 将预测概率按真实标签分组
    class_0_probs = y_pred_prob[y_true == 0].flatten()
    class_1_probs = y_pred_prob[y_true == 1].flatten()
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 概率直方图
    axes[0, 0].hist(class_0_probs, bins=30, alpha=0.7, color='blue', 
                   label=f'{label_encoder.inverse_transform([0])[0]}', 
                   edgecolor='black')
    axes[0, 0].hist(class_1_probs, bins=30, alpha=0.7, color='red', 
                   label=f'{label_encoder.inverse_transform([1])[0]}', 
                   edgecolor='black')
    axes[0, 0].set_xlabel('Predicted Probability', fontsize=12)
    axes[0, 0].set_ylabel('Count', fontsize=12)
    axes[0, 0].set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 累积分布函数
    axes[0, 1].hist(class_0_probs, bins=30, alpha=0.7, color='blue', 
                   cumulative=True, density=True, histtype='step', linewidth=3,
                   label=f'{label_encoder.inverse_transform([0])[0]}')
    axes[0, 1].hist(class_1_probs, bins=30, alpha=0.7, color='red', 
                   cumulative=True, density=True, histtype='step', linewidth=3,
                   label=f'{label_encoder.inverse_transform([1])[0]}')
    axes[0, 1].set_xlabel('Predicted Probability', fontsize=12)
    axes[0, 1].set_ylabel('Cumulative Probability', fontsize=12)
    axes[0, 1].set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 箱线图
    bp_data = [class_0_probs, class_1_probs]
    bp_labels = [label_encoder.inverse_transform([0])[0], 
                 label_encoder.inverse_transform([1])[0]]
    axes[1, 0].boxplot(bp_data, labels=bp_labels, patch_artist=True,
                      boxprops=dict(facecolor='lightgray', color='black'),
                      medianprops=dict(color='red', linewidth=2))
    axes[1, 0].set_ylabel('Predicted Probability', fontsize=12)
    axes[1, 0].set_title('Probability Distribution by Class', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. 决策边界分析
    thresholds = np.arange(0.1, 1.0, 0.1)
    accuracies = []
    for thresh in thresholds:
        pred_labels = (y_pred_prob > thresh).astype(int)
        acc = np.mean(pred_labels.flatten() == y_true)
        accuracies.append(acc)
    
    axes[1, 1].plot(thresholds, accuracies, 'o-', linewidth=3, markersize=8)
    best_idx = np.argmax(accuracies)
    axes[1, 1].scatter(thresholds[best_idx], accuracies[best_idx], 
                      s=200, color='red', alpha=0.6, 
                      label=f'Best: {thresholds[best_idx]:.2f} ({accuracies[best_idx]:.4f})')
    axes[1, 1].set_xlabel('Decision Threshold', fontsize=12)
    axes[1, 1].set_ylabel('Accuracy', fontsize=12)
    axes[1, 1].set_title('Accuracy vs Decision Threshold', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{dataset_name} Set: Prediction Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 为不同数据集生成不同的文件名
    dataset_suffix = f'_{dataset_name.lower()}'
    
    # 保存PNG版本
    plt.savefig(f'{save_path}{dataset_suffix}.png', dpi=300, bbox_inches='tight')
    # 保存PDF版本
    plt.savefig(f'{save_path}{dataset_suffix}.pdf', bbox_inches='tight', transparent=False)
    plt.close()
    print(f"Prediction distribution plot for {dataset_name} set saved to {save_path}{dataset_suffix}.png and {save_path}{dataset_suffix}.pdf")

# 6. 模型结构可视化（简化版） - 注意：plot_model通常不支持PDF格式
def visualize_model_structure(model, save_path='best_models/model_structure'):
    from tensorflow.keras.utils import plot_model
    
    try:
        # 保存PNG版本
        plot_model(model, to_file=f'{save_path}.png', show_shapes=True, 
                  show_layer_names=True, dpi=300)
        print(f"Model structure plot saved to {save_path}.png")
        
        # 注意：plot_model通常不支持直接保存为PDF
        # 但我们可以尝试保存为SVG或其他矢量格式
        try:
            plot_model(model, to_file=f'{save_path}.svg', show_shapes=True, 
                      show_layer_names=True)
            print(f"Model structure plot also saved to {save_path}.svg")
        except:
            print(f"Could not save model structure as SVG/PDF")
            
    except Exception as e:
        print(f"Could not generate model structure plot: {e}")

# ==============================================
# 生成所有可视化图表 - 同时保存PNG和PDF
# ==============================================

print("\n" + "="*60)
print("Generating Visualization Plots (PNG and PDF)...")
print("="*60)

# 1. 训练历史曲线
plot_training_history(history, 'best_models/training_history')

# 2. ROC曲线对比
plot_roc_curves(y_val, y_val_pred_prob, y_test, y_test_pred_prob, 
                'best_models/roc_curves_comparison')

# 3. 混淆矩阵
plot_confusion_matrix(y_val, y_val_pred_labels, label_encoder.classes_, 
                     "Validation", 'best_models/confusion_matrix')
plot_confusion_matrix(y_test, y_test_pred_labels, label_encoder.classes_, 
                     "Test", 'best_models/confusion_matrix')

# 4. 性能指标对比
metrics_dict = {
    'validation_accuracy': val_accuracy,
    'validation_auc': val_auc_keras,
    'validation_mcc': val_mcc,
    'test_accuracy': test_accuracy,
    'test_auc': test_auc_keras,
    'test_mcc': test_mcc
}
plot_performance_comparison(metrics_dict, 'best_models/performance_comparison')

# 5. 预测概率分布
plot_prediction_distribution(y_val, y_val_pred_prob, "Validation", 
                            'best_models/prediction_distribution')
plot_prediction_distribution(y_test, y_test_pred_prob, "Test", 
                            'best_models/prediction_distribution')

# 6. 模型结构可视化
visualize_model_structure(best_model, 'best_models/model_architecture')

# ==============================================
# 7. 创建所有图表汇总的PDF（可选）
# ==============================================

def create_summary_pdf():
    """创建包含所有图表的汇总PDF（需要reportlab库）"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
        import os
        
        pdf_path = 'best_models/all_visualizations_summary.pdf'
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter
        
        # 图表文件列表
        chart_files = [
            ('Training History', 'best_models/training_history.png'),
            ('ROC Curves Comparison', 'best_models/roc_curves_comparison.png'),
            ('Confusion Matrix - Validation', 'best_models/confusion_matrix_validation.png'),
            ('Confusion Matrix - Test', 'best_models/confusion_matrix_test.png'),
            ('Performance Comparison', 'best_models/performance_comparison.png'),
            ('Prediction Distribution - Validation', 'best_models/prediction_distribution_validation.png'),
            ('Prediction Distribution - Test', 'best_models/prediction_distribution_test.png'),
            ('Model Architecture', 'best_models/model_architecture.png'),
        ]
        
        for i, (title, img_path) in enumerate(chart_files):
            if os.path.exists(img_path):
                # 添加标题
                c.setFont("Helvetica-Bold", 14)
                c.drawString(50, height - 50, f"{i+1}. {title}")
                
                # 添加图像
                try:
                    img = ImageReader(img_path)
                    # 调整图像大小以适应页面
                    img_width, img_height = img.getSize()
                    scale = min((width-100)/img_width, (height-100)/img_height)
                    c.drawImage(img, 50, height - 50 - (img_height*scale + 10), 
                               width=img_width*scale, height=img_height*scale)
                except:
                    c.setFont("Helvetica", 12)
                    c.drawString(50, height - 80, f"Could not load image: {img_path}")
                
                # 添加新页面
                c.showPage()
        
        c.save()
        print(f"\nSummary PDF created: {pdf_path}")
        
    except ImportError:
        print("\nNote: To create a summary PDF, install reportlab: pip install reportlab")
    except Exception as e:
        print(f"\nCould not create summary PDF: {e}")

# 尝试创建汇总PDF
create_summary_pdf()

# ==============================================
# 最终性能总结
# ==============================================

print("\n" + "="*60)
print("FINAL MODEL PERFORMANCE SUMMARY")
print("="*60)
print(f"Validation Set:")
print(f"  - Accuracy: {val_accuracy:.4f}")
print(f"  - Loss: {val_loss:.4f}")
print(f"  - AUC: {val_auc_keras:.4f}")
print(f"  - MCC: {val_mcc:.4f}")
print(f"\nTest Set:")
print(f"  - Accuracy: {test_accuracy:.4f}")
print(f"  - Loss: {test_loss:.4f}")
print(f"  - AUC: {test_auc_keras:.4f}")
print(f"  - MCC: {test_mcc:.4f}")
print("="*60)

# 保存性能指标
performance_metrics = {
    'validation_accuracy': val_accuracy,
    'validation_loss': val_loss,
    'validation_auc_keras': val_auc_keras,
    'validation_auc_sklearn': val_auc_sklearn,
    'validation_mcc': val_mcc,
    'test_accuracy': test_accuracy,
    'test_loss': test_loss,
    'test_auc_keras': test_auc_keras,
    'test_auc_sklearn': test_auc_sklearn,
    'test_mcc': test_mcc,
    'history': history.history
}

with open('best_models/performance_metrics.pkl', 'wb') as f:
    pickle.dump(performance_metrics, f)
print("\nPerformance metrics saved to 'best_models/performance_metrics.pkl'")

# 打印生成的图表列表
print("\n" + "="*60)
print("GENERATED VISUALIZATIONS")
print("="*60)
print("\nPNG and PDF files saved to 'best_models' directory:")

print("\n1. Training History:")
print("   - training_history.png")
print("   - training_history.pdf")

print("\n2. ROC Curves:")
print("   - roc_curves_comparison.png")
print("   - roc_curves_comparison.pdf")

print("\n3. Confusion Matrices:")
print("   - confusion_matrix_validation.png/.pdf")
print("   - confusion_matrix_test.png/.pdf")

print("\n4. Performance Comparison:")
print("   - performance_comparison.png")
print("   - performance_comparison.pdf")

print("\n5. Prediction Distributions:")
print("   - prediction_distribution_validation.png/.pdf")
print("   - prediction_distribution_test.png/.pdf")

print("\n6. Model Architecture:")
print("   - model_architecture.png")
print("   - model_architecture.svg (if supported)")

print("\n7. Summary (if reportlab installed):")
print("   - all_visualizations_summary.pdf")

print("\n" + "="*60)
print("PDF files are suitable for publication and presentations.")
print("="*60)
