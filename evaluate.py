import torch
from torch.utils.data import DataLoader
import sys
import os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from resnet3d import resnet3d_34
from dataset import BrainMRIDataset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

"""
Baseline Model Evaluation Script

Evaluates the original baseline ResNet3D-34 model (trained for 100 epochs)
on the test set and generates comprehensive performance metrics.

This script:
    - Loads the baseline model (best_model_binary_128.pth)
    - Generates predictions on test set
    - Calculates classification and clinical metrics
    - Creates confusion matrix and ROC curve visualizations
    - Saves predictions CSV for demographic analysis

Outputs:
    - results/test_predictions.csv (predictions for demographic analysis)
    - results/confusion_matrix_test.png (confusion matrix heatmap)
    - results/roc_curve_test.png (ROC curve with AUC)

Metrics Computed:
    - Overall accuracy
    - Sensitivity (True Positive Rate / Recall)
    - Specificity (True Negative Rate)
    - PPV (Positive Predictive Value / Precision)
    - NPV (Negative Predictive Value)
    - AUC-ROC (Area Under ROC Curve)

Usage:
    python evaluate.py

Note:
    This evaluates the ORIGINAL baseline model. For lower_lr evaluation,
    use evaluate_all_configs.py or demographic_analysis.py
"""

def evaluate():
    """
        Evaluate baseline model on test set and generate comprehensive metrics.

        Loads the trained baseline model, generates predictions on test data,
        calculates clinical performance metrics, and creates visualizations.

        The function:
            1. Loads model from results/best_model_binary_128.pth
            2. Generates predictions on test set (no augmentation)
            3. Saves predictions CSV for downstream demographic analysis
            4. Computes and prints classification metrics
            5. Creates confusion matrix and ROC curve plots

        Outputs:
            - Prints: Classification report, confusion matrix, clinical metrics
            - Saves: test_predictions.csv, confusion_matrix_test.png, roc_curve_test.png

        Returns:
            None (prints results and saves files)
        """

    device = torch.device('cpu')  # Force CPU (RTX 5080 not supported)
    print(f"Using device: {device}")

    #Load test data
    test_csv = r"C:\Users\todor\PycharmProjects\PyCharm-Work\Msc-AD\data\splits\res_128_binary\test.csv"
    test_dataset = BrainMRIDataset(test_csv)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    #Loading of trained model = ResNet-34
    model = resnet3d_34(num_classes=2).to(device)
    model.load_state_dict(torch.load('results/best_model_binary_128.pth'))
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    # Generate predictions on test set (no gradients needed)

    print("Evaluating on test set:")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of VeryMild class

    #Convert back to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs) # Save predictions for demographic analysis (used by demographic_analysis.py)

    #Saved Predictions For Demographic Analysis
    results_df = pd.read_csv(test_csv)
    results_df['predicted'] = all_preds
    results_df['predicted_prob'] = all_probs
    results_df.to_csv('results/test_predictions.csv', index=False)
    print("Predictions saved for demographic analysis\n")


    #Results
    print("="*60)
    print("Evaluation - Binary Classification")
    print("Task: CN (Cognitively Normal) vs VeryMild Impairment")
    print("="*60)   # Print detailed classification metrics

    #Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['CN', 'VeryMild'], digits=3))

    #Confusion matrix = Creates and saves confusion matrix visualization
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print("              Predicted")
    print("              CN  VeryMild")
    print(f"Actual CN     {cm[0, 0]:3d}  {cm[0, 1]:3d}")
    print(f"       VeryMild {cm[1, 0]:3d}  {cm[1, 1]:3d}")

    #Calculate metrics
    accuracy = (all_preds == all_labels).sum() / len(all_labels)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        print("\nClinical Metrics:")
        print(f"  Accuracy:    {accuracy*100:.2f}%")
        print(f"  Sensitivity: {sensitivity*100:.2f}% (ability to detect VeryMild)")
        print(f"  Specificity: {specificity*100:.2f}% (ability to identify CN correctly)")
        print(f"  PPV:         {ppv*100:.2f}% (precision for VeryMild)")
        print(f"  NPV:         {npv*100:.2f}% (precision for CN)")

        #AUC-ROC
        try:
            auc = roc_auc_score(all_labels, all_probs)
            print(f"  AUC-ROC: {auc:.3f}")
        except:
            print("  AUC-ROC: Could not compute (need probabilities)")

    #Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['CN', 'VeryMild'],
                yticklabels=['CN', 'VeryMild'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix - Test Set\n(CN vs Very Mild Impairment)')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix_test.png', dpi=150)
    print("\n✓ Confusion matrix saved to: results/confusion_matrix_test.png")

    #ROC curve
    if len(np.unique(all_labels)) == 2:
        try:
            fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
            auc = roc_auc_score(all_labels, all_probs)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate (Sensitivity)')
            plt.title('ROC Curve - Early Detection (CN vs VeryMild)')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig('results/roc_curve_test.png', dpi=150)
            print("✓ ROC curve saved to: results/roc_curve_test.png")
        except Exception as e:
            print(f"Could not generate ROC curve: {e}")

    print("\n" + "="*60)

if __name__ == '__main__':
    evaluate()