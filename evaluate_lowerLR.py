import torch
from torch.utils.data import DataLoader
import sys
import numpy as np
import pandas as pd
import os

sys.path.insert(0, os.path.dirname(__file__))
from resnet3d import resnet3d_34
from dataset import BrainMRIDataset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate():

    device = torch.device('cpu')
    print(f"Using device: {device}")

    #Load test data
    test_csv = r"C:\Users\todor\PycharmProjects\PyCharm-Work\Msc-AD\data\splits\res_128_binary\test.csv"
    test_dataset = BrainMRIDataset(test_csv)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    #Loading lower_lr model (best model)
    model = resnet3d_34(num_classes=2, dropout=0.3).to(device)
    model.load_state_dict(torch.load('results/hyperparam_search/lower_lr_best.pth'))
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    print("Evaluating lower_lr model on test set:")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    #Convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    #Save predictions for demographic analysis
    results_df = pd.read_csv(test_csv)
    results_df['predicted'] = all_preds
    results_df['predicted_prob'] = all_probs
    results_df.to_csv('results/test_predictions_lower_lr.csv', index=False)
    print("✓ Predictions saved to: results/test_predictions_lower_lr.csv\n")

    #Results
    print("="*60)
    print("LOWER_LR MODEL EVALUATION - Binary Classification")
    print("Task: CN (Cognitively Normal) vs VeryMild Impairment")
    print("="*60)

    #Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['CN', 'VeryMild'], digits=3))

    #Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print("              Predicted")
    print("              CN  VeryMild")
    print(f"Actual CN     {cm[0,0]:3d}  {cm[0,1]:3d}")
    print(f"       VeryMild {cm[1,0]:3d}  {cm[1,1]:3d}")

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

        # AUC-ROC
        try:
            auc = roc_auc_score(all_labels, all_probs)
            print(f"  AUC-ROC:     {auc:.3f}")
        except:
            print("  AUC-ROC:     Could not compute")

    #Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['CN', 'VeryMild'],
                yticklabels=['CN', 'VeryMild'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix - Lower_LR Model\n(CN vs Very Mild Impairment)')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix_test_lower_lr.png', dpi=150)
    print("\n✓ Confusion matrix saved to: results/confusion_matrix_test_lower_lr.png")

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
            plt.title('ROC Curve - Lower_LR Model (CN vs VeryMild)')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig('results/roc_curve_test_lower_lr.png', dpi=150)
            print("✓ ROC curve saved to: results/roc_curve_test_lower_lr.png")
        except Exception as e:
            print(f"Could not generate ROC curve: {e}")

    print("\n" + "="*60)

if __name__ == '__main__':
    evaluate()