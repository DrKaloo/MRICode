import torch
from torch.utils.data import DataLoader
import pandas as pd
import sys
import os
from sklearn.metrics import accuracy_score, confusion_matrix

"""
Evaluate All Hyperparameter Configurations on Test Set

Compares performance of all 5 hyperparameter configurations on the test set
with focus on both overall accuracy and demographic fairness (age gap).

This script:
    - Loads each trained model from hyperparameter search
    - Evaluates on test set (41 patients)
    - Calculates overall and age-stratified metrics
    - Computes age gap (performance difference between youngest and oldest)
    - Generates comparison table showing fairness-performance tradeoffs

Key Findings:
    lower_lr configuration achieved best results:
        - Test accuracy: 70.73% (+2.44pp vs baseline)
        - Age gap: 23.9% (-43% vs baseline's 42%)
        - Elderly sensitivity: 40% (doubled from baseline's 20%)

Outputs:
    - Prints: Comparison table with all 5 configurations
    - Saves: results/hyperparam_search/test_results.csv

Metrics Computed:
    - Overall test accuracy
    - Age gap (abs difference between <70 and >80 accuracy)
    - Per-group accuracy (<70, 70-80, >80 years)
    - Elderly sensitivity (>80 years, ability to detect VeryMild)

Usage:
    python evaluate_all_configs.py

Note:
    This script requires models from hyperparameter_search.py to be trained first.
    Each config must have its model saved in results/hyperparam_search/
"""

sys.path.insert(0, os.path.dirname(__file__))
from resnet3d import resnet3d_34
from dataset import BrainMRIDataset

def evaluate_config(config_name, dropout):
    """
        Evaluate a single hyperparameter configuration on test set.

        Loads the trained model for a specific configuration and evaluates it
        on the test set, computing both overall metrics and age-stratified
        performance to assess demographic fairness.

        Args:
            config_name (str): Name of configuration (e.g., 'lower_lr')
            dropout (float): Dropout value used during training (must match model)

        Returns:
            dict: Evaluation results containing:
                - config_name (str): Configuration identifier
                - test_acc (float): Overall test accuracy (%)
                - age_gap (float): Accuracy gap between <70 and >80 age groups (%)
                - age_<70_acc (float): Accuracy for patients <70 years (%)
                - age_70-80_acc (float): Accuracy for patients 70-80 years (%)
                - age_>80_acc (float): Accuracy for patients >80 years (%)
                - age_>80_sens (float): Sensitivity for patients >80 years (%)

        Age Groups:
            - <70: Young patients (n≈16)
            - 70-80: Middle-aged patients (n≈14)
            - >80: Elderly patients (n≈11)

        Age Gap Calculation:
            age_gap = |accuracy(<70) - accuracy(>80)|
            Lower gap = better fairness across age groups
        """

    device = torch.device('cpu')

    #Load test data
    test_csv = r"C:\Users\todor\PycharmProjects\PyCharm-Work\Msc-AD\data\splits\res_128_binary\test.csv"
    test_dataset = BrainMRIDataset(test_csv)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    #Load model
    model = resnet3d_34(num_classes=2, dropout=dropout).to(device)
    model.load_state_dict(torch.load(f'results/hyperparam_search/{config_name}_best.pth'))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    #Calculate metrics
    test_df = pd.read_csv(test_csv)
    test_df['predicted'] = all_preds    # Calculate overall test accuracy

    #Overall accuracy
    overall_acc = accuracy_score(all_labels, all_preds) # Stratify test set by age groups for fairness analysis

    #Age-stratified accuracy
    test_df['age_group'] = pd.cut(test_df['age'],
                                  bins=[0, 70, 80, 100],
                                  labels=['<70', '70-80', '>80'])

    age_results = {}    # Calculate accuracy and sensitivity for each age group
    for group in ['<70', '70-80', '>80']:
        subset = test_df[test_df['age_group'] == group]
        if len(subset) > 0:
            acc = accuracy_score(subset['label'], subset['predicted'])
            cm = confusion_matrix(subset['label'], subset['predicted'])

            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            else:
                sens = 0

            age_results[group] = {'acc': acc, 'sens': sens, 'n': len(subset)}

    #Calculate age gap
    if '<70' in age_results and '>80' in age_results:
        age_gap = abs(age_results['<70']['acc'] - age_results['>80']['acc']) * 100
    else:
        age_gap = None

    return {
        'config_name': config_name,
        'test_acc': overall_acc * 100,
        'age_gap': age_gap,
        'age_<70_acc': age_results.get('<70', {}).get('acc', 0) * 100,
        'age_70-80_acc': age_results.get('70-80', {}).get('acc', 0) * 100,
        'age_>80_acc': age_results.get('>80', {}).get('acc', 0) * 100,
        'age_>80_sens': age_results.get('>80', {}).get('sens', 0) * 100
    }


#Configs with their dropout values
configs = [
    ('baseline', 0.3),
    ('lower_lr', 0.3),
    ('higher_dropout', 0.5),
    ('stronger_weights', 0.3),
    ('conservative', 0.5)
]

results = []
for config_name, dropout in configs:
    print(f"Evaluating {config_name}...")
    result = evaluate_config(config_name, dropout)
    results.append(result)

#Summary
results_df = pd.DataFrame(results)
print("\n" + "="*80)
print("Hyperparameter test - Test Set Results")
print("="*80)
print(results_df.to_string(index=False))
print("\n")
print(f"Age gap range: {results_df['age_gap'].min():.1f}% - {results_df['age_gap'].max():.1f}%")

# Check if all configs exhibit substantial age bias (gap > 30%)
# This shows baseline problem exists across all standard configurations
print(f"All configs show age bias (gap > 30%): {(results_df['age_gap'] > 30).all()}")
print("="*80)

results_df.to_csv('results/hyperparam_search/test_results.csv', index=False)
