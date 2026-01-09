"""
Calibration Analysis
Tests if model confidence scores are reliable across age groups
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import os

os.makedirs('results', exist_ok=True)

print("="*60)
print("Calibration Analysis")
print("Testing model confidence reliability")
print("="*60)

#Load test data with predictions
print("\n1. Loading data...")
test_csv = r"C:\Users\todor\PycharmProjects\PyCharm-Work\Msc-AD\data\splits\res_128_binary\test.csv"
test_df = pd.read_csv(test_csv)

try:
    pred_df = pd.read_csv('results/test_predictions_lower_lr.csv')
    test_df['predicted'] = pred_df['predicted'].values
    test_df['predicted_prob'] = pred_df['predicted_prob'].values
    print(f"Loaded {len(test_df)} test cases")
except:
    print(" No predictions found. Run evaluate.py first.")
    exit()

#Create age groups
test_df['age_group'] = pd.cut(test_df['age'],
                              bins=[0, 70, 80, 100],
                              labels=['<70', '70-80', '>80'])

print("\n2. Analyzing calibration by age group...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

#Plot 1: Overall Calibration
ax = axes[0, 0]
try:
    prob_true, prob_pred = calibration_curve(
        test_df['label'],
        test_df['predicted_prob'],
        n_bins=5,
        strategy='uniform'
    )
    ax.plot(prob_pred, prob_true, 's-', linewidth=2, markersize=10, label='Model')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('True Fraction of Positives', fontsize=12)
    ax.set_title('Overall Calibration Curve\n(All Ages Combined)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
except Exception as e:
    ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')

#Plot 2: Calibration by Age Group
ax = axes[0, 1]
age_groups = ['<70', '70-80', '>80']
colors = ['green', 'orange', 'red']

for age_group, color in zip(age_groups, colors):
    subset = test_df[test_df['age_group'] == age_group]
    if len(subset) > 5:
        try:
            prob_true, prob_pred = calibration_curve(
                subset['label'],
                subset['predicted_prob'],
                n_bins=min(5, len(subset)//2),
                strategy='uniform'
            )
            ax.plot(prob_pred, prob_true, 'o-', linewidth=2,
                   markersize=8, label=f'Age {age_group} (n={len(subset)})',
                   color=color)
        except:
            pass

ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
ax.set_xlabel('Predicted Probability', fontsize=12)
ax.set_ylabel('True Fraction of Positives', fontsize=12)
ax.set_title('Calibration by Age Group', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

#Plot 3: Confidence Distribution
ax = axes[1, 0]

for age_group, color in zip(age_groups, colors):
    subset = test_df[test_df['age_group'] == age_group]
    if len(subset) > 0:
        confidence = subset.apply(lambda row:
            row['predicted_prob'] if row['predicted'] == 1
            else 1 - row['predicted_prob'], axis=1)

        ax.hist(confidence, bins=10, alpha=0.5, label=f'{age_group} (n={len(subset)})',
               color=color, edgecolor='black')

ax.set_xlabel('Model Confidence', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Confidence Distribution by Age', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

#Plot 4: Statistics Table
ax = axes[1, 1]
ax.axis('off')

stats_text = "CALIBRATION STATISTICS:\n\n"

for age_group in age_groups:
    subset = test_df[test_df['age_group'] == age_group]
    if len(subset) > 0:
        confidence = subset.apply(lambda row:
            row['predicted_prob'] if row['predicted'] == 1
            else 1 - row['predicted_prob'], axis=1)

        correct = (subset['label'] == subset['predicted'])

        avg_confidence = confidence.mean()
        avg_accuracy = correct.mean()
        calibration_error = abs(avg_confidence - avg_accuracy)

        stats_text += f"Age {age_group} (n={len(subset)}):\n"
        stats_text += f"  Avg Confidence: {avg_confidence:.3f}\n"
        stats_text += f"  Actual Accuracy: {avg_accuracy:.3f}\n"
        stats_text += f"  Calibration Error: {calibration_error:.3f}\n"

        if calibration_error > 0.1:
            if avg_confidence > avg_accuracy:
                stats_text += f"OVERCONFIDENT\n"
            else:
                stats_text += f"UNDERCONFIDENT\n"
        else:
            stats_text += f" Well calibrated\n"
        stats_text += "\n"

ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
        family='monospace')

plt.tight_layout()
plt.savefig('results/calibration_analysis_lower_lr.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: results/calibration_analysis_lower_lr.png")

print("\n" + "="*60)
print("Calibration Analysis Complete")
print("="*60)