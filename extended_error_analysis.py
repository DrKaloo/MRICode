"""
Extended Error Analysis - Deep dive into model failures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import confusion_matrix
import os

os.makedirs('results', exist_ok=True)

print("="*60)
print("EXTENDED ERROR ANALYSIS")
print("="*60)

# Load data
test_csv = r"C:\Users\todor\PycharmProjects\PyCharm-Work\Msc-AD\data\splits\res_96_binary\test.csv"
test_df = pd.read_csv(test_csv)

try:
    pred_df = pd.read_csv('results/test_predictions.csv')
    test_df['predicted'] = pred_df['predicted'].values
    test_df['predicted_prob'] = pred_df['predicted_prob'].values
except:
    print("   ⚠️ No predictions found. Run evaluate.py first.")
    exit()

# Create error categories
test_df['error_type'] = 'Unknown'
test_df.loc[(test_df['label'] == 0) & (test_df['predicted'] == 0), 'error_type'] = 'TN'
test_df.loc[(test_df['label'] == 0) & (test_df['predicted'] == 1), 'error_type'] = 'FP'
test_df.loc[(test_df['label'] == 1) & (test_df['predicted'] == 0), 'error_type'] = 'FN'
test_df.loc[(test_df['label'] == 1) & (test_df['predicted'] == 1), 'error_type'] = 'TP'

print(f"\nError breakdown:")
print(test_df['error_type'].value_counts())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ========== AGE COMPARISON ==========
ax = axes[0, 0]
error_types = ['TN', 'FP', 'FN', 'TP']
colors = ['green', 'orange', 'red', 'blue']

for i, (error_type, color) in enumerate(zip(error_types, colors)):
    subset = test_df[test_df['error_type'] == error_type]
    if len(subset) > 0:
        bp = ax.boxplot([subset['age']], positions=[i], widths=0.6,
                        patch_artist=True, showfliers=True)
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.7)

ax.set_xticks(range(len(error_types)))
ax.set_xticklabels([f"{et}\n(n={len(test_df[test_df['error_type']==et])})" for et in error_types])
ax.set_ylabel('Age (years)', fontsize=12)
ax.set_title('Age Distribution by Error Type', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# ========== MMSE COMPARISON ==========
ax = axes[0, 1]
if 'mmse' in test_df.columns:
    for i, (error_type, color) in enumerate(zip(error_types, colors)):
        subset = test_df[test_df['error_type'] == error_type]
        if len(subset) > 0:
            bp = ax.boxplot([subset['mmse'].dropna()], positions=[i], widths=0.6,
                            patch_artist=True, showfliers=True)
            bp['boxes'][0].set_facecolor(color)
            bp['boxes'][0].set_alpha(0.7)

    ax.set_xticks(range(len(error_types)))
    ax.set_xticklabels([f"{et}\n(n={len(test_df[test_df['error_type']==et])})" for et in error_types])
    ax.set_ylabel('MMSE Score', fontsize=12)
    ax.set_title('MMSE by Error Type', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

# ========== CONFIDENCE COMPARISON ==========
ax = axes[1, 0]
for i, (error_type, color) in enumerate(zip(error_types, colors)):
    subset = test_df[test_df['error_type'] == error_type]
    if len(subset) > 0:
        confidence = subset.apply(lambda row:
            row['predicted_prob'] if row['predicted'] == 1
            else 1 - row['predicted_prob'], axis=1)

        bp = ax.boxplot([confidence], positions=[i], widths=0.6,
                        patch_artist=True, showfliers=True)
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.7)

ax.set_xticks(range(len(error_types)))
ax.set_xticklabels([f"{et}\n(n={len(test_df[test_df['error_type']==et])})" for et in error_types])
ax.set_ylabel('Model Confidence', fontsize=12)
ax.set_title('Confidence by Error Type', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# ========== SUMMARY TABLE ==========
ax = axes[1, 1]
ax.axis('off')

summary_text = "ERROR STATISTICS:\n\n"

for error_type in error_types:
    subset = test_df[test_df['error_type'] == error_type]
    if len(subset) > 0:
        summary_text += f"{error_type} (n={len(subset)}):\n"
        summary_text += f"  Age: {subset['age'].mean():.1f}±{subset['age'].std():.1f}\n"
        if 'mmse' in subset.columns:
            summary_text += f"  MMSE: {subset['mmse'].mean():.1f}±{subset['mmse'].std():.1f}\n"
        summary_text += "\n"

ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        family='monospace')

plt.tight_layout()
plt.savefig('results/extended_error_analysis.png', dpi=150)
print(f"\n📁 Saved: results/extended_error_analysis.png")

# Statistical tests
print("\n" + "="*60)
print("STATISTICAL TESTS")
print("="*60)

correct = test_df[test_df['label'] == test_df['predicted']]
incorrect = test_df[test_df['label'] != test_df['predicted']]

if len(correct) > 0 and len(incorrect) > 0:
    t_stat, p_val = stats.ttest_ind(correct['age'], incorrect['age'])
    print(f"\nAge: Correct vs Incorrect")
    print(f"   Correct: {correct['age'].mean():.1f}±{correct['age'].std():.1f}")
    print(f"   Incorrect: {incorrect['age'].mean():.1f}±{incorrect['age'].std():.1f}")
    print(f"   p-value: {p_val:.4f} {'✓ SIGNIFICANT' if p_val < 0.05 else ''}")

print("\n" + "="*60)
print("✅ EXTENDED ERROR ANALYSIS COMPLETE")
print("="*60)