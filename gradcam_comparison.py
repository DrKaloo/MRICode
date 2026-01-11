import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load both
lower_lr = pd.read_csv('results/gradcam/gradcam_statistics.csv')
original = pd.read_csv('results/gradcam_original/gradcam_statistics.csv')

print("="*60)
print("GRAD-CAM COMPARISON: Original vs Lower_LR")
print("="*60)

# Group by category
print("\n📊 HIPPOCAMPUS ATTENTION COMPARISON:\n")
print("Category             | Original | Lower_LR | Improvement")
print("-" * 60)

for category in ['correct_young', 'correct_elderly', 'incorrect_young', 'incorrect_elderly', 'true_CN', 'true_VeryMild']:
    orig_subset = original[original['category'] == category]
    lr_subset = lower_lr[lower_lr['category'] == category]

    if len(orig_subset) > 0 and len(lr_subset) > 0:
        orig_mean = orig_subset['hippocampus_attention'].mean()
        lr_mean = lr_subset['hippocampus_attention'].mean()
        improvement = lr_mean - orig_mean

        print(f"{category:20s} | {orig_mean:.3f}    | {lr_mean:.3f}    | {improvement:+.3f}")

print("\n" + "="*60)




# Data
categories = ['Correct\nYoung', 'Correct\nElderly', 'Incorrect\nYoung', 'Incorrect\nElderly', 'True\nCN', 'True\nVeryMild']
original = [0.553, 0.667, 0.743, 0.719, 0.698, 0.835]
lower_lr = [0.578, 0.689, 0.576, 0.565, 0.669, 0.512]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 7))

bars1 = ax.bar(x - width/2, original, width, label='Original (68.3% acc)',
               color='lightcoral', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, lower_lr, width, label='Lower_LR (70.7% acc)',
               color='lightblue', alpha=0.8, edgecolor='black')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Highlight the key findings
ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.3, label='High attention (>0.7)')
ax.text(5.5, 0.72, 'OVERFITTING ZONE', fontsize=10, color='red',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

ax.set_xlabel('Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Hippocampal Attention', fontsize=12, fontweight='bold')
ax.set_title('Grad-CAM Comparison: Excessive Hippocampal Focus Correlates with Errors\n' +
             'Lower_LR achieves better accuracy with MORE BALANCED attention',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(fontsize=11)
ax.grid(alpha=0.3, axis='y')
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('results/gradcam_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: results/gradcam_comparison.png")
plt.show()