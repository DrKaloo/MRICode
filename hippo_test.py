import pandas as pd
import numpy as np
import nibabel as nib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs('results', exist_ok=True)

print("="*60)
print("BASELINE COMPARISON: Simple Features vs Deep Learning")
print("="*60)

# Load data
print("\n1. Loading test data...")
test_csv = r"C:\Users\todor\PycharmProjects\PyCharm-Work\Msc-AD\data\splits\res_96_binary\test.csv"
train_csv = r"C:\Users\todor\PycharmProjects\PyCharm-Work\Msc-AD\data\splits\res_96_binary\train.csv"

test_df = pd.read_csv(test_csv)
train_df = pd.read_csv(train_csv)

print(f"   Train: {len(train_df)} cases")
print(f"   Test: {len(test_df)} cases")

# ========== EXTRACT SIMPLE FEATURES ==========
print("\n2. Extracting simple features from MRI...")

def extract_simple_features(scan_path):
    """
    Extract simple volumetric features
    - Total brain volume
    - Hippocampal region volume (approximate)
    - Mean intensity
    - Intensity standard deviation
    """
    try:
        img = nib.load(scan_path.replace('/', '\\'))
        data = img.get_fdata()

        # Brain mask (simple threshold)
        brain_mask = data > data.mean()
        brain_volume = brain_mask.sum()

        # Approximate hippocampus region (temporal lobe area)
        # These are rough coordinates - not precise!
        hippocampus_region = data[40:55, 25:50, 30:55]
        hippocampus_volume = (hippocampus_region > hippocampus_region.mean()).sum()

        # Intensity features
        brain_voxels = data[brain_mask]
        mean_intensity = brain_voxels.mean()
        std_intensity = brain_voxels.std()

        return {
            'brain_volume': brain_volume,
            'hippocampus_volume': hippocampus_volume,
            'hippocampus_ratio': hippocampus_volume / brain_volume if brain_volume > 0 else 0,
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
        }
    except Exception as e:
        print(f"      Error: {e}")
        return None

# Extract features for train set
print("   Extracting training features...")
train_features = []
for idx, row in train_df.iterrows():
    features = extract_simple_features(row['scan_path'])
    if features is not None:
        features['label'] = row['label']
        features['age'] = row['age']
        train_features.append(features)
    if (idx + 1) % 20 == 0:
        print(f"      Processed {idx + 1}/{len(train_df)}")

# Extract features for test set
print("   Extracting test features...")
test_features = []
for idx, row in test_df.iterrows():
    features = extract_simple_features(row['scan_path'])
    if features is not None:
        features['label'] = row['label']
        features['age'] = row['age']
        features['patient_id'] = row['patient_id']
        test_features.append(features)
    if (idx + 1) % 10 == 0:
        print(f"      Processed {idx + 1}/{len(test_df)}")

train_feat_df = pd.DataFrame(train_features)
test_feat_df = pd.DataFrame(test_features)

print(f"\n   Extracted features for {len(train_feat_df)} train and {len(test_feat_df)} test cases")


# Save features
train_feat_df.to_csv('results/train_simple_features.csv', index=False)
test_feat_df.to_csv('results/test_simple_features.csv', index=False)

# ========== TRAIN BASELINE MODELS ==========
print("\n3. Training baseline classifiers...")

feature_cols = ['brain_volume', 'hippocampus_volume', 'hippocampus_ratio',
                'mean_intensity', 'std_intensity', 'age']

X_train = train_feat_df[feature_cols].values
y_train = train_feat_df['label'].values
X_test = test_feat_df[feature_cols].values
y_test = test_feat_df['label'].values

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
}

results = []

for model_name, model in models.items():
    print(f"\n   Training {model_name}...")

    # Train
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results.append({
        'Model': model_name,
        'Accuracy': acc,
        'AUC': auc,
        'Features': 'Simple (6 features)'
    })

    print(f"      Accuracy: {acc:.3f}")
    print(f"      AUC: {auc:.3f}")

# ========== COMPARE TO DEEP LEARNING ==========
print("\n4. Loading deep learning results...")

# Try multiple possible paths
possible_paths = [
    'results/test_predictions.csv',
    '../results/test_predictions.csv',
    r'C:\Users\todor\PycharmProjects\PyCharm-Work\Msc-AD\results\test_predictions.csv'
]

pred_df = None
for path in possible_paths:
    if os.path.exists(path):
        try:
            pred_df = pd.read_csv(path)
            print(f"   ✓ Found predictions at: {path}")
            break
        except:
            pass

if pred_df is not None:
    dl_acc = accuracy_score(pred_df['label'], pred_df['predicted'])
    dl_auc = roc_auc_score(pred_df['label'], pred_df['predicted_prob'])

    results.append({
        'Model': 'ResNet3D-34 (Deep Learning)',
        'Accuracy': dl_acc,
        'AUC': dl_auc,
        'Features': 'Learned (63M params)'
    })

    print(f"   Deep Learning Accuracy: {dl_acc:.3f}")
    print(f"   Deep Learning AUC: {dl_auc:.3f}")
else:
    print("   ⚠️ Could not load deep learning results")

# ========== VISUALIZATION ==========
print("\n5. Creating comparison plots...")

results_df = pd.DataFrame(results)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Accuracy comparison
ax = axes[0]
colors = ['lightblue', 'lightgreen', 'lightyellow', 'coral']
bars = ax.bar(range(len(results_df)), results_df['Accuracy'], color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_xticks(range(len(results_df)))
ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Model Comparison: Accuracy\n(Higher = Better)', fontsize=13, fontweight='bold')
ax.set_ylim([0, 1])
ax.grid(alpha=0.3, axis='y')
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance (50%)')

# Add value labels
for bar, val in zip(bars, results_df['Accuracy']):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
            f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

ax.legend()

# Plot 2: AUC comparison
ax = axes[1]
bars = ax.bar(range(len(results_df)), results_df['AUC'], color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_xticks(range(len(results_df)))
ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax.set_ylabel('AUC-ROC', fontsize=12)
ax.set_title('Model Comparison: AUC\n(Higher = Better)', fontsize=13, fontweight='bold')
ax.set_ylim([0, 1])
ax.grid(alpha=0.3, axis='y')
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance (0.5)')

# Add value labels
for bar, val in zip(bars, results_df['AUC']):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
            f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

ax.legend()

plt.tight_layout()
plt.savefig('results/baseline_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n   📁 Saved: results/baseline_comparison.png")

# ========== SUMMARY TABLE ==========
print("\n" + "="*60)
print("BASELINE COMPARISON SUMMARY")
print("="*60)
print(results_df.to_string(index=False))

results_df.to_csv('results/baseline_comparison.csv', index=False)
print(f"\n📁 Saved: results/baseline_comparison.csv")

# ========== FEATURE IMPORTANCE ==========
print("\n6. Feature importance (Random Forest)...")

rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.to_string(index=False))

# ========== KEY INSIGHTS ==========
print("\n" + "="*60)
print("✅ BASELINE COMPARISON COMPLETE")
print("="*60)

print("\n📊 KEY INSIGHTS:")
if 'ResNet3D-34 (Deep Learning)' in results_df['Model'].values:
    dl_row = results_df[results_df['Model'] == 'ResNet3D-34 (Deep Learning)'].iloc[0]
    best_baseline = results_df[results_df['Model'] != 'ResNet3D-34 (Deep Learning)']['Accuracy'].max()

    improvement = dl_row['Accuracy'] - best_baseline
    print(f"   Deep Learning Accuracy: {dl_row['Accuracy']:.3f}")
    print(f"   Best Baseline Accuracy: {best_baseline:.3f}")
    print(f"   Improvement: {improvement:+.3f} ({improvement/best_baseline*100:+.1f}%)")

    if improvement > 0.05:
        print(f"\n   ✓ Deep learning provides meaningful improvement!")
    elif improvement > 0:
        print(f"\n   ✓ Deep learning slightly better than baselines")
    else:
        print(f"\n   ⚠️ Baselines competitive with deep learning!")
        print(f"      Suggests task may not need complex models")

print(f"\n   Most important features: {', '.join(feature_importance.head(3)['Feature'].values)}")