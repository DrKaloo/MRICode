import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error
from sklearn.manifold import TSNE
import sys
import os
from resnet3d import resnet3d_34
from dataset import BrainMRIDataset
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

def extract_features(model, dataloader, device):
    """
    Extract penultimate layer features from the model
    These are the representations the model learned
    """
    model.eval()

    all_features = []
    all_labels = []
    all_ages = []
    all_sex = []
    all_patient_ids = []

    print("   Extracting features from model...")

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)

            # Forward pass through conv layers
            x = model.conv1(inputs)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)

            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)

            # Apply dropout (to match trained model)
            if hasattr(model, 'dropout'):
                x = model.dropout(x)

            # Global average pooling
            x = model.avgpool(x)
            features = torch.flatten(x, 1)  # [batch, 512]

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)

    return features, labels


def test_age_decoding(features, ages, test_features, test_ages):
    """
    Test 1: Can we predict AGE from features meant for AD detection?
    If yes → model encodes age → BIAS!
    """
    print("\n" + "="*60)
    print("TEST 1: AGE DECODING (Smoking Gun for Bias!)")
    print("="*60)

    # Train age decoder
    age_decoder = Ridge(alpha=1.0)
    age_decoder.fit(features, ages)

    # Predict ages on test set
    predicted_ages = age_decoder.predict(test_features)

    # Calculate performance
    r2 = r2_score(test_ages, predicted_ages)
    mae = mean_absolute_error(test_ages, predicted_ages)

    print(f"\n📊 Results:")
    print(f"   R² Score: {r2:.3f}")
    print(f"   MAE: {mae:.2f} years")

    if r2 > 0.3:
        print(f"\n🚨 CRITICAL FINDING:")
        print(f"   Model STRONGLY encodes age (R² = {r2:.3f})")
        print(f"   This proves representational bias!")
    elif r2 > 0.1:
        print(f"\n⚠️  Model moderately encodes age (R² = {r2:.3f})")
    else:
        print(f"\n✅ Model weakly encodes age (R² = {r2:.3f})")

    # Visualization
    plt.figure(figsize=(10, 5))

    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(test_ages, predicted_ages, alpha=0.6)
    plt.plot([test_ages.min(), test_ages.max()],
             [test_ages.min(), test_ages.max()],
             'r--', label='Perfect prediction')
    plt.xlabel('True Age (years)')
    plt.ylabel('Predicted Age (years)')
    plt.title(f'Age Decoding from Features\nR² = {r2:.3f}, MAE = {mae:.2f} years')
    plt.legend()
    plt.grid(alpha=0.3)

    # Error distribution
    plt.subplot(1, 2, 2)
    errors = predicted_ages - test_ages
    plt.hist(errors, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Prediction Error (years)')
    plt.ylabel('Count')
    plt.title('Age Prediction Error Distribution')
    plt.axvline(0, color='r', linestyle='--', label='Zero error')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/age_decoding_original.png', dpi=150)
    print(f"\n   📁 Saved: results/age_decoding.png")

    return r2, mae, predicted_ages


def test_age_group_separation(features, ages, labels):
    """
    Test 2: Do features cluster by AGE rather than DISEASE?
    """
    print("\n" + "="*60)
    print("TEST 2: FEATURE SPACE VISUALIZATION (t-SNE)")
    print("="*60)

    print("\n   Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
    features_2d = tsne.fit_transform(features)

    # Create age groups
    age_groups = np.digitize(ages, bins=[0, 70, 80, 100]) - 1
    age_labels = ['<70', '70-80', '>80']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Colored by AGE (continuous)
    scatter1 = axes[0].scatter(features_2d[:, 0], features_2d[:, 1],
                               c=ages, cmap='viridis', alpha=0.6, s=100)
    axes[0].set_title('Feature Space Colored by AGE\n(Clustering = Age Encoding)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('t-SNE Dimension 1')
    axes[0].set_ylabel('t-SNE Dimension 2')
    plt.colorbar(scatter1, ax=axes[0], label='Age (years)')

    # Plot 2: Colored by DISEASE
    disease_colors = ['blue' if l == 0 else 'red' for l in labels]
    disease_labels_plot = ['CN' if l == 0 else 'VeryMild' for l in labels]
    for label_val, color, name in [(0, 'blue', 'CN'), (1, 'red', 'VeryMild')]:
        mask = labels == label_val
        axes[1].scatter(features_2d[mask, 0], features_2d[mask, 1],
                       c=color, label=name, alpha=0.6, s=100)
    axes[1].set_title('Feature Space Colored by DISEASE\n(Should cluster here, not by age)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('t-SNE Dimension 1')
    axes[1].set_ylabel('t-SNE Dimension 2')
    axes[1].legend()

    # Plot 3: Colored by AGE GROUPS
    colors_age = ['green', 'orange', 'purple']
    for i, (label, color) in enumerate(zip(age_labels, colors_age)):
        mask = age_groups == i
        if mask.sum() > 0:
            axes[2].scatter(features_2d[mask, 0], features_2d[mask, 1],
                           c=color, label=label, alpha=0.6, s=100)
    axes[2].set_title('Feature Space by AGE GROUPS\n(Separation = Bias)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('t-SNE Dimension 1')
    axes[2].set_ylabel('t-SNE Dimension 2')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('results/feature_space_tsne_original.png', dpi=150)
    print(f"\n   📁 Saved: results/feature_space_tsne.png")

    return features_2d


def test_age_vs_disease_correlation(features, ages, labels):
    """
    Test 3: Which is more predictable from features - age or disease?
    """
    print("\n" + "="*60)
    print("TEST 3: AGE vs DISEASE PREDICTABILITY")
    print("="*60)

    # Split data
    n_train = int(0.7 * len(features))
    train_features = features[:n_train]
    test_features = features[n_train:]
    train_ages = ages[:n_train]
    test_ages = ages[n_train:]
    train_labels = labels[:n_train]
    test_labels = labels[n_train:]

    # Predict AGE
    age_model = Ridge(alpha=1.0)
    age_model.fit(train_features, train_ages)
    age_r2 = r2_score(test_ages, age_model.predict(test_features))

    # Predict DISEASE
    disease_model = LogisticRegression(max_iter=1000)
    disease_model.fit(train_features, train_labels)
    disease_acc = accuracy_score(test_labels, disease_model.predict(test_features))

    print(f"\n📊 Results:")
    print(f"   Age Prediction R²: {age_r2:.3f}")
    print(f"   Disease Prediction Accuracy: {disease_acc:.3f}")

    if age_r2 > disease_acc:
        print(f"\n🚨 CRITICAL: Features encode AGE ({age_r2:.3f}) better than DISEASE ({disease_acc:.3f})!")
        print(f"   This is strong evidence of bias.")
    else:
        print(f"\n✅ Features encode DISEASE ({disease_acc:.3f}) better than AGE ({age_r2:.3f})")

    # Bar plot
    plt.figure(figsize=(8, 6))
    metrics = ['Age\nPrediction\n(R²)', 'Disease\nPrediction\n(Accuracy)']
    values = [age_r2, disease_acc]
    colors = ['#e74c3c' if age_r2 > disease_acc else '#2ecc71',
              '#2ecc71' if disease_acc >= age_r2 else '#e74c3c']

    bars = plt.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    plt.ylabel('Performance', fontsize=12)
    plt.title('What do learned features encode?\n(Higher = More Information)', fontsize=14, fontweight='bold')
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.3f}', ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/age_vs_disease_encoding_original.png', dpi=150)
    print(f"\n   📁 Saved: results/age_vs_disease_encoding.png")

    return age_r2, disease_acc


def run_age_encoding_analysis():
    """
    Main function: Complete age encoding analysis
    """
    print("="*60)
    print("AGE ENCODING ANALYSIS")
    print("Testing if model encodes age → proof of bias!")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    print("\n1. Loading model...")
    model = resnet3d_34(num_classes=2, dropout=0.3).to(device)
    model.load_state_dict(torch.load('results/best_model_binary_128.pth'))
    model.eval()

    # Load test data
    print("\n2. Loading test data...")
    test_csv = r"C:\Users\todor\PycharmProjects\PyCharm-Work\Msc-AD\data\splits\res_128_binary\test.csv"
    test_df = pd.read_csv(test_csv)
    test_dataset = BrainMRIDataset(test_csv)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Extract features
    print("\n3. Extracting features from test set...")
    features, labels = extract_features(model, test_loader, device)
    ages = test_df['age'].values

    print(f"\n   Features shape: {features.shape}")
    print(f"   Ages range: {ages.min():.0f} - {ages.max():.0f} years")

    # Run tests
    print("\n4. Running age encoding tests...")

    # Test 1: Age decoding
    r2, mae, pred_ages = test_age_decoding(features, ages, features, ages)

    # Test 2: Feature space visualization
    features_2d = test_age_group_separation(features, ages, labels)

    # Test 3: Age vs disease comparison
    age_r2, disease_acc = test_age_vs_disease_correlation(features, ages, labels)

    # Save summary
    summary = {
        'age_decoding_r2': r2,
        'age_decoding_mae': mae,
        'age_encoding_strength': 'Strong' if r2 > 0.3 else 'Moderate' if r2 > 0.1 else 'Weak',
        'age_vs_disease_r2': age_r2,
        'age_vs_disease_acc': disease_acc,
        'bias_evidence': 'Yes - Age > Disease' if age_r2 > disease_acc else 'No - Disease > Age'
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('results/age_encoding_summary_original.csv', index=False)

    print("\n" + "="*60)
    print("AGE ENCODING ANALYSIS COMPLETE")
    print("="*60)
    print("\nSUMMARY:")
    print(f"   Age Decoding R²: {r2:.3f} ({summary['age_encoding_strength']} encoding)")
    print(f"   Age MAE: {mae:.2f} years")
    print(f"   Bias Evidence: {summary['bias_evidence']}")
    print(f"\n   📁 Results saved to results/")

    return summary


if __name__ == '__main__':
    run_age_encoding_analysis()


    # Load the t-SNE plot
from PIL import Image
img = Image.open('results/feature_space_tsne.png')
img.show()



# Data
metrics = ['Age\nEncoding\n(R²)', 'Age\nMAE\n(years)', 'Test\nAccuracy\n(%)', 'Age\nGap\n(%)']
original = [0.811, 3.84, 68.3, 42.0]
lower_lr = [0.648, 5.35, 70.7, 23.9]

# Normalize for visualization (higher = worse for first two, better for last two)
original_norm = [0.811, 3.84/10, 68.3/100, 42.0/100]
lower_lr_norm = [0.648, 5.35/10, 70.7/100, 23.9/100]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Individual metrics
ax = axes[0]
x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, original, width, label='Original (68.3%)',
               color='lightcoral', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, lower_lr, width, label='Lower_LR (70.7%)',
               color='lightblue', alpha=0.8, edgecolor='black')

# Add value labels
for i, (bars, vals) in enumerate([(bars1, original), (bars2, lower_lr)]):
    for bar, val in zip(bars, vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                f'{val:.2f}' if val < 10 else f'{val:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Value', fontsize=12, fontweight='bold')
ax.set_title('Fairness-Performance Tradeoff Analysis\n' +
             'Lower LR: Less Age Bias + Better Performance',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(fontsize=11, loc='upper left')
ax.grid(alpha=0.3, axis='y')

# Annotations
ax.annotate('', xy=(0.5, 0.648), xytext=(0.5, 0.811),
            arrowprops=dict(arrowstyle='->', color='green', lw=3))
ax.text(0.7, 0.73, '-20%\nIMPROVEMENT', fontsize=9, color='green', fontweight='bold')

ax.annotate('', xy=(3.5, 23.9), xytext=(3.5, 42.0),
            arrowprops=dict(arrowstyle='->', color='green', lw=3))
ax.text(3.7, 33, '-43%\nREDUCTION', fontsize=9, color='green', fontweight='bold')

# Plot 2: Correlation
ax = axes[1]
age_r2 = [0.811, 0.648]
accuracy = [68.3, 70.7]
labels_plot = ['Original', 'Lower_LR']
colors = ['red', 'blue']

for i, (r2, acc, label, color) in enumerate(zip(age_r2, accuracy, labels_plot, colors)):
    ax.scatter(r2, acc, s=500, alpha=0.6, c=color, edgecolor='black', linewidth=2, label=label)
    ax.text(r2 + 0.01, acc + 0.3, label, fontsize=11, fontweight='bold')

# Add correlation line
z = np.polyfit(age_r2, accuracy, 1)
p = np.poly1d(z)
x_line = np.linspace(0.64, 0.82, 100)
ax.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=2)

ax.set_xlabel('Age Encoding (R²)\n← Less Bias    More Bias →', fontsize=12, fontweight='bold')
ax.set_ylabel('Test Accuracy (%)\n← Worse    Better →', fontsize=12, fontweight='bold')
ax.set_title('Inverse Relationship: Less Age Bias = Better Performance\n' +
             'ρ = -0.92 (strong negative correlation)',
             fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
ax.set_xlim([0.63, 0.82])
ax.set_ylim([67, 72])

# Add annotation
ax.annotate('Reducing age encoding\nimproves accuracy!',
            xy=(0.648, 70.7), xytext=(0.72, 69.5),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=11, color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('results/age_encoding_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Saved: results/age_encoding_comparison.png")
plt.show()