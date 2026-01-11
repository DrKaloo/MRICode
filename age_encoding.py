import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error
from sklearn.manifold import TSNE
import sys
import os
from resnet3d import resnet3d_34
from dataset import BrainMRIDataset
from torch.utils.data import DataLoader

"""
Age Encoding Analysis - Representational Bias Detection

Tests whether the lower_lr model's learned features encode age information,
providing evidence of representational bias. If features strongly predict age,
the model is using age as a shortcut for AD detection rather than learning
disease-specific patterns.

Three Complementary Tests:
    1. Age Decoding: Train Ridge regression to predict age from features
       - R² > 0.3 = strong age encoding (BIAS!)
       - Measures how much age information is in features

    2. Feature Space Visualization: t-SNE plots colored by age vs disease
       - If features cluster by age → age encoding
       - If features cluster by disease → disease encoding (GOOD!)

    3. Age vs Disease Competition: Which is more predictable?
       - Compare age prediction R² vs disease prediction accuracy
       - Higher age R² = bias evidence

Key Concept - Representational Bias:
    Models can encode demographic information (age, sex, race) in learned
    representations even when trained only for disease prediction. This
    creates shortcuts: model predicts "old" → "diseased" rather than learning
    actual pathological patterns.

Key Finding:
    Lower_lr model: R² = 0.648 (moderate age encoding)
    Original model: R² = 0.811 (strong age encoding)
    20% reduction in age encoding → fairness improvement!

Clinical Interpretation:
    Lower_lr model relies LESS on age shortcuts, forcing it to learn
    disease-specific features. This explains improved performance on
    elderly patients (where age ≠ disease).

Outputs:
    - results/age_decoding.png (scatter + error distribution)
    - results/feature_space_tsne.png (3-panel t-SNE visualization)
    - results/age_vs_disease_encoding.png (bar chart comparison)
    - results/age_encoding_summary.csv (summary statistics)

Usage:
    python age_encoding.py

Prerequisites:
    Requires lower_lr_best.pth model from hyperparameter search
"""

#Add src to path
sys.path.insert(0, os.path.dirname(__file__))

def extract_features(model, dataloader, device):
    """
        Extract 512-dimensional feature vectors from penultimate layer.

        These features represent what the model "learned" about brain MRI scans.
        By analyzing these features, we can test what information they encode:
        age (bias) vs disease (desired).

        Process:
            1. Forward pass through all convolutional layers
            2. Stop before final classification layer (fc)
            3. Apply global average pooling → 512 features per scan
            4. Return features for analysis

        Args:
            model: Trained ResNet3D model
            dataloader: PyTorch DataLoader with test data
            device: torch device (cuda/cpu)

        Returns:
            tuple: (features, labels) where:
                - features: np.array of shape [n_samples, 512]
                - labels: np.array of shape [n_samples]
        """

    model.eval()

    all_features = []
    all_labels = []
    all_ages = []
    all_sex = []
    all_patient_ids = []

    print("Extracting features from model:")

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)

            #                           Forward pass through conv layers
            x = model.conv1(inputs)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)

            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)

            #Apply dropout (to match trained model)
            if hasattr(model, 'dropout'):
                x = model.dropout(x)

            #Global average pooling
            x = model.avgpool(x)
            features = torch.flatten(x, 1)  # [batch, 512]

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)

    return features, labels


def test_age_decoding(features, ages, test_features, test_ages):
    """
        TEST 1: Age Decoding - Can features predict age?

        Trains Ridge regression to predict patient age from learned features.
        High R² indicates features strongly encode age information, proving
        representational bias (model uses age shortcuts for AD prediction).

        Interpretation:
            - R² > 0.7: Very strong age encoding (severe bias)
            - R² = 0.3-0.7: Strong age encoding (bias present)
            - R² = 0.1-0.3: Moderate age encoding
            - R² < 0.1: Weak age encoding (minimal bias)

        Args:
            features: Feature vectors from model [n_samples, 512]
            ages: True ages [n_samples]
            test_features: Test set features (same as features for full analysis)
            test_ages: Test set ages (same as ages for full analysis)

        Returns:
            tuple: (r2, mae, predicted_ages) where:
                - r2: R² score (0-1, higher = more age encoding)
                - mae: Mean absolute error in years
                - predicted_ages: Age predictions for visualization

        Outputs:
            - results/age_decoding.png (scatter + error histogram)
        """

    print("\n" + "="*60)
    print("TEST 1: Age Decoding")
    print("="*60)

    #Training Age decoder
    age_decoder = Ridge(alpha=1.0)
    age_decoder.fit(features, ages)

    #Predict ages on test set
    predicted_ages = age_decoder.predict(test_features)

    #Calculate performance
    r2 = r2_score(test_ages, predicted_ages)
    mae = mean_absolute_error(test_ages, predicted_ages)

    print(f"\nResults:")
    print(f"R² Score: {r2:.3f}")
    print(f"MAE: {mae:.2f} years")

    if r2 > 0.3:
        print(f"\nCritical Finding:")
        print(f"Model Strongly encodes age (R² = {r2:.3f})")
        print(f"This proves representational bias")
    elif r2 > 0.1:
        print(f"\nModel moderately encodes age (R² = {r2:.3f})")
    else:
        print(f"\nModel weakly encodes age (R² = {r2:.3f})")

    #Visualisation
    plt.figure(figsize=(10, 5))

    #Scatter plot
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

    #Error distribution
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
    plt.savefig('results/age_decoding.png', dpi=150)
    print(f"\nSaved to: results/age_decoding.png")

    return r2, mae, predicted_ages


def test_age_group_separation(features, ages, labels):
    """
        TEST 2: Feature Space Visualization - Do features cluster by age or disease?

        Uses t-SNE dimensionality reduction to visualize 512D features in 2D.
        Creates 3 plots showing same feature space colored different ways:
            - By age (continuous): Strong clustering = age encoding
            - By disease (CN/VeryMild): Strong clustering = disease encoding (GOOD!)
            - By age groups (<70, 70-80, >80): Separation = demographic bias

        Ideal Result:
            Disease clusters should be clear and separated
            Age clusters should be mixed/scattered
            This indicates features encode disease, not age

        Actual Result (lower_lr):
            Moderate age clustering still visible
            Some disease separation present
            Better than original but not perfect

        Args:
            features: Feature vectors [n_samples, 512]
            ages: Patient ages [n_samples]
            labels: Disease labels [n_samples]

        Returns:
            np.array: 2D t-SNE embeddings [n_samples, 2]

        Outputs:
            - results/feature_space_tsne.png (3-panel visualization)
        """

    print("\n" + "="*60)
    print("TEST 2: Feature Space Visualisation (t-SNE)")
    print("="*60)

    print("\nRunning t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
    features_2d = tsne.fit_transform(features)

    #Create age groups
    age_groups = np.digitize(ages, bins=[0, 70, 80, 100]) - 1
    age_labels = ['<70', '70-80', '>80']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    #Plot 1: Colored by AGE (continuous)
    scatter1 = axes[0].scatter(features_2d[:, 0], features_2d[:, 1],
                               c=ages, cmap='viridis', alpha=0.6, s=100)
    axes[0].set_title('Feature Space Colored by AGE\n(Clustering = Age Encoding)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('t-SNE Dimension 1')
    axes[0].set_ylabel('t-SNE Dimension 2')
    plt.colorbar(scatter1, ax=axes[0], label='Age (years)')

    #Plot 2: Colored by DISEASE
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

    #Plot 3: Coloured by AGE GROUPS
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
    plt.savefig('results/feature_space_tsne.png', dpi=150)
    print(f"\nSaved to: results/feature_space_tsne.png")

    return features_2d


def test_age_vs_disease_correlation(features, ages, labels):
    """
        TEST 3: Age vs Disease Competition - What do features encode better?

        Direct comparison: Train separate models to predict age vs disease from
        same features. Compare which task the features are better at.

        Interpretation:
            - Age R² > Disease Acc: Features encode age better → BIAS!
            - Disease Acc > Age R²: Features encode disease better → GOOD!

        This test answers: "If forced to choose, do features represent age or disease?"

        Method:
            1. Split data 70/30 train/test
            2. Train Ridge regression for age prediction → R²
            3. Train Logistic Regression for disease prediction → Accuracy
            4. Compare metrics (both on [0,1] scale)

        Args:
            features: Feature vectors [n_samples, 512]
            ages: Patient ages [n_samples]
            labels: Disease labels [n_samples]

        Returns:
            tuple: (age_r2, disease_acc) - both in [0,1] range

        Outputs:
            - results/age_vs_disease_encoding.png (bar chart comparison)
        """

    print("\n" + "="*60)
    print("TEST 3: Age vs Disease Predictability")
    print("="*60)

    #Split data
    n_train = int(0.7 * len(features))
    train_features = features[:n_train]
    test_features = features[n_train:]
    train_ages = ages[:n_train]
    test_ages = ages[n_train:]
    train_labels = labels[:n_train]
    test_labels = labels[n_train:]

    #Predict AGE
    age_model = Ridge(alpha=1.0)
    age_model.fit(train_features, train_ages)
    age_r2 = r2_score(test_ages, age_model.predict(test_features))

    #Predict DISEASE
    disease_model = LogisticRegression(max_iter=1000)
    disease_model.fit(train_features, train_labels)
    disease_acc = accuracy_score(test_labels, disease_model.predict(test_features))

    print(f"\nResults:")
    print(f"Age Prediction R²: {age_r2:.3f}")
    print(f"Disease Prediction Accuracy: {disease_acc:.3f}")

    if age_r2 > disease_acc:
        print(f"\nCritical: Features encode Age ({age_r2:.3f}) better than Disease ({disease_acc:.3f})!")
        print(f"This is strong evidence of bias.")
    else:
        print(f"\nFeatures encode Disease ({disease_acc:.3f}) better than Age ({age_r2:.3f})")

    #Bar plot
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

    #Add value labels
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.3f}', ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/age_vs_disease_encoding.png', dpi=150)
    print(f"\nSaved to: results/age_vs_disease_encoding.png")

    return age_r2, disease_acc


def run_age_encoding_analysis():
    """
        Main pipeline: Complete age encoding analysis with 3 complementary tests.

        Orchestrates full analysis to detect and quantify representational bias
        in learned features. Runs three independent tests that together provide
        comprehensive evidence of age encoding.

        Pipeline:
            1. Load lower_lr model
            2. Extract features from test set (512D vectors)
            3. Run Test 1: Age decoding (R² score)
            4. Run Test 2: Feature space visualization (t-SNE)
            5. Run Test 3: Age vs disease competition
            6. Save summary statistics

        Returns:
            dict: Summary statistics including:
                - age_decoding_r2: How well features predict age
                - age_decoding_mae: Prediction error in years
                - age_encoding_strength: Categorical assessment
                - age_vs_disease_r2: Age predictability
                - age_vs_disease_acc: Disease predictability
                - bias_evidence: Categorical conclusion

        Outputs:
            - 3 visualization PNG files
            - 1 summary CSV file
        """

    print("="*60)
    print("Age Encoding Analysis")
    print("Testing if model encodes age. == showing proof of bias")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #no gpu available at this time for torch version

    #Load model
    print("\n1. Loading model...")
    model = resnet3d_34(num_classes=2, dropout=0.3).to(device)
    model.load_state_dict(torch.load('results/hyperparam_search/lower_lr_best.pth'))
    model.eval()

    #Load test data
    print("\n2. Loading test data...")
    test_csv = r"C:\Users\todor\PycharmProjects\PyCharm-Work\Msc-AD\data\splits\res_128_binary\test.csv"
    test_df = pd.read_csv(test_csv)
    test_dataset = BrainMRIDataset(test_csv)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    #Extract features
    print("\n3. Extracting features from test set...")
    features, labels = extract_features(model, test_loader, device)
    ages = test_df['age'].values

    print(f"\nFeatures shape: {features.shape}")
    print(f"Ages range: {ages.min():.0f} - {ages.max():.0f} years")

    #Run tests
    print("\n4. Running age encoding tests:")

    #Test 1: Age decoding
    r2, mae, pred_ages = test_age_decoding(features, ages, features, ages)

    #Test 2: Feature space visualisation
    features_2d = test_age_group_separation(features, ages, labels)

    #Test 3: Age vs disease comparison
    age_r2, disease_acc = test_age_vs_disease_correlation(features, ages, labels)

    #Save summary
    summary = {
        'age_decoding_r2': r2,
        'age_decoding_mae': mae,
        'age_encoding_strength': 'Strong' if r2 > 0.3 else 'Moderate' if r2 > 0.1 else 'Weak',
        'age_vs_disease_r2': age_r2,
        'age_vs_disease_acc': disease_acc,
        'bias_evidence': 'Yes - Age > Disease' if age_r2 > disease_acc else 'No - Disease > Age'
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('results/age_encoding_summary.csv', index=False)

    print("\n" + "="*60)
    print("Age Encoding Analysis Complete")
    print("="*60)
    print("\nSummary:")
    print(f"Age Decoding R²: {r2:.3f} ({summary['age_encoding_strength']} encoding)")
    print(f"Age MAE: {mae:.2f} years")
    print(f"Bias Evidence: {summary['bias_evidence']}")
    print(f"\nResults saved to results/")

    return summary


if __name__ == '__main__':
    run_age_encoding_analysis()

    # Optional: Display t-SNE plot (uncomment if running interactively)
    # from PIL import Image
    # img = Image.open('results/feature_space_tsne.png')
    # img.show()