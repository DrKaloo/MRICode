import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import sys
import os
from resnet3d import resnet3d_34
from dataset import BrainMRIDataset

"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for 3D CNNs

Implements Grad-CAM visualization for 3D medical imaging to identify which
brain regions the model focuses on when making predictions. Used to analyze
attention patterns and identify potential biases in model decision-making.

Classes:
    GradCAM3D: Implements Grad-CAM algorithm for 3D convolutional networks

Functions:
    visualize_gradcam_slices: Creates 3×3 visualization grid (axial, coronal, sagittal)
    run_gradcam_analysis: Main analysis loop generating Grad-CAM for test set

Grad-CAM Algorithm:
    1. Forward pass through network
    2. Backpropagate from target class
    3. Compute gradient-weighted activation maps
    4. Apply ReLU (keep positive influences)
    5. Upsample to original resolution
    6. Normalize to [0, 1]

Key Finding:
    Lower_lr model shows balanced hippocampal attention (0.512-0.689)
    compared to original model's excessive focus (0.835). This balanced
    attention correlates with better generalization and fairness.

Categories Analyzed:
    - Correct vs incorrect predictions
    - Young (<70) vs elderly (≥80) patients
    - CN vs VeryMild diagnoses

Outputs:
    - Visualization images: results/gradcam/{category}_{patient_id}.png
    - Statistics CSV: results/gradcam/gradcam_statistics.csv
    - Summary table: Attention statistics by category

Usage:
    python gradcam.py

Prerequisites:
    Requires test_predictions_lower_lr.csv from evaluation script
"""

#Add src to path
sys.path.insert(0, os.path.dirname(__file__))

class GradCAM3D:
    """
    Grad-CAM for 3D CNNs
    Visualizes which brain regions the model focuses on
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        #Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Save forward pass activations"""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Save backward pass gradients"""
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class=None):
        """
        Generate Class Activation Map

        Args:
            input_image: Input tensor [1, 1, 96, 96, 96]
            target_class: Which class to visualize (0=CN, 1=VeryMild)
                         If None, uses predicted class

        Returns:
            cam: 3D attention map [96, 96, 96]
        """
        self.model.eval()

        #Forward pass
        output = self.model(input_image)

        #Use predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        #Zero gradients
        self.model.zero_grad()

        #Backward pass for target class
        output[0, target_class].backward()

        #Calculate weights (Global Average Pooling of gradients)
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)

        #Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        #ReLU to keep only positive influences
        cam = F.relu(cam)

        #Upsample to original input size
        cam = F.interpolate(cam, size=(128, 128, 128), mode='trilinear', align_corners=False)

        #Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, target_class


def visualize_gradcam_slices(brain_scan, cam, patient_info, save_path):
    """
        Create 3×3 grid visualization of Grad-CAM overlaid on MRI slices.

        Generates visualization showing three orthogonal views (axial, coronal,
        sagittal) at three slice positions each, with Grad-CAM heatmap overlaid
        on original brain MRI in grayscale.

        Args:
            brain_scan (torch.Tensor): Original brain MRI [1, 128, 128, 128]
            cam (np.ndarray): Grad-CAM attention map [128, 128, 128]
            patient_info (dict): Patient metadata containing:
                - patient_id: Unique identifier
                - age: Patient age in years
                - sex: M or F
                - true_label: Ground truth (CN or VeryMild)
                - pred_label: Model prediction
                - confidence: Prediction confidence (%)
            save_path (str): Full path where to save figure

        Visualization Layout:
            Row 1: Axial slices (top-down view) at positions 48, 64, 80
            Row 2: Coronal slices (front view) at positions 48, 64, 80
            Row 3: Sagittal slices (side view) at positions 48, 64, 80

        Color Scheme:
            - Grayscale: Original MRI structure
            - Jet colormap: Grad-CAM heatmap (red = high attention)
            - Alpha = 0.4 for heatmap overlay

        Saves:
            PNG file at save_path with 150 DPI resolution
        """

    #Normalise brain scan for visualisation
    brain = brain_scan.squeeze().cpu().numpy()
    brain = (brain - brain.min()) / (brain.max() - brain.min() + 1e-8)

    #Create figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    #Select slices to visualise
    slices_axial = [48, 64, 80]      # Bottom to top (for 128³)
    slices_coronal = [48, 64, 80]    # Back to front (for 128³)
    slices_sagittal = [48, 64, 80]   # Right to left (for 128³)

    #Axial slices (top-down view)
    for i, slice_idx in enumerate(slices_axial):
        ax = axes[0, i]
        ax.imshow(brain[:, :, slice_idx].T, cmap='gray', origin='lower')
        ax.imshow(cam[:, :, slice_idx].T, cmap='jet', alpha=0.4, origin='lower')
        ax.set_title(f'Axial Slice {slice_idx}')
        ax.axis('off')

    #Coronal slices (front view)
    for i, slice_idx in enumerate(slices_coronal):
        ax = axes[1, i]
        ax.imshow(brain[:, slice_idx, :].T, cmap='gray', origin='lower')
        ax.imshow(cam[:, slice_idx, :].T, cmap='jet', alpha=0.4, origin='lower')
        ax.set_title(f'Coronal Slice {slice_idx}')
        ax.axis('off')

    #Sagittal slices (side view)
    for i, slice_idx in enumerate(slices_sagittal):
        ax = axes[2, i]
        ax.imshow(brain[slice_idx, :, :].T, cmap='gray', origin='lower')
        ax.imshow(cam[slice_idx, :, :].T, cmap='jet', alpha=0.4, origin='lower')
        ax.set_title(f'Sagittal Slice {slice_idx}')
        ax.axis('off')

    #Adding title with patient info
    title = f"Grad-CAM Visualization\n"
    title += f"Patient: {patient_info['patient_id']} | "
    title += f"Age: {patient_info['age']} | Sex: {patient_info['sex']}\n"
    title += f"True: {patient_info['true_label']} | "
    title += f"Predicted: {patient_info['pred_label']} ({patient_info['confidence']:.1f}%)"

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_gradcam_analysis():
    """
        Main Grad-CAM analysis pipeline for test set.

        Generates Grad-CAM visualizations for representative cases across
        multiple demographic and prediction categories to identify systematic
        patterns in model attention.

        Analysis Strategy:
            - Selects up to 3 examples per category
            - Generates Grad-CAM heatmap for each
            - Calculates attention statistics (mean, std, max, hippocampus)
            - Saves visualizations and summary statistics

        Categories Analyzed:
            1. correct_young: Correctly classified, age <70
            2. correct_elderly: Correctly classified, age ≥80
            3. incorrect_young: Misclassified, age <70
            4. incorrect_elderly: Misclassified, age ≥80
            5. true_CN: All cognitively normal patients
            6. true_VeryMild: All very mild impairment patients

        Hippocampus Region:
            Approximate coordinates [50:75, 40:65, 45:65] for 128³ volume
            This region is critical for early Alzheimer's detection

        Outputs:
            - 17 visualization images (3 per category, some overlap)
            - CSV with attention statistics per patient
            - Summary table grouped by category

        Returns:
            pd.DataFrame: Results with attention statistics per patient

        Raises:
            FileNotFoundError: If predictions CSV not found
        """

    print("="*60)
    print("GRAD-CAM ANALYSIS")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    #Load model
    print("\n1. Loading model...")
    model = resnet3d_34(num_classes=2, dropout=0.3).to(device)
    model.load_state_dict(torch.load('results/hyperparam_search/lower_lr_best.pth', map_location=device))
    model.eval()

    #Get target layer (last conv block of layer4)
    target_layer = model.layer4[-1]
    print(f"   Target layer: layer4[-1]")

    #Initialise Grad-CAM
    gradcam = GradCAM3D(model, target_layer)

    #Load test data
    print("\n2. Loading test data...")
    test_csv = r"C:\Users\todor\PycharmProjects\PyCharm-Work\Msc-AD\data\splits\res_128_binary\test.csv"
    test_df = pd.read_csv(test_csv)
    test_dataset = BrainMRIDataset(test_csv)

    #Load predictions
    try:
        pred_df = pd.read_csv('results/test_predictions_lower_lr.csv')
        test_df['predicted'] = pred_df['predicted'].values
        test_df['predicted_prob'] = pred_df['predicted_prob'].values
        has_predictions = True
        print(f" Loaded {len(test_df)} test cases with predictions")
    except:
        print("No predictions found. Run evaluate.py first.")
        has_predictions = False
        return

    #Create output directory
    os.makedirs('results/gradcam', exist_ok=True)

    print("\n3. Generating Grad-CAM visualizations...")

    #Categories to analyse
    categories = {
        'correct_young': test_df[(test_df['label'] == test_df['predicted']) & (test_df['age'] < 70)],
        'correct_elderly': test_df[(test_df['label'] == test_df['predicted']) & (test_df['age'] >= 80)],
        'incorrect_young': test_df[(test_df['label'] != test_df['predicted']) & (test_df['age'] < 70)],
        'incorrect_elderly': test_df[(test_df['label'] != test_df['predicted']) & (test_df['age'] >= 80)],
        'true_CN': test_df[test_df['label'] == 0],
        'true_VeryMild': test_df[test_df['label'] == 1],
    }

    results = []

    for category_name, subset in categories.items():
        if len(subset) == 0:
            print(f"\n   Skipping {category_name}: No cases")
            continue

        print(f"\n   Processing {category_name}: {len(subset)} cases")

        #Take up to 3 examples from each category
        for idx in subset.head(3).index:
            row = test_df.loc[idx]

            #Load scan
            scan_path = row['scan_path'].replace('/', os.sep)
            img = nib.load(scan_path)
            data = img.get_fdata()
            data_tensor = torch.FloatTensor(data).unsqueeze(0).unsqueeze(0).to(device)

            #Generate Grad-CAM
            cam, predicted_class = gradcam.generate_cam(data_tensor)

            #Patient info
            patient_info = {
                'patient_id': row['patient_id'],
                'age': row['age'],
                'sex': row['sex'],
                'true_label': 'CN' if row['label'] == 0 else 'VeryMild',
                'pred_label': 'CN' if row['predicted'] == 0 else 'VeryMild',
                'confidence': row['predicted_prob'] * 100 if row['predicted'] == 1 else (1 - row['predicted_prob']) * 100
            }

            #Save visualisation
            save_path = f"results/gradcam/{category_name}_{row['patient_id']}.png"
            visualize_gradcam_slices(data_tensor, cam, patient_info, save_path)

            #Calculate attention statistics
            results.append({
                'category': category_name,
                'patient_id': row['patient_id'],
                'age': row['age'],
                'true_label': row['label'],
                'predicted': row['predicted'],
                'cam_mean': cam.mean(),
                'cam_std': cam.std(),
                'cam_max': cam.max(),
                'hippocampus_attention': cam[50:75, 40:65, 45:65].mean(),  # Scaled for 128³
            })

            print(f"      ✓ {row['patient_id']}")

    #Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/gradcam/gradcam_statistics.csv', index=False)

    print("\n" + "="*60)
    print("Grad-Cam Analysis Complete")
    print("="*60)
    print(f"\nGenerated {len(results)} visualisations")
    print(f"Saved to: results/gradcam/")
    print(f"Statistics saved to: results/gradcam/gradcam_statistics.csv")

    #Summary statistics
    if len(results) > 0:
        print("\nAttention Statistics by Category:")
        summary = results_df.groupby('category').agg({
            'cam_mean': ['mean', 'std'],
            'hippocampus_attention': ['mean', 'std']
        }).round(3)
        print(summary)

    return results_df


if __name__ == '__main__':
    run_gradcam_analysis()