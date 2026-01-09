"""
Visualize 3D brain MRI scans with model predictions
"""

import torch
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('src')
from models.resnet3d import ResNet3D_18
from torch.utils.data import DataLoader
from training.dataset import BrainMRIDataset
import pandas as pd

def visualize_brain_3d(scan_path, title="Brain MRI"):
    """
    Visualize 3D brain scan as multiple 2D slices
    """
    # Load scan
    img = nib.load(scan_path)
    data = img.get_fdata()

    # Normalize for display
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)

    # Get middle slices from each axis
    mid_sagittal = data.shape[0] // 2
    mid_coronal = data.shape[1] // 2
    mid_axial = data.shape[2] // 2

    # Create figure with multiple slices
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Sagittal slices (side view)
    for i, offset in enumerate([-15, 0, 15]):
        slice_idx = mid_sagittal + offset
        axes[0, i].imshow(data[slice_idx, :, :].T, cmap='gray', origin='lower')
        axes[0, i].set_title(f'Sagittal {slice_idx}')
        axes[0, i].axis('off')

    # Coronal slices (front view)
    for i, offset in enumerate([-15, 0, 15]):
        slice_idx = mid_coronal + offset
        axes[1, i].imshow(data[:, slice_idx, :].T, cmap='gray', origin='lower')
        axes[1, i].set_title(f'Coronal {slice_idx}')
        axes[1, i].axis('off')

    # Axial slices (top view)
    for i, offset in enumerate([-15, 0, 15]):
        slice_idx = mid_axial + offset
        axes[2, i].imshow(data[:, :, slice_idx].T, cmap='gray', origin='lower')
        axes[2, i].set_title(f'Axial {slice_idx}')
        axes[2, i].axis('off')

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig

def visualize_with_predictions():
    """
    Visualize brains with model predictions
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = ResNet3D_18(num_classes=2).to(device)
    model.load_state_dict(torch.load('results/best_model_binary.pth'))
    model.eval()

    # Load test data
    test_csv = r"C:\Users\todor\PycharmProjects\PyCharm-Work\Msc-AD\data\splits\res_96_binary\test.csv"
    df = pd.read_csv(test_csv)

    # Get examples from each class
    cn_examples = df[df['label'] == 0].head(3)
    verymild_examples = df[df['label'] == 1].head(3)

    print("Visualizing CN examples...")
    for idx, row in cn_examples.iterrows():
        scan_path = row['scan_path']
        patient_id = row['patient_id']

        # Get model prediction
        dataset = BrainMRIDataset(test_csv)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        for inputs, labels in loader:
            if labels[0].item() == row['label']:
                inputs = inputs.to(device)
                with torch.no_grad():
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = outputs.max(1)

                pred_class = "CN" if predicted[0].item() == 0 else "VeryMild"
                confidence = probs[0, predicted[0]].item() * 100

                title = f"Patient: {patient_id}\nTrue: CN | Predicted: {pred_class} ({confidence:.1f}% confident)"

                fig = visualize_brain_3d(scan_path, title)
                plt.savefig(f'results/viz_CN_{patient_id}.png', dpi=150, bbox_inches='tight')
                plt.close()
                break

    print("Visualizing VeryMild examples...")
    for idx, row in verymild_examples.iterrows():
        scan_path = row['scan_path']
        patient_id = row['patient_id']

        # Similar prediction code...
        dataset = BrainMRIDataset(test_csv)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        for inputs, labels in loader:
            if labels[0].item() == row['label']:
                inputs = inputs.to(device)
                with torch.no_grad():
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = outputs.max(1)

                pred_class = "CN" if predicted[0].item() == 0 else "VeryMild"
                confidence = probs[0, predicted[0]].item() * 100

                title = f"Patient: {patient_id}\nTrue: VeryMild | Predicted: {pred_class} ({confidence:.1f}% confident)"

                fig = visualize_brain_3d(scan_path, title)
                plt.savefig(f'results/viz_VeryMild_{patient_id}.png', dpi=150, bbox_inches='tight')
                plt.close()
                break

    print("\n✓ Visualizations saved to results/ folder")

def quick_viz():
    """Quick visualization of first test sample"""
    test_csv = r"C:\Users\todor\PycharmProjects\PyCharm-Work\Msc-AD\data\splits\res_96_binary\test.csv"
    df = pd.read_csv(test_csv)

    # Get first sample
    first_scan = df.iloc[0]
    scan_path = first_scan['scan_path']
    diagnosis = first_scan['diagnosis']
    patient_id = first_scan['patient_id']
    age = first_scan['age']
    sex = first_scan['sex']

    title = f"Patient: {patient_id}\nDiagnosis: {diagnosis}\nAge: {age}, Sex: {sex}"

    fig = visualize_brain_3d(scan_path, title)
    plt.savefig('results/example_brain_3d.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: results/example_brain_3d.png")
    plt.show()

if __name__ == '__main__':
    # Quick single visualization
    quick_viz()

    # Uncomment to visualize with predictions:
    # visualize_with_predictions()