import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from resnet3d import resnet3d_34
from dataset import BrainMRIDataset, RandomFlip3D

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100. * correct / total

def main():
    #Binary Classification = CN vs Very Mild
    task = "binary"  #Changing to "3class" for CN vs VeryMild vs Dementia 0, 0.5, 1

    resolution = 128
    split_folder = f"res_{resolution}_{task}"

    train_csv = rf"C:\Users\todor\PycharmProjects\PyCharm-Work\Msc-AD\data\splits\{split_folder}\train.csv"
    val_csv = rf"C:\Users\todor\PycharmProjects\PyCharm-Work\Msc-AD\data\splits\{split_folder}\val.csv"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Testing GPU...")
    test = torch.randn(1, 1, 128, 128, 128).to(device)
    print(f"GPU test successful! Tensor on: {test.device}")
    del test
    torch.cuda.empty_cache()

    train_dataset = BrainMRIDataset(train_csv, transform=RandomFlip3D())
    val_dataset = BrainMRIDataset(val_csv)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    print(f"\nTask: {'Binary (CN vs VeryMild)' if task == 'binary' else '3-Class'}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    num_classes = 2 if task == "binary" else 3
    model = resnet3d_34(num_classes=2)
    model = model.to(device)
    print(f"Model moved to: {next(model.parameters()).device}")

    print(f"Model: ResNet3D-34 ({num_classes} classes)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    class_weights = torch.tensor([70.0, 135.0]).to(device)
    class_weights = torch.tensor([1.0, 1.7], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_acc = 0.0

    for epoch in range(100):
        print(f"\nEpoch {epoch+1}/100")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('results', exist_ok=True)
            torch.save(model.state_dict(), f'results/best_model_{task}_128.pth')
            print(f"Saved (Val Acc: {val_acc:.2f}%)")

    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"Best Val Acc: {best_val_acc:.2f}%")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()