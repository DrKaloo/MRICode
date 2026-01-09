import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from resnet3d import resnet3d_34
from dataset import BrainMRIDataset, RandomFlip3D

def train_with_config(config, config_name):
    """Train model with specific hyperparameter configuration"""

    #Setup
    device = torch.device('cpu')
    print(f"\n{'='*60}")
    print(f"Training Configuration: {config_name}")
    print(f"{'='*60}")
    print(f"Learning Rate: {config['lr']}")
    print(f"Dropout: {config['dropout']}")
    print(f"Weight Decay: {config['weight_decay']}")
    print(f"Class Weights: {config['class_weights']}")
    print(f"{'='*60}\n")

    #Data
    resolution = 128
    task = "binary"
    split_folder = f"res_{resolution}_{task}"

    train_csv = rf"C:\Users\todor\PycharmProjects\PyCharm-Work\Msc-AD\data\splits\{split_folder}\train.csv"
    val_csv = rf"C:\Users\todor\PycharmProjects\PyCharm-Work\Msc-AD\data\splits\{split_folder}\val.csv"

    train_dataset = BrainMRIDataset(train_csv, transform=RandomFlip3D())
    val_dataset = BrainMRIDataset(val_csv)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    #Model with specified dropout
    model = resnet3d_34(num_classes=2, dropout=config['dropout']).to(device)

    #Loss with specified class weights
    class_weights = torch.tensor(config['class_weights'], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    #Optimizer with specified lr and weight_decay
    optimizer = optim.Adam(model.parameters(),
                          lr=config['lr'],
                          weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=5)

    best_val_acc = 0.0
    history = []

    #Training for fewer epochs (50 instead of 100) to save time while learning
    for epoch in range(50):
        #Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/50 - Train"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total

        #Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/50 - Val"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total

        scheduler.step(val_loss)

        #Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('results/hyperparam_search', exist_ok=True)
            torch.save(model.state_dict(),
                      f'results/hyperparam_search/{config_name}_best.pth')

        #Record history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    print(f"\n{config_name} - Best Val Acc: {best_val_acc:.2f}%\n")

    return {
        'config_name': config_name,
        'config': config,
        'best_val_acc': best_val_acc,
        'history': history
    }


def main():
    #Define configurations to test
    configs = [
        {
            'name': 'baseline',
            'lr': 1e-4,
            'dropout': 0.3,
            'weight_decay': 1e-5,
            'class_weights': [1.0, 1.7]
        },
        {
            'name': 'lower_lr',
            'lr': 5e-5,
            'dropout': 0.3,
            'weight_decay': 1e-5,
            'class_weights': [1.0, 1.7]
        },
        {
            'name': 'higher_dropout',
            'lr': 1e-4,
            'dropout': 0.5,
            'weight_decay': 1e-5,
            'class_weights': [1.0, 1.7]
        },
        {
            'name': 'stronger_weights',
            'lr': 1e-4,
            'dropout': 0.3,
            'weight_decay': 1e-5,
            'class_weights': [1.0, 2.5]
        },
        {
            'name': 'conservative',
            'lr': 5e-5,
            'dropout': 0.5,
            'weight_decay': 1e-4,
            'class_weights': [1.0, 2.0]
        }
    ]

    results = []

    for config in configs:
        result = train_with_config(config, config['name'])
        results.append(result)

        #Save intermediate results
        results_df = pd.DataFrame([{
            'config_name': r['config_name'],
            'lr': r['config']['lr'],
            'dropout': r['config']['dropout'],
            'weight_decay': r['config']['weight_decay'],
            'class_weights': str(r['config']['class_weights']),
            'best_val_acc': r['best_val_acc']
        } for r in results])

        results_df.to_csv('results/hyperparam_search/summary.csv', index=False)
        print("\n" + "="*60)
        print("CURRENT RESULTS:")
        print(results_df.to_string(index=False))
        print("="*60 + "\n")

    #Final summary
    print("\n" + "="*60)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("="*60)
    print(results_df.sort_values('best_val_acc', ascending=False).to_string(index=False))
    print("\nBest configuration:", results_df.loc[results_df['best_val_acc'].idxmax(), 'config_name'])
    print(f"Best validation accuracy: {results_df['best_val_acc'].max():.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()