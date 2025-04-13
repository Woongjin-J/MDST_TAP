import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import gc

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print("Loading US Accidents dataset...")

# Load and preprocess data
file_path = 'Data/us_accidents_data_week8.csv'
us_accidents = pd.read_csv(file_path).sample(frac=0.1, random_state=42)

def balance_data(df, target_column):
    print("Original class distribution:")
    print(df[target_column].value_counts())

    # Instead of oversampling to max, undersample majority and oversample minority to mean
    target_counts = df[target_column].value_counts()
    mean_count = int(target_counts.mean())

    balanced_dfs = []
    for class_label in target_counts.index:
        class_df = df[df[target_column] == class_label]
        if len(class_df) > mean_count:
            # Undersample
            balanced_df = class_df.sample(n=mean_count, random_state=42)
        else:
            # Oversample
            balanced_df = resample(class_df, replace=True, n_samples=mean_count, random_state=42)
        balanced_dfs.append(balanced_df)

    df_balanced = pd.concat(balanced_dfs)
    print("\nBalanced class distribution:")
    print(df_balanced[target_column].value_counts())
    return df_balanced

us_accidents = balance_data(us_accidents, 'Severity')

print("Preprocessing the dataset...")

# Standardize all numerical features
scaler = StandardScaler()
numerical_features = ['Accident_Duration', 'Distance(mi)',
                     'Start_Hour_Sin', 'Start_Hour_Cos',
                     'Start_Month_Sin', 'Start_Month_Cos']

us_accidents[numerical_features] = scaler.fit_transform(us_accidents[numerical_features])

X = us_accidents[['Traffic_Signal_Flag', 'Crossing_Flag', 'Highway_Flag', 'Distance(mi)',
                  'Start_Hour_Sin', 'Start_Hour_Cos', 'Start_Month_Sin', 'Start_Month_Cos',
                  'Accident_Duration']].values
y = us_accidents['Severity'].values

# Convert to tensors but keep on CPU initially
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y - 1, dtype=torch.long)

print(f"X shape: {X_tensor.shape}, y shape: {y_tensor.shape}")

# Calculate class weights for weighted loss
class_counts = np.bincount(y-1)
class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device)
class_weights = class_weights / class_weights.sum()  # Normalize weights

train_size = int(0.8 * len(X_tensor))
test_size = len(X_tensor) - train_size

class AccidentDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Split the data
indices = torch.randperm(len(X_tensor))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

train_dataset = AccidentDataset(X_tensor[train_indices], y_tensor[train_indices])
test_dataset = AccidentDataset(X_tensor[test_indices], y_tensor[test_indices])

# Smaller batch size for better generalization
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

# Clear some memory
del us_accidents, X, y
gc.collect()
torch.cuda.empty_cache()

class AccidentSeverityModel(nn.Module):
    def __init__(self, dropout_rate=0.4):
        super(AccidentSeverityModel, self).__init__()
        self.fc1 = nn.Linear(9, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 4)

        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.LeakyReLU(negative_slope=0.1)

        # Initialize weights with He initialization
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc4.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        return x

model = AccidentSeverityModel().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                               patience=3, verbose=True, min_lr=1e-6)

print(f"Training samples: {train_size}, Test samples: {test_size}")

num_epochs = 50
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

min_test_loss = float('inf')
patience, trials = 5, 0  # Increased patience

print("Training the model...")
torch.cuda.empty_cache()

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    all_train_preds = []
    all_train_targets = []

    for batch in train_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_train_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        all_train_preds.extend(preds.cpu().numpy())
        all_train_targets.extend(targets.cpu().numpy())

        del inputs, targets, outputs, loss
        torch.cuda.empty_cache()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    train_accuracy = accuracy_score(all_train_targets, all_train_preds)
    train_accuracies.append(train_accuracy)

    model.eval()
    total_test_loss = 0
    all_test_preds = []
    all_test_targets = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_test_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_test_preds.extend(preds.cpu().numpy())
            all_test_targets.extend(targets.cpu().numpy())

            del inputs, targets, outputs, loss
            torch.cuda.empty_cache()

    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    test_accuracy = accuracy_score(all_test_targets, all_test_preds)
    test_accuracies.append(test_accuracy)

    scheduler.step(avg_test_loss)

    if avg_test_loss < min_test_loss:
        min_test_loss = avg_test_loss
        trials = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': min_test_loss,
        }, 'best_accident_severity_model.pth')
        print("Best model saved.")
    else:
        trials += 1
        if trials >= patience:
            print("Early stopping!")
            break

    report = classification_report(all_test_targets, all_test_preds,
                                target_names=['Severity 1', 'Severity 2', 'Severity 3', 'Severity 4'],
                                output_dict=True, zero_division=0)

    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_test_targets, all_test_preds,
                              target_names=['Severity 1', 'Severity 2', 'Severity 3', 'Severity 4'],
                              zero_division=0))

    for severity, metrics in report.items():
        if isinstance(metrics, dict):
            precision = metrics["precision"]
            recall = metrics["recall"]
            if precision < 0.5 or recall < 0.5:  # Reduced threshold
                print(f"Class {severity} failed to meet precision and recall thresholds.")

# Plot training results
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, len(test_losses)+1), test_losses, label='Test Loss', marker='s')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Testing Loss")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(range(1, len(train_accuracies)+1), train_accuracies, label='Train Accuracy', marker='o')
plt.plot(range(1, len(test_accuracies)+1), test_accuracies, label='Test Accuracy', marker='s')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Testing Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Final cleanup
torch.cuda.empty_cache()
gc.collect()