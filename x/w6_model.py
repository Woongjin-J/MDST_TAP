import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.init as init

print("Loading US Accidents dataset...")

file_path = 'Data/us_accidents_data_cleaned.csv'

# Use random sampling instead of fixed step skipping
us_accidents = pd.read_csv(file_path).sample(frac=0.3, random_state=42)

# Verify the size of the sample
print(f"Sample size: {len(us_accidents)}")

print("Preprocessing the dataset...")

# Normalize numerical features before splitting
us_accidents['Accident_Duration'] = (us_accidents['Accident_Duration'] - us_accidents['Accident_Duration'].mean()) / us_accidents['Accident_Duration'].std()
us_accidents['Distance(mi)'] = (us_accidents['Distance(mi)'] - us_accidents['Distance(mi)'].mean()) / us_accidents['Distance(mi)'].std()

# Split the dataset into features and target
X = us_accidents[['Traffic_Signal_Flag', 'Crossing_Flag', 'Highway_Flag', 'Distance(mi)',
                  'Start_Hour_Sin', 'Start_Hour_Cos', 'Start_Month_Sin', 'Start_Month_Cos', 'Accident_Duration']].values
y = us_accidents['Severity'].values

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y - 1, dtype=torch.long)  # Shift labels from 1-4 to 0-3
print(f"X shape: {X_tensor.shape}, y shape: {y_tensor.shape}")

train_size = int(0.8 * len(X_tensor))
test_size = len(X_tensor) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(list(zip(X_tensor, y_tensor)), [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print("First few feature rows:\n", X_tensor[:5])
print("First few labels:\n", y_tensor[:5])

class AccidentSeverityModel(nn.Module):
    def __init__(self):
        super(AccidentSeverityModel, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 4)  # Directly outputs 4 classes
        self.relu = nn.ReLU()
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
model = AccidentSeverityModel()
criterion = nn.CrossEntropyLoss()
us_accidents['Accident_Duration'] = (us_accidents['Accident_Duration'] - us_accidents['Accident_Duration'].mean()) / us_accidents['Accident_Duration'].std()

# Print class distribution
print(pd.Series(y).value_counts(normalize=True))

# Update optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# Print dataset size
print(f"Training samples: {train_size}, Test samples: {test_size}")

num_epochs = 20
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

print("Training the model...")
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    all_train_preds = []
    all_train_targets = []

    for batch in train_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        # Compute predictions
        _, preds = torch.max(outputs, 1)
        all_train_preds.extend(preds.cpu().numpy())
        all_train_targets.extend(targets.cpu().numpy())

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Compute train accuracy
    train_accuracy = accuracy_score(all_train_targets, all_train_preds)
    train_accuracies.append(train_accuracy)

    model.eval()
    total_test_loss = 0
    all_test_preds = []
    all_test_targets = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_test_loss += loss.item()

            # Compute predictions
            _, preds = torch.max(outputs, 1)
            all_test_preds.extend(preds.cpu().numpy())
            all_test_targets.extend(targets.cpu().numpy())

    print("Prediction Distribution:", pd.Series(all_test_preds).value_counts(normalize=True))

    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    # Compute test accuracy
    test_accuracy = accuracy_score(all_test_targets, all_test_preds)
    test_accuracies.append(test_accuracy)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Accuracy: {train_accuracy:.4f} | Test Accuracy: {test_accuracy:.4f}")

# Plot training vs test loss
plt.figure(figsize=(8,6))
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss', marker='s')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Testing Loss Over Epochs")
plt.legend()
plt.grid()
plt.show()