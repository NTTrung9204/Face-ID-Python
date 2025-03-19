import torch
from sklearn.metrics import accuracy_score
import sys

def train(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=10):
    train_losses = []
    valid_accuracies = []
    valid_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = batch
            targets = targets.long()

            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            sys.stdout.write(f"\rEpoch [{epoch + 1:4d}/{num_epochs}] Batch [{batch_idx + 1}/{total_batches}], Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        valid_accuracy, valid_loss = evaluate(model, valid_loader, criterion, device)
        valid_accuracies.append(valid_accuracy)
        valid_losses.append(valid_loss)

        print(f"\nEpoch [{epoch + 1:4d}/{num_epochs}], Avg Loss: {avg_train_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}, Validation Loss: {valid_loss:.4f}")

    return train_losses, valid_accuracies, valid_losses

def evaluate(model, valid_loader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0

    with torch.no_grad():
        for batch in valid_loader:
            inputs, targets = batch
            targets = targets.long()
            
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            # Predicted class labels
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    avg_valid_loss = running_loss / len(valid_loader)

    return accuracy, avg_valid_loss