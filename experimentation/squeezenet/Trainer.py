import torch
import matplotlib.pyplot as plt


class ModelTrainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

    def train(self, train_loader, val_loader, n_epochs):
        for epoch in range(n_epochs):
            self.model.train()
            running_loss,total,correct = 0,0,0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (preds == targets).sum().item()
            accuracy = 100 * correct / total
            self.train_loss.append(round(running_loss / len(train_loader),4))
            self.train_acc.append(round(accuracy,4))
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {running_loss/len(train_loader):.4f},Accuracy: {accuracy:.2f}%")

            # optional: validation loop
            self.validate(val_loader)

    def validate(self, val_loader,test=False):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            running_loss = 0.0
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (preds == targets).sum().item()
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
        acc = 100 * correct / total
        if not test:
            self.val_loss.append(round(running_loss / len(val_loader),4))
            self.val_acc.append(round(100 * correct / total,4))
            print(f"Validation Accuracy: {acc:.2f}%")
        else:
            print(f"Test Accuracy: {acc:.2f}%")
            
    def plot_metrics(self):
        
        epochs = range(1, len(self.train_loss) + 1)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_loss, 'b', label='Training loss')
        plt.plot(epochs, self.val_loss, 'r', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_acc, 'b', label='Training Accuracy')
        plt.plot(epochs, self.val_acc, 'r', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.savefig('results/training-metrics.png')
        plt.show()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)