import os
import pickle
from rbm_scratch import RBM
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader


def encode_dataset(rbm, data_loader, device):
    # Lists to store encodings and labels
    encodings = []
    labels = []

    # Iterate through the dataset
    for batch, batch_labels in data_loader:
        batch = batch.to(device)  # Move to the correct device
        # Compute hidden neuron activations for the entire batch
        hidden_activations = rbm.visible_to_hidden(batch)
        encodings.append(hidden_activations.cpu())  # Store encodings
        labels.append(batch_labels)  # Store labels

    # Concatenate all batches into a single tensor
    encodings = torch.cat(encodings, dim=0)
    labels = torch.cat(labels, dim=0)

    return encodings, labels


def save_encodings(path, encodings, labels):
    # Save the encoded representations and labels using pickle
    with open(path, 'wb') as f:
        pickle.dump({'encodings': encodings, 'labels': labels}, f)
    print(f"Encodings and labels saved to {path}")


def load_encodings(path):
    # Load the encoded representations and labels using pickle
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"Encodings and labels loaded from {path}")
    return data['encodings'], data['labels']


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    

def train_classifier_on_mnist(train_loader, input_dim, num_classes, device, num_epochs=50, lr=0.001):
    # Define the classifier
    classifier = SimpleClassifier(input_dim, num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    # Training loop
    print("Training classifier...")
    for epoch in range(num_epochs):
        classifier.train()
        epoch_loss = 0
        for batch, labels in train_loader:
            batch, labels = batch.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    return classifier


def evaluate_classifier(classifier, test_loader, device):
    classifier.eval()  # Set the classifier to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch, labels in test_loader:
            batch, labels = batch.to(device), labels.to(device)
            outputs = classifier(batch)
            _, predicted = torch.max(outputs, 1)  # Get the class with the highest score
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy


def main():
    visible_dim = 784  # For MNIST
    hidden_dim = 256  # Number of hidden neurons in RBM
    num_classes = 10  # Digits 0-9

    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    rbm4_path = 'rbm4.pth'
    rbm8_path = 'rbm8.pth'
    rbm4_train_encodings_path = 'rbm4_train_encodings.pkl'
    rbm4_test_encodings_path = 'rbm4_test_encodings.pkl'
    rbm8_train_encodings_path = 'rbm8_train_encodings.pkl'
    rbm8_test_encodings_path = 'rbm8_test_encodings.pkl'
    classifier4_path = "mnist_classifier4.pth"
    classifier8_path = "mnist_classifier8.pth"

    # Initialize RBMs
    rbm4 = RBM(visible_dim, hidden_dim, k=4, lr=0.01, device=device)
    rbm8 = RBM(visible_dim, hidden_dim, k=8, lr=0.01, device=device)

    # Load the MNIST training and test data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    mnist_train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(mnist_train_data, batch_size=10, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnist_test_data, batch_size=10, shuffle=False)

    # Check and load/train RBM4
    if os.path.exists(rbm4_path):
        print("RBM4 model found. Loading model...")
        rbm4.load_model(rbm4_path)
    else:
        print("RBM4 model not found. Training RBM4...")
        rbm4.train(train_loader, epochs=100)
        rbm4.save_model(rbm4_path)

    # Encode training and test datasets using RBM4
    if os.path.exists(rbm4_train_encodings_path) and os.path.exists(rbm4_test_encodings_path):
        print("RBM4 encodings found. Loading encodings...")
        rbm4_train_encodings, rbm4_train_labels = load_encodings(rbm4_train_encodings_path)
        rbm4_test_encodings, rbm4_test_labels = load_encodings(rbm4_test_encodings_path)
    else:
        print("RBM4 encodings not found. Encoding datasets using RBM4...")
        rbm4_train_encodings, rbm4_train_labels = encode_dataset(rbm4, train_loader, device)
        rbm4_test_encodings, rbm4_test_labels = encode_dataset(rbm4, test_loader, device)
        save_encodings(rbm4_train_encodings_path, rbm4_train_encodings, rbm4_train_labels)
        save_encodings(rbm4_test_encodings_path, rbm4_test_encodings, rbm4_test_labels)

    # Check and load/train RBM8
    if os.path.exists(rbm8_path):
        print("RBM8 model found. Loading model...")
        rbm8.load_model(rbm8_path)
    else:
        print("RBM8 model not found. Training RBM8...")
        rbm8.train(train_loader, epochs=100)
        rbm8.save_model(rbm8_path)

    # Encode training and test datasets using RBM8
    if os.path.exists(rbm8_train_encodings_path) and os.path.exists(rbm8_test_encodings_path):
        print("RBM8 encodings found. Loading encodings...")
        rbm8_train_encodings, rbm8_train_labels = load_encodings(rbm8_train_encodings_path)
        rbm8_test_encodings, rbm8_test_labels = load_encodings(rbm8_test_encodings_path)
    else:
        print("RBM8 encodings not found. Encoding datasets using RBM8...")
        rbm8_train_encodings, rbm8_train_labels = encode_dataset(rbm8, train_loader, device)
        rbm8_test_encodings, rbm8_test_labels = encode_dataset(rbm8, test_loader, device)
        save_encodings(rbm8_train_encodings_path, rbm8_train_encodings, rbm8_train_labels)
        save_encodings(rbm8_test_encodings_path, rbm8_test_encodings, rbm8_test_labels)

    # Check if the classifier4 already exists
    if os.path.exists(classifier4_path):
        print("Classifier4 found. Loading classifier4...")
        classifier4 = SimpleClassifier(hidden_dim, num_classes).to(device)
        classifier4.load_state_dict(torch.load(classifier4_path))
        print("Classifier4 loaded.")
    else:
        # Train the classifier4 on MNIST training data
        # Create DataLoader for RBM4-encoded training data
        rbm4_train_dataset = TensorDataset(rbm4_train_encodings, rbm4_train_labels)
        rbm4_train_loader = DataLoader(rbm4_train_dataset, batch_size=100, shuffle=True)
        classifier4 = train_classifier_on_mnist(rbm4_train_loader, hidden_dim, num_classes, device)
        # Save the trained classifier4
        torch.save(classifier4.state_dict(), classifier4_path)
        print("Classifier4 trained and saved.")

    # Check if the classifier8 already exists
    if os.path.exists(classifier8_path):
        print("Classifier8 found. Loading classifier8...")
        classifier8 = SimpleClassifier(hidden_dim, num_classes).to(device)
        classifier8.load_state_dict(torch.load(classifier8_path))
        print("Classifier8 loaded.")
    else:
        # Train the classifier8 on MNIST training data
        # Create DataLoader for RBM8-encoded training data
        rbm8_train_dataset = TensorDataset(rbm8_train_encodings, rbm8_train_labels)
        rbm8_train_loader = DataLoader(rbm8_train_dataset, batch_size=100, shuffle=True)
        classifier8 = train_classifier_on_mnist(rbm8_train_loader, hidden_dim, num_classes, device)
        # Save the trained classifier8
        torch.save(classifier8.state_dict(), classifier8_path)
        print("Classifier8 trained and saved.")

    # Evaluate the classifiers on the MNIST test data encodings
    rbm4_test_dataset = TensorDataset(rbm4_test_encodings, rbm8_test_labels)
    rbm4_test_loader = DataLoader(rbm4_test_dataset, batch_size=100, shuffle=False)
    accuracy4 = evaluate_classifier(classifier4, rbm4_test_loader, device)
    print(f"Classifier4 accuracy: {accuracy4:.4f}")

    rbm8_test_dataset = TensorDataset(rbm8_test_encodings, rbm8_test_labels)
    rbm8_test_loader = DataLoader(rbm8_test_dataset, batch_size=100, shuffle=False)
    accuracy8 = evaluate_classifier(classifier8, rbm8_test_loader, device)
    print(f"Classifier8 accuracy: {accuracy8:.4f}")


if __name__ == "__main__":
    main()