import os
import pickle
from rbm_scratch import RBM
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


def encode_test_images(rbm, test_loader, device):
    # Dictionary to store one image per digit (0-9)
    digit_representations = {}

    # Iterate through the test dataset
    for batch, labels in test_loader:
        batch = batch.to(device)  # Move to the correct device
        for img, label in zip(batch, labels):
            label = label.item()  # Convert label tensor to integer
            if label not in digit_representations:
                # Compute hidden neuron activations
                hidden_activations = rbm.visible_to_hidden(img.view(1, -1))
                digit_representations[label] = hidden_activations
            # Stop if we have one image per digit
            if len(digit_representations) == 10:
                break
        if len(digit_representations) == 10:
            break

    return digit_representations


def save_encodings(path, encodings):
    # Save the encoded representations using pickle
    with open(path, 'wb') as f:
        pickle.dump(encodings, f)
    print(f"Encodings saved to {path}")


def load_encodings(path):
    # Load the encoded representations using pickle
    with open(path, 'rb') as f:
        encodings = pickle.load(f)
    print(f"Encodings loaded from {path}")
    return encodings

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)  # Fully connected layer

    def forward(self, x):
        return self.fc(x)
    
def train_classifier_on_mnist(train_loader, input_dim, num_classes, device, num_epochs=100, lr=0.001):
    # Define the classifier
    classifier = SimpleClassifier(input_dim, num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    # Training loop
    print("Training classifier on raw MNIST data...")
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

def classify_encodings(rbm, classifier, encodings, device):
    classifier.eval()  # Set the classifier to evaluation mode

    # Reconstruct the original inputs from the encodings
    reconstructed_inputs = []
    for encoding in encodings.values():
        reconstructed = rbm.hidden_to_visible(encoding).view(-1)  # Reconstruct and flatten
        reconstructed_inputs.append(reconstructed)

    # Stack the reconstructed inputs into a single tensor
    reconstructed_tensor = torch.stack(reconstructed_inputs).to(device)

    with torch.no_grad():  # Disable gradient computation for evaluation
        outputs = classifier(reconstructed_tensor)
        _, predicted = torch.max(outputs, 1)  # Get the class with the highest score

    return predicted.cpu().numpy()  # Return predictions as a NumPy array


def main():
    visible_dim = 784  # For MNIST
    hidden_dim = 256
    num_classes = 10  # Digits 0-9

    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    rbm4_path = 'rbm4.pth'
    rbm8_path = 'rbm8.pth'
    rbm4_encodings_path = 'rbm4_encodings.pkl'
    rbm8_encodings_path = 'rbm8_encodings.pkl'
    classifier_path = "mnist_classifier.pth"

    # Initialize RBMs
    rbm4 = RBM(visible_dim, hidden_dim, k=4, lr=0.01, device=device)
    rbm8 = RBM(visible_dim, hidden_dim, k=8, lr=0.01, device=device)

    # Load the MNIST training data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(mnist_data, batch_size=10, shuffle=True)

    # Check and load/train RBM4
    if os.path.exists(rbm4_path):
        print("RBM4 model found. Loading model...")
        rbm4.load_model(rbm4_path)
    else:
        print("RBM4 model not found. Training RBM4...")
        rbm4.train(data_loader, epochs=100)
        rbm4.save_model(rbm4_path)

    # Check and load/train RBM8
    if os.path.exists(rbm8_path):
        print("RBM8 model found. Loading model...")
        rbm8.load_model(rbm8_path)
    else:
        print("RBM8 model not found. Training RBM8...")
        rbm8.train(data_loader, epochs=100)
        rbm8.save_model(rbm8_path)

    # Load MNIST test data
    mnist_test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(mnist_test_data, batch_size=10, shuffle=False)

    # Encode test images using RBM4
    if os.path.exists(rbm4_encodings_path):
        print("RBM4 encodings found. Loading encodings...")
        rbm4_representations = load_encodings(rbm4_encodings_path)
    else:
        print("RBM4 encodings not found. Encoding test images using RBM4...")
        rbm4_representations = encode_test_images(rbm4, test_loader, device)
        save_encodings(rbm4_encodings_path, rbm4_representations)

    # Encode test images using RBM8
    if os.path.exists(rbm8_encodings_path):
        print("RBM8 encodings found. Loading encodings...")
        rbm8_representations = load_encodings(rbm8_encodings_path)
    else:
        print("RBM8 encodings not found. Encoding test images using RBM8...")
        rbm8_representations = encode_test_images(rbm8, test_loader, device)
        save_encodings(rbm8_encodings_path, rbm8_representations)

    # Check if the classifier already exists
    if os.path.exists(classifier_path):
        print("Classifier found. Loading classifier...")
        classifier = SimpleClassifier(visible_dim, num_classes).to(device)
        classifier.load_state_dict(torch.load(classifier_path))
        print("Classifier loaded.")
    else:
        # Train the classifier on MNIST training data
        classifier = train_classifier_on_mnist(data_loader, visible_dim, num_classes, device)
        # Save the trained classifier
        torch.save(classifier.state_dict(), classifier_path)
        print("Classifier trained and saved.")

    # Evaluate the classifier on the MNIST test data
    accuracy = evaluate_classifier(classifier, test_loader, device)
    print(f"Classifier accuracy on MNIST test data: {accuracy:.4f}")

    # Classify RBM4 encodings
    print("Classifying RBM4 encodings...")
    rbm4_predictions = classify_encodings(rbm4, classifier, rbm4_representations, device)
    print("RBM4 Predictions:")
    for digit, prediction in zip(rbm4_representations.keys(), rbm4_predictions):
        print(f"Original Digit: {digit}, Predicted: {prediction}")

    # Classify RBM8 encodings
    print("Classifying RBM8 encodings...")
    rbm8_predictions = classify_encodings(rbm8, classifier, rbm8_representations, device)
    print("RBM8 Predictions:")
    for digit, prediction in zip(rbm8_representations.keys(), rbm8_predictions):
        print(f"Original Digit: {digit}, Predicted: {prediction}")


if __name__ == "__main__":
    main()