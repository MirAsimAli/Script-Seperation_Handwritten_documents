import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import csv  # Add this line to import the csv module
from PIL import Image  # Added for binarization

# Binarization function
def binarize(img, threshold=128):
    img = img.convert('L')
    return img.point(lambda p: 255 if p > threshold else 0).convert('L')

# Constants

# Function to get class names and counts
def get_class_info(directory):
    classes = sorted(os.listdir(directory))
    class_lengths = {class_name: len(os.listdir(os.path.join(directory, class_name))) for class_name in classes}
    return classes, class_lengths

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        # this is the original getitem method
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

# 1. Data Loading and Transformation
def load_data(TEST, batch_size=10):  # 25
    transform_test = transforms.Compose([
        transforms.Lambda(lambda img: binarize(img, threshold=128)),
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = ImageFolderWithPaths(root=TEST, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    classes = sorted(os.listdir(TEST))
    return test_loader, classes, len(test_dataset)

# 2. Model Loading and Configuration
def load_model(model_weights_path, num_classes, device):
    model = models.resnet18(weights=None)  # Changed to resnet18
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_weights_path))
    model = model.to(device)
    model.eval()
    return model

# 3. Evaluation for Testing
def evaluate(model, test_iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, labels, _) in enumerate(test_iterator):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # Calculate overall accuracy
    accuracy = 100. * correct / total
    avg_loss = epoch_loss / len(test_iterator)
    print(f'\nTesting completed. Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%\n')
    return avg_loss, accuracy

# 4. Prediction and Label Gathering
def get_all_predictions(model, iterator, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_filenames = []  # to store the filenames

    with torch.no_grad():
        for data, labels, paths in iterator:
            data = data.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            # Retrieving file paths
            all_filenames.extend(paths)
    return all_preds, all_labels, all_filenames

# 5. Confusion Matrix Creation and Visualization
def plot_and_save_confusion_matrix(true_labels, predictions, classes, save_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    conf_mat = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()  # Close the plot to free memory
    print(f"Confusion matrix saved to {save_path}")

# Main Execution
def main():
    TEST_DIR = '/home/pola/02_handwritten_document/Student2-Ajit/more_hw_languages/English_bengali/D3/test'
    MODEL_WEIGHTS_PATH = "/home/pola/02_handwritten_document/Student2-Ajit/more_hw_languages/English_bengali/model_weights_onD5/196576_33310_14_weights.pt"
    directory, filename = os.path.split(MODEL_WEIGHTS_PATH)
    RESULTS_DIR = '/home/pola/02_handwritten_document/Student2-Ajit/more_hw_languages/English_bengali/results/'
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    train_classes, train_class_lengths = get_class_info(TEST_DIR)
    max_class_name_length = max(len(name) for name in train_classes)

    print(f"{'Class Name'.ljust(max_class_name_length)} | {'Test Images'.ljust(16)} ")
    print('-' * (max_class_name_length + 36))
    for class_name in sorted(set(train_classes)):
        train_count = train_class_lengths.get(class_name, 0)
        print(f"{class_name.ljust(max_class_name_length)} | {str(train_count).ljust(16)} ")
    print(f"\nTotal images in Training Dataset: {sum(train_class_lengths.values())}")
    classes = sorted(os.listdir(TEST_DIR))

    # Initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()

    # Load Data
    test_loader, classes, num_test_samples = load_data(TEST_DIR)

    # Load Model
    model = load_model(MODEL_WEIGHTS_PATH, len(classes), device)
    print("Loaded model with classes:", classes)

    # Evaluate Model
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    # Plot and Save Confusion Matrix
    save_path = './results'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    all_preds, all_labels, all_filenames = get_all_predictions(model, test_loader, device)

    # Dynamic filename construction
    filename_details = f"{filename}_{num_test_samples}_smpls_{test_acc:.2f}_pct_acc_{len(classes)}_classes"
    csv_filename = f"{RESULTS_DIR}/pred_{filename_details}.csv"
    confusion_matrix_filename = f"{RESULTS_DIR}conf_{filename_details}_.png"

    # Save all details to a CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Filename', 'True Label', 'Predicted Label'])
        for filename, true_label, predicted_label in zip(all_filenames, all_labels, all_preds):
            csvwriter.writerow([filename, classes[true_label], classes[predicted_label]])

    print(f"All prediction details saved in CSV: {csv_filename}")
    plot_and_save_confusion_matrix(all_labels, all_preds, classes, confusion_matrix_filename)

if __name__ == "__main__":
    main()
