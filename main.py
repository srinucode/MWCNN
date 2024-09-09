import torch
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torch.nn.functional as F
import pywt
import numpy as np
import torchvision.models as models

import matplotlib.pyplot as plt
from ImageDataset import ImageDataset
from MWCNN import MWCNN
from Teacher import Teacher

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# # Load the original CSV file
# df = pd.read_csv('train/train_labels.csv')

# # Select 50 random entries
# df_sampled = df.sample(n=50)       #, random_state=42)

# # Save the new CSV file
# df_sampled.to_csv('train/sample_labels.csv', index=False)

# print("New file with 50 entries has been created.")


train_dataset = ImageDataset(csv_file='train/train_labels.csv', root_dir='train', transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)

# # Load the original CSV file
# df = pd.read_csv('test/test_labels.csv')

# # Select 50 random entries
# df_sampled = df.sample(n=50)      #, random_state=42)

# # Save the new CSV file
# df_sampled.to_csv('test/sample_labels.csv', index=False)

# print("New file with 50 entries has been created.")



test_dataset = ImageDataset(csv_file='test/test_labels.csv', root_dir='test', transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=4)

def train(model, train_loader, epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        for data, target in train_loader:
            # inputs: A collection of batch_size images
            # labels: A vector of dimensionality batch_size with integers denoting class of each image
            inputs, labels = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            # outputs: Output of the network for the collection of images. A tensor of dimensionality batch_size x num_classes
            # labels: The actual labels of the images. Vector of dimensionality batch_size
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss}")
        losses.append(running_loss)
        # Iterate through each named parameter (weights) in the model
        k = 0
        # for name, param in model.named_parameters():
        #     print(f"Layer: {name} | Weights: {param.data}")
        #     k = k + 1
        #     if k == 1:
        #         break
    return losses



def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode
    losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # Forward pass with the student model
            student_logits = student(inputs)

            #Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
        losses.append(running_loss)
        
    return losses

def test(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy



# Initialize the ResNet18 model (pretrained on ImageNet)
resnet18_model = models.resnet18(pretrained=False)

# Modify the final layer of ResNet18 for the desired number of classes (assuming 10 classes)
resnet18_model.fc = nn.Linear(resnet18_model.fc.in_features, 4)

#torch.manual_seed(42)
# Set the device to the desired GPU (e.g., GPU 0)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
teacher = Teacher(resnet18_model).to(device)
student = MWCNN(in_channels=3, out_channels=64, wavelet='haar').to(device)



# nn_deep = DeepNN(num_classes=10).to(device)
teacher_losses = train(teacher, train_dataloader, epochs=10, learning_rate=0.001, device=device)
test_accuracy_teacher = test(teacher, test_dataloader, device)



# Apply ``train_knowledge_distillation`` with a temperature of 2. Arbitrarily set the weights to 0.75 for CE and 0.25 for distillation loss.
losses_student_KD = train_knowledge_distillation(teacher=teacher, student=student, train_loader=train_dataloader, epochs=10, learning_rate=0.05, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75, device=device)
test_accuracy_student_ce_and_kd = test(student, test_dataloader, device)


# Creating a dictionary
data = {
    'Variable': ['test_accuracy_teacher', 'test_accuracy_student_ce_and_kd'],
    'Value': [test_accuracy_teacher, test_accuracy_student_ce_and_kd]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Saving to CSV
filename = 'accuracies.csv'
df.to_csv(filename, index=False)

print(f'Variables saved to {filename}')






# Step 1: Plot teacher losses
plt.plot(teacher_losses, marker='o', linestyle='-', color='b', label='Teacher Loss')

# Step 2: Plot student KD losses
plt.plot(losses_student_KD, marker='x', linestyle='--', color='r', label='Student KD Loss')

# Step 3: Add titles and labels
plt.title('Teacher vs Student KD Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Step 4: Add a legend
plt.legend()

# Step 5: Save the plot to the current folder
plt.savefig('teacher_vs_student_KD_loss_plot.png')

# Step 6: Show the plot (optional)
plt.show()
