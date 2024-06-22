# Import necessary libraries
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
import logging
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_pil_image
import torch
torch.cuda.empty_cache()

# Initialize logging
logging.basicConfig(filename='training.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Device configuration
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Load configuration
config_file = "config.yaml"
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)
config['data_pct'] = 100

# Data loading
sys.path.append('data/data_loader.py')
from data import DataLoader
data_ins = DataLoader(config)
train_loader, valid_loader, test_loader = data_ins.GetNihDataset()

#ResNet18-based model with BYOL
class BYOL(nn.Module):
    def __init__(self, base_encoder, hidden_dim=4096, projection_dim=256, num_classes=15, moving_average_decay=0.99):
        super(BYOL, self).__init__()
        self.base_encoder = base_encoder
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        #the output size from base_encoder
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224).to(device)
            output_size = self.base_encoder(dummy_input).view(1, -1).size(1)

        self.online_encoder = nn.Sequential(
            self.base_encoder,
            nn.Linear(output_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

        self.target_encoder = nn.Sequential(
            self.base_encoder,
            nn.Linear(output_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

        for param_online, param_target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_target.data.copy_(param_online.data)#weight copying
            param_target.requires_grad = False #Freeze Target Encoder

        self.moving_average_decay = moving_average_decay

        self.classifier = nn.Sequential(
            nn.Linear(output_size, num_classes),
            nn.Sigmoid()
        )

    # Downstream Task
    def forward(self, x1, x2=None):
        if x2 is None:
            return self.classifier(self.base_encoder(x1))

        online_proj_one = self.online_encoder(x1) #x1:first augmented view of img.
        online_proj_two = self.online_encoder(x2) #x2:second augmented view of img.
        target_proj_one = self.target_encoder(x1).detach() #to stop gradients
        target_proj_two = self.target_encoder(x2).detach()
        return online_proj_one, online_proj_two, target_proj_one, target_proj_two

    def update_target_network(self):
        for param_online, param_target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_target.data = self.moving_average_decay * param_target.data + (1 - self.moving_average_decay) * param_online.data

#BYOL loss function
def byol_loss(p1, p2, z1, z2):
    loss_one = 2 - 2 * (p1 * z2.detach()).sum(dim=-1)
    loss_two = 2 - 2 * (p2 * z1.detach()).sum(dim=-1)
    return (loss_one + loss_two).mean()

#transformations for BYOL
byol_transforms = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(30),
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model initialization
num_classes = 15
base_encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
byol_model = BYOL(base_encoder, hidden_dim=4096, projection_dim=256, num_classes=num_classes).to(device)

# Training and validation setup
num_epochs = 10
learning_rate = 0.001
optimizer = torch.optim.Adam(byol_model.parameters(), lr=learning_rate)
criterion = byol_loss
classification_criterion = nn.BCELoss()

# BYOL Pre-training
total_start_time = time.time()
roc_auc_scores = []

for epoch in range(num_epochs):
    byol_model.train()
    epoch_loss = 0
    for images, _ in tqdm(train_loader):
        images = images.to(device)
        
        # Convert each tensor image to a PIL image, apply transformations, and stack them back into a tensor
        images_transformed = torch.stack([byol_transforms(to_pil_image(img.cpu())) for img in images]).to(device)
        
        optimizer.zero_grad()
        
        online_proj_one, online_proj_two, target_proj_one, target_proj_two = byol_model(images, images_transformed)
        loss = criterion(online_proj_one, online_proj_two, target_proj_one, target_proj_two)
        
        loss.backward()
        optimizer.step()
        
        byol_model.update_target_network()
        
        epoch_loss += loss.item()
    
    logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

# Training
for epoch in range(num_epochs):
    byol_model.train()
    epoch_loss = 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = byol_model(images)
        loss = classification_criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    logging.info(f"Epoch [{epoch+1}/{num_epochs}], Classification Loss: {epoch_loss/len(train_loader):.4f}")

    # Validation
    byol_model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in tqdm(valid_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = byol_model(images) # Performs a forward pass
            all_labels.append(labels.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())
    
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    roc_auc = roc_auc_score(all_labels, all_preds, average=None)
    roc_auc_scores.append(roc_auc)
    
    logging.info(f"Epoch [{epoch+1}/{num_epochs}], Validation ROC AUC: {roc_auc}")

total_end_time = time.time()
total_duration = total_end_time - total_start_time
logging.info(f"Total Training Time: {total_duration:.2f} seconds")

# Save the trained model
torch.save(byol_model.state_dict(), "byol_model.pth")

# Plot ROC AUC scores
plt.figure(figsize=(10, 8))
for i in range(num_classes):
    plt.plot([roc_auc[i] for roc_auc in roc_auc_scores], label=f'Class {i}')
plt.xlabel('Epoch')
plt.ylabel('ROC AUC')
plt.title('ROC AUC Scores per Epoch')
plt.legend()
plt.grid(True)
plt.show()
