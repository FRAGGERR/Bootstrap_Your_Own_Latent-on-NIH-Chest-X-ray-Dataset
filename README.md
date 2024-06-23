# Bootstrap_Your_Own_Latent-on-NIH-Chest-X-ray-Dataset
Hey Everyone, In this repo I'm sharing my Code on BootStrap your own Latent using ResNet - 18 as a Classifier on BYOL taking NIH Chest X-ray Dataset. Here in this code I added on Dataloader file also. If you want to run the code you've to config the dataloader file to the BYOL.py file. And from my code you can also take reference. 

---

# BYOL with ResNet18 for Multi-Class Classification

This repository contains the implementation of a BYOL (Bootstrap Your Own Latent) model using ResNet18 as the backbone. The model is designed for multi-class classification on the NIH Chest X-ray dataset. The project demonstrates self-supervised learning followed by supervised fine-tuning for classification tasks.

## Repository Contents

- **Model Architecture**: Implementation of a BYOL model with a ResNet18 encoder.
- **Data Loading**: Scripts to load and preprocess the NIH Chest X-ray dataset.
- **Transformations**: Data augmentations and transformations applied during training.
- **Training and Validation**: Training loops for both self-supervised pre-training and supervised fine-tuning, including logging of losses and ROC AUC scores.
- **Model Saving and Evaluation**: Saving the trained model and evaluating its performance using ROC AUC scores.

## Repository Structure

- `byol_training.py`: Main script for training and evaluating the BYOL model.
- `config.yaml`: Configuration file containing hyperparameters and data paths.
- `data_loader.py`: Script for loading the NIH Chest X-ray dataset.
- `training.log`: Log file containing training and validation information.
- `byol_model.pth`: Saved weights of the trained BYOL model.

## How to Use

1. **Setup Environment**:
   - Ensure you have Python and necessary libraries installed. You can set up a virtual environment and install the required packages using:
     ```bash
     pip install torch torchvision tqdm pyyaml scikit-learn matplotlib
     ```

2. **Prepare Data**:
   - Make sure the NIH Chest X-ray dataset is available and correctly referenced in `config.yaml`.

3. **Run the Training Script**:
   - To pre-train the BYOL model and fine-tune it for classification, run:
     ```bash
     python byol_training.py
     ```

4. **Evaluate the Model**:
   - After training, you can evaluate the model's performance using the provided ROC AUC scores and visualize them:
     ```python
     import torch
     import matplotlib.pyplot as plt
     import numpy as np

     # Load the saved model
     byol_model = BYOL(base_encoder, hidden_dim=4096, projection_dim=256, num_classes=num_classes).to(device)
     byol_model.load_state_dict(torch.load("byol_model.pth"))

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
     ```

## Model Architecture

The model consists of:
- **Base Encoder**: ResNet18 pretrained on ImageNet.
- **Online and Target Encoders**: Two identical networks for the online and target encoders.
- **Projection and Prediction Heads**: MLPs for projecting the encoded features and predicting the target projections.
- **Classifier**: A simple linear classifier for downstream classification tasks.

## Training Process

1. **Self-Supervised Pre-Training**:
   - Train the BYOL model with self-supervised learning using two augmented views of each image.
   - Update the target network parameters with a moving average of the online network parameters.

2. **Supervised Fine-Tuning**:
   - Train the model with labeled data to perform multi-class classification.
   - Monitor and log the classification loss and ROC AUC scores during training.

## Results

The model's performance is evaluated using ROC AUC scores for each class. The training process is logged, and the final results are visualized using ROC AUC plots.

## Contributing

Feel free to submit issues or pull requests if you have any improvements or bug fixes.

## License

This project is licensed under the MIT License.

---

This description provides a comprehensive overview on my project, including its functionality, how to use it, and details about the model architecture and training process. Adjust the details as needed based on your specific implementation and results.
