# ┌────────────────────────────────────────────────────────────────────────────── ┐
# │ Citation                                                                      │
# ├────────────────────────────────────────────────────────────────────────────── ┤
# │ @misc{rakshith2024customPEFTscript,                                           │
# │   author = {Rakshith B N},                                                    │
# │   title = {Parameter-Efficient Fine-Tuning of Vision Transformer for Image    │
# │            Classification},                                                   │
# │   year = {2024},                                                              │
# │   howpublished = {\url{https://github.com/rakshith077-bn/MLD-Fintune}},       │
# │                                                                               │
# │ }                                                                             │
# └────────────────────────────────────────────────────────────────────────────── ┘
#

import argparse
import logging
import torch
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
from transformers import ViTForImageClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from utilities import transformations
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

import os
from tqdm import tqdm
import datetime, time

from load_dataset import ImageDataset


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(data_path: str, batch_size: int):
    transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageDataset(root_dir=data_path, transform=transformations)
    
    # Stratified Sampling
    labels = np.array(dataset.labels)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_indices, test_indices in splitter.split(np.zeros(len(labels)), labels):
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def prepare_model(device):
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', torch_dtype=torch.float16, device_map='auto')
    model = model.to(device)
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=['encoder.layer.11.attention.attention.query', 'encoder.layer.11.attention.attention.value'],
        inference_mode=False
    )
    return get_peft_model(model, config)

def train_model(model, train_loader, device, num_epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_start = time.time()
        logging.info(f"Starting Epoch {epoch + 1}/{num_epochs}")
        train_loss = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        logging.info(f"Epoch {epoch + 1} completed. Training Loss: {avg_train_loss:.4f} - Time: {time.time() - epoch_start:.2f}s")

def evaluate_model(model, test_loader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    logging.info(f"Test Loss: {avg_test_loss:.4f}")

def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)
    logging.info(f"Saved Model: {path}")

def main():
    setup_logging()
    
    # parser = argparese.ArgumentParser(description='')
    parser = argparse.ArgumentParser(description='Fine-tune Vision Transformer model for Image classification. Fine tune your image dataset with a few commands.')

    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset file (str). Your dataset has to be structured in the manner specified for training. Check README.md.')
    
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs (int). Default is set to 10.')
    
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (int). Default is set to 16.')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu")
    logging.info(f"Using device: {device}")
    
    if True:
        train_loader, test_loader = load_data(args.data_path, args.batch_size)
        model = prepare_model(device)
        logging.info("DatasetLoaded")
    
    if torch.cuda.is_available(): 
        torch.cuda.empty_cache() # Clear Cache
    
    train_model(model, train_loader, device, args.num_epochs)
    evaluate_model(model, test_loader, device)
    save_model(model)

if __name__ == '__main__':
    main()
