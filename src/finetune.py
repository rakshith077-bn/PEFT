import argparse
import logging
import torch
import typer
from typing_extensions import Annotated
from rich import print
import progress.bar as bar

import torch.utils
from torch.utils.data import DataLoader, Subset
import torch.utils.data
import torch.nn as nn

from lion_pytorch import Lion
from transformers import ViTForImageClassification, Trainer, TrainingArguments

from peft import LoraConfig, get_peft_model

from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold
import numpy as np

import sys
import os
from tqdm import tqdm
import datetime, time

from data import data_load
from load_dataset import ImageDataset
from model import prepare_model
from header import data_load, welcome_note, general_info

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

    return dataset
    return train_loader, test_loader

def train_model(model, train_loader, device, num_epochs):

    optimizer = nn.Linear(10, 1)
    opt = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2) # LIONs Optimizer

    criterion = torch.nn.CrossEntropyLoss()

    width = os.get_terminal_size().columns - 10
    with bar.Bar('Epochs', max=num_epochs, max_width=width) as epoch_bar:
        for epoch in range(num_epochs):

            model.train()
            epoch_start = time.time()
            logging.info(f"Starting Epoch {epoch + 1}/{num_epochs}")
            train_loss = 0

            with bar.Bar(f'Epoch {epoch + 1}', max=len(train_loader)) as batch_bar:
                for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):

                    images, labels = images.to(device), labels.to(device)
                    opt.step()
                    opt.zero_grad()
                    outputs = model(pixel_values=images).logits
                    loss = criterion(outputs, labels)
                    loss.backward()
                    opt.step()
                    train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch + 1} completed.")
            print(f"Training Loss: {avg_train_loss:.4f}")
            print(f"Time taken: {time.time() - epoch_start:.2f}s.\n")

def evaluate_model(model, dataset, device):
    k=4
    batch_size=batch_size

    kfold = KFold(n_splits=k, shuffle=True)
    criterion = torch.nm.CrossEntropyLoss()
    total_loss = 0

    for fold, (train, val) in enumerate(kfold.split(dataset)):
        print("\n Evaluating...\n")

        val_subset = Subset(dataset, val)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model = prepare_model(device)
        model.eval()

        fold_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(pixel_values=images).logits
                loss = criterion(outputs, labels)
                fold_loss += loss.item()

        avg_fold_loss = fold_loss / len(val_loader)
        total_loss += avg_fold_loss

        print(f"Fold {fold + 1} Loss: {avg_fold_loss:.4f}")

    avg_test_loss = total_loss / k
    print(f"\n Avg Loss: {avg_fold_loss:.4f} \n")


def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)
    print(f"Saved Model: {path}")

def main(
    data_path: Annotated[str, typer.Option(help="Path to the dataset file (str). Your dataset has to be structured in the manner specified for training. Check README.md.")],
    num_epochs: Annotated[int, typer.Option(help="Number of training epochs (int). Default is set to 10.")] = 10,
    batch_size: Annotated[int, typer.Option(help="Batch size (int). Default is set to 16.")] = 16,
    ):

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu") 
    print(f"Using Device : {device}")

    try:
        if True:
            train_loader, test_loader, train_dataset, test_dataset = load_data(data_path, batch_size)
            model = prepare_model(device) # Load model into GPU
            data_load(device)
    except Exception as e:
        print("Dataset did not load successfully")
        print(e)

    if torch.cuda.is_available():
        torch.cuda.empty_cache() # Clear Cache

    train_model(model, train_loader, device, num_epochs)
    evaluate_model(model, test_loader, device)
    save_model(model)

if __name__ == '__main__':
    welcome_note()
    typer.run(main)
