import os
import random
import zipfile
import requests
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from timeit import default_timer as timer
from tqdm.auto import tqdm
import torch
from torch import nn
import torchvision
from torchvision.transforms import v2
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold


def walk_through_dir(dir_path):
  """Walks through dir_path returning its contents."""
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

# Use ImageFolder to create dataset(s)
def ImageFolder_data(transform, dir):
  ImageFolder = datasets.ImageFolder(root=dir, transform=transform)

  return ImageFolder

def create_dataloader(dataset, indices, BATCH_SIZE=32, NUM_WORKERS=None):
  if NUM_WORKERS is None:
    NUM_WORKERS = torch.cuda.device_count() if torch.cuda.is_available() else os.cpu_count()

  subset = Subset(dataset, indices)
  dataloader = DataLoader(dataset=subset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
  return dataloader


def zero_one_loss_fn(output, target):
  loss = (output != target).to(torch.float32).mean()
  return loss


def SaveModel(model, name):
  F = MODEL_PATH/name
  torch.save(obj=model, f=f"{F}.pt")
  print(f"Saving: {name}")


def loadModel(name):
  F = MODEL_PATH/name
  model = torch.load(f=f"{F}.pt")
  model.to(device)
  model.eval()
  print(f"Loading: {name}")
  return model

# 1. Create a train function that takes in various model parameters + optimizer + dataloaders + loss function and measures the BCE
def train_fn(model,
             train_dataloader,
             test_dataloader,
             fold_num,
             # model_type,
             optimizer_type,
             # learning_rate: List,
             num_epochs = 40,
             writer = None,
             loss_fn = nn.BCEWithLogitsLoss(),
             device=device):

  # 2. Create empty results dictionary
  results = {"train_loss": [],
             "train_ZOL": [],
             "train_accuracy": [],
             "test_loss": [],
             "test_ZOL": [],
             "test_accuracy": []}

  # 3. Train the model
  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(num_epochs)):

    # Put the model in train mode
    model.train()

    # Setup some variables
    train_acc, train_loss, train_ZOL = 0, 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(train_dataloader):

      # Send data to the target device
      X, y = X.to(device), y.to(device).to(torch.float64)

      # 1. Forward pass
      y_logits = model(X).squeeze().to(torch.float64)
      y_pred_class = torch.round(torch.sigmoid(y_logits)) # We need to apply sigmoid and then round to convert logits to 0/1

      # 2. Calculate the loss
      loss = loss_fn(y_logits, y)
      loss_zero_one = zero_one_loss_fn(y_pred_class, y)

      train_loss += loss
      train_ZOL += loss_zero_one

      train_acc += (y_pred_class==y).sum().item()/len(y_pred_class)

      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # 4. Loss backward
      loss.backward()

      # 5. Optimizer step
      optimizer.step()

    # Adjust metrics to get average loss and accuracy per batch
    train_acc = (train_acc / len(train_dataloader)) * 100
    train_loss /= len(train_dataloader)
    train_ZOL /= len(train_dataloader)             # Agar jame in o accuracy 1 nemishod * 32 bokon age baz nashod beja 32, 148 bezar

    # Print out what's happening
    print(f'Train:\nEpoch: ({epoch+1}) | Accuracy: {train_acc:.3f}% | Zero one Loss: {train_ZOL.item():.4f} | Entropy Loss: {train_loss.item():.4f}')

    # Update results dictionary
    results["train_loss"].append(train_loss.item())
    results["train_ZOL"].append(train_ZOL.item())
    results["train_accuracy"].append(train_acc)

    # Test steps
    # Put model in eval mode
    model.eval()
    test_acc, test_loss, test_ZOL = 0, 0, 0

    with torch.inference_mode():
      # Loop through DataLoader batches
      for batch, (X, y) in enumerate(test_dataloader):
        # Send data to the target device
        X, y = X.to(device), y.to(device).to(torch.float64)

        # 1. Forward pass
        test_pred_logits = model(X).squeeze().to(torch.float64)
        test_pred_labels = torch.round(torch.sigmoid(test_pred_logits))

        # 2. Calculate the loss
        test_loss += loss_fn(test_pred_logits, y)
        test_ZOL += zero_one_loss_fn(test_pred_labels, y)

        test_acc += (test_pred_labels==y).sum().item()/len(test_pred_labels)

      # Adjust metrics
      test_acc = (test_acc / len(test_dataloader)) * 100
      test_loss /= len(test_dataloader)
      test_ZOL /= len(test_dataloader)

      print(f'Test:\nEpoch: ({epoch+1}) | Accuracy:: {test_acc:.3f}% | Zero one Loss: {test_ZOL.item():.4f} | Entropy Loss: {test_loss.item():.4f}')

      # Update results dictionary
      results["test_loss"].append(test_loss.item())
      results["test_ZOL"].append(test_ZOL.item())
      results["test_accuracy"].append(test_acc)

    if writer:
      writer.add_scalars(f'{model_type}/{optimizer_type}LR_{learning_rate}/Fold{fold_num}/Loss', {'train':train_loss, 'test':test_loss} , epoch)
      writer.add_scalars(f'{model_type}/{optimizer_type}/LR_{learning_rate}/Fold{fold_num}/ZeroOneLoss', {'train':train_ZOL, 'test':test_ZOL} , epoch)
      writer.add_scalars(f'{model_type}/{optimizer_type}/LR_{learning_rate}/Fold{fold_num}/Accuracy', {'train':train_acc, 'test':test_acc} , epoch)

  return model, results
