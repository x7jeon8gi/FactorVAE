import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import numpy as np
import wandb
from tqdm.auto import tqdm

def train(factor_model, dataloader, optimizer, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factor_model.to(device)
    factor_model.train()
    total_loss = 0
    with tqdm(total=len(dataloader)-args.seq_len+1) as pbar:
        for char, returns in dataloader:
            if char.shape[1] != args.seq_len:
                continue
            inputs = char.to(device)
            labels = returns[:,-1].reshape(-1,1).to(device)
            inputs = inputs.float()
            labels = labels.float()
            
            optimizer.zero_grad()
            loss, reconstruction, factor_mu, factor_sigma, pred_mu, pred_sigma = factor_model(inputs, labels)
            total_loss += loss.item() * inputs.size(0)
            loss.backward()
            optimizer.step()
            pbar.update(1)
        # print(loss)
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


@torch.no_grad()
def validate(factor_model, dataloader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factor_model.to(device)
    factor_model.eval()
    total_loss = 0
    with tqdm(total=len(dataloader)-args.seq_len+1) as pbar:
        for char, returns in dataloader:
            if char.shape[1] != args.seq_len:
                continue
            inputs = char.to(device)
            labels = returns[:,-1].reshape(-1,1).to(device)
            inputs = inputs.float()
            labels = labels.float()
            
            loss, reconstruction, factor_mu, factor_sigma, pred_mu, pred_sigma = factor_model(inputs, labels)
            total_loss += loss.item() * inputs.size(0)
            pbar.update(1)
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

@torch.no_grad()
def test(factor_model, dataloader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factor_model.to(device)
    factor_model.eval()
    total_loss = 0
    with tqdm(total=len(dataloader)-args.seq_len+1) as pbar:
        for char, returns in dataloader:
            if char.shape[1] != args.seq_len:
                continue
            inputs = char.to(device)
            labels = returns[:,-1].reshape(-1,1).to(device)
            inputs = inputs.float()
            labels = labels.float()
            
            loss, reconstruction, factor_mu, factor_sigma, pred_mu, pred_sigma = factor_model(inputs, labels)
            total_loss += loss.item() * inputs.size(0)
            pbar.update(1)
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

def run(factor_model, train_loader, val_loader, test_loader, lr, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # wandb.init(project="FactorVAE", name="replicate")
    factor_model.to(device)
    best_val_loss = float('inf')
    optimizer = torch.optim.AdamW(factor_model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        train_loss = train(factor_model, train_loader, optimizer)
        val_loss = validate(factor_model, val_loader)
        test_loss = test(factor_model, test_loader)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Validation Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(factor_model.state_dict(), 'best_model.pt')
