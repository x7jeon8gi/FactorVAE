import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
import datetime
import random
from tqdm.auto import tqdm
import argparse
from Layers import FactorVAE, FeatureExtractor, FactorDecoder, FactorEncoder, FactorPredictor, AlphaLayer, BetaLayer
from stockdata import StockDataset
from train_model import train, validate, test
import wandb

parser = argparse.ArgumentParser(description='Train a FactorVAE model on stock data')
parser.add_argument('--train_data', type=str, default='train_data.csv', help='path to training data')
parser.add_argument('--val_data', type=str, default='val_data.csv', help='path to validation data')
parser.add_argument('--test_data', type=str, default='test_data.csv', help='path to test data')
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
parser.add_argument('--batch_size', type=int, default=300, help='batch size')
parser.add_argument('--num_latent', type=int, default=20, help='number of latent variables')
parser.add_argument('--seq_len', type=int, default=20, help='sequence length')
parser.add_argument('--num_factor', type=int, default=8, help='number of factors')
parser.add_argument('--hidden_size', type=int, default=20, help='hidden size')
args = parser.parse_args()


train_df = pd.read_pickle("./train_csi300.pkl")
valid_df = pd.read_pickle("./valid_csi300.pkl")
test_df = pd.read_pickle("./test_csi300.pkl")  # read pickles for valid and tes


def main():
    
    feature_extractor = FeatureExtractor(num_latent=args.num_latent, hidden_size=args.hidden_size)
    factor_encoder = FactorEncoder(num_factors=args.num_factor, num_portfolio=args.num_latent, hidden_size=args.hidden_size)
    alpha_layer = AlphaLayer(args.hidden_size)
    beta_layer = BetaLayer(args.hidden_size, args.num_factor)
    factor_decoder = FactorDecoder(alpha_layer, beta_layer)
    factor_predictor = FactorPredictor(args.batch_size, args.hidden_size, args.num_factor)
    factorVAE = FactorVAE(feature_extractor, factor_encoder, factor_decoder, factor_predictor)
    
    train_ds = StockDataset(train_df)
    valid_ds = StockDataset(valid_df)
    test_ds = StockDataset(test_df)
    
    # Assuming you want to create a mini-batch of size 300
    train_dataloader = DataLoader(train_ds, batch_size=300, shuffle=False)
    valid_dataloader = DataLoader(valid_ds, batch_size=300, shuffle=False)
    test_dataloader = DataLoader(test_ds, batch_size=300, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project="FactorVAE", name="replicate")
    factorVAE.to(device)
    best_val_loss = float('inf')
    optimizer = torch.optim.Adam(factorVAE.parameters(), lr=args.lr)
    for epoch in range(args.num_epochs):
        train_loss = train(factorVAE, train_dataloader, optimizer)
        val_loss = validate(factorVAE, valid_dataloader)
        test_loss = test(factorVAE, test_dataloader)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Validation Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(factorVAE.state_dict(), 'best_model.pt')
    wandb.finish()
    
if __name__ == '__main__':
    main()
