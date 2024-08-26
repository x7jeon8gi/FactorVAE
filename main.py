import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
import argparse
from module import FactorVAE, FeatureExtractor, FactorDecoder, FactorEncoder, FactorPredictor, AlphaLayer, BetaLayer
from dataset import init_data_loader
from train_model import train, validate
from utils import set_seed, DataArgument
import wandb


def main(args, data_args):
    
    set_seed(args.seed)
    # make directory to save model
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # create model
    feature_extractor = FeatureExtractor(num_latent=args.num_latent, hidden_size=args.hidden_size)
    factor_encoder = FactorEncoder(num_factors=args.num_factor, num_portfolio=args.num_portfolio, hidden_size=args.hidden_size)
    alpha_layer = AlphaLayer(args.hidden_size)
    beta_layer = BetaLayer(args.hidden_size, args.num_factor)
    factor_decoder = FactorDecoder(alpha_layer, beta_layer)
    factor_predictor = FactorPredictor(args.hidden_size, args.num_factor)
    factorVAE = FactorVAE(feature_extractor, factor_encoder, factor_decoder, factor_predictor)
    
    # create dataloaders
    dataset = pd.read_pickle(args.dataset).iloc[:, :159] # market info 제외
    dataset.rename(columns={dataset.columns[-1]: 'LABEL0'}, inplace=True) # 마지막 컬럼 이름 변경 'LABEL0'
    train_dataloader = init_data_loader(dataset,
                                        shuffle=True,
                                        step_len=data_args.seq_len, 
                                        start=data_args.start_time,
                                        end=data_args.fit_end_time, 
                                        select_feature=data_args.select_feature)
    
    valid_dataloader = init_data_loader(dataset,
                                        shuffle=False, 
                                        step_len=data_args.seq_len, 
                                        start=data_args.val_start_time, 
                                        end=data_args.val_end_time, 
                                        select_feature=data_args.select_feature)
    
    T_max = len(train_dataloader) * args.num_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"*************** Using {device} ***************")
    args.device = device
        
    factorVAE.to(device)
    best_val_loss = 10000.0
    optimizer = torch.optim.Adam(factorVAE.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    if args.wandb:
        wandb.init(project="FactorVAE", config=args, name=f"{args.run_name}")
        wandb.config.update(args)

    # Start Trainig
    for epoch in tqdm(range(args.num_epochs)):
        train_loss = train(factorVAE, train_dataloader, optimizer, scheduler, args)
        val_loss = validate(factorVAE, valid_dataloader, args)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}") 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            #? save model in save_dir
            
            #? torch.save
            save_root = os.path.join(args.save_dir, f'{args.run_name}_factor_{args.num_factor}_hdn_{args.hidden_size}_port_{args.num_portfolio}_seed_{args.seed}.pt')
            torch.save(factorVAE.state_dict(), save_root)
            print(f"Model saved at {save_root}")
            
        if args.wandb:
            wandb.log({"Train Loss": train_loss, "Validation Loss": val_loss, "Learning Rate": scheduler.get_last_lr()[0]})
    
    if args.wandb:
        wandb.log({"Best Validation Loss": best_val_loss})
        wandb.finish()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a FactorVAE model on stock data')

    parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')

    parser.add_argument('--num_latent', type=int, default=158, help='number of variables')
    parser.add_argument('--num_portfolio', type=int, default=128, help='number of stocks')

    parser.add_argument('--seq_len', type=int, default=20, help='sequence length')
    parser.add_argument('--num_factor', type=int, default=96, help='number of factors')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')

    parser.add_argument('--dataset', type=str, default='./data/csi_data.pkl', help='dataset to use')
    parser.add_argument('--start_time', type=str, default='2009-01-01', help='start time')
    parser.add_argument('--fit_end_time', type=str, default='2017-12-31', help='fit end time')
    parser.add_argument('--val_start_time', type=str, default='2018-01-01', help='validation start time')
    parser.add_argument('--val_end_time', type=str, default='2018-12-31', help='validation end time')
    parser.add_argument('--end_time', type=str, default='2020-12-31', help='end time')

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--run_name', type=str, default='VAE-Revision2', help='name of the run')
    parser.add_argument('--save_dir', type=str, default='./best_models', help='directory to save model')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--wandb', action='store_true', help='whether to use wandb')
    args = parser.parse_args()

    data_args = DataArgument(
        start_time=args.start_time,
        end_time=args.end_time,
        fit_end_time=args.fit_end_time,
        val_start_time=args.val_start_time,
        val_end_time=args.val_end_time,
        seq_len=args.seq_len,
    )

    main(args, data_args)
