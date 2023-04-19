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

class StockDataset(torch.utils.data.Dataset):
    def __init__(self, df, num_stock, sequence_length):
        self.df = df
        self.num_stock = num_stock
        self.sequence_length = sequence_length
        
        self.groups = df.groupby('datetime')
        self.rows_to_keep = []

        for _, group in self.groups:
            # drop row if their LABEL0 is NaN
            group = group.dropna(subset=['LABEL0'], axis=0)

            #? 도대체 수익률이 전혀 없는 날짜 무엇?
            if len(group) == 0:
                continue
            
            elif len(group) < num_stock:
                new_group = pd.DataFrame(np.nan, index=np.arange(num_stock), columns=df.columns)
                new_group.iloc[:len(group), :] = group.values
                
                # create MultiIndex with stock name and date
                new_stock_names = pd.Index(['empty' for i in range(1, num_stock-len(group)+1)])
                                
                dates = group.index.get_level_values(0).unique()
                new_index = pd.MultiIndex.from_product([dates, new_stock_names])
                merged_index = group.index.append(new_index)
                new_group = new_group.set_index(merged_index)
                
                # sort by LABEL0
                new_group = new_group.sort_values(by='LABEL0', ascending=False)
                new_group = new_group.fillna(0)

                self.rows_to_keep.append(new_group)
                
                assert len(new_group) == num_stock, 'new_group should have num_stock number of rows'
            else:
                self.rows_to_keep.append(group)
                
        self.df = pd.concat(self.rows_to_keep) # , ignore_index=True)
        
        assert self.df['LABEL0'].isnull().sum() == 0, 'LABEL0 column should not contain any NaN values'
        
        #self.input_size = sequence_length * len(self.df.columns)  # calculate input size based on 30 days of data

    def __len__(self):
        return len(self.df) - self.sequence_length +1  # subtract 30 to account for accumulation of 30 days of data

    def __getitem__(self, idx):
        input_data = []
        label_data = []
        
        idx_list = [300*i + idx for i in range(self.sequence_length) if 300*i + idx < len(self.df)]

        data = self.df.iloc[idx_list, :-1].values #(seq_len, character)
        label = self.df.iloc[idx_list, -1].values #(seq_len, 1)
        
        input_data.append(data)
        input_data = np.concatenate(input_data, axis=0)
        label_data.append(label)        
        
        # label 도출 값은 반드시 잘라서 써야함.
        return input_data, label
