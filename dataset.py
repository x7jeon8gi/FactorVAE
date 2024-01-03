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
        
        self.df = self.df.dropna(subset=['LABEL0'])

        self.df = self.df.groupby('datetime').apply(lambda group: self._adjust_group(group, num_stock))
        self.df.index = self.df.index.droplevel(0)        
        assert self.df['LABEL0'].isnull().sum() == 0, 'LABEL0 column should not contain any NaN values'

        self.instrument_groups = df.groupby('instrument')
        self.group_indices = []
        for name, group in self.instrument_groups:
            # The index of the group will be a consecutive value starting at 0 and ending at the length of the group.
            indices = [(name, i) for i in range(len(group) - sequence_length + 1)]
            self.group_indices.extend(indices)


    def _adjust_group(self, group, num_stock):
        if len(group) == 0:
            return group
        
        elif len(group) < num_stock:
            additional_rows = num_stock - len(group)

            dates = group.index.get_level_values(0).unique()
            empty_stock_names = ['empty' for _ in range(additional_rows)]
            new_index = pd.MultiIndex.from_product([dates, empty_stock_names], names=['datetime', 'stock_name'])
            extra_data = pd.DataFrame(0, index=new_index, columns=group.columns)
            group = pd.concat([group, extra_data])

        group.sort_values(by='LABEL0', ascending=False)    

        return group

    def __len__(self):
        return len(self.group_indices)
        # return len(self.df) - self.sequence_length +1  # subtract 30 to account for accumulation of 30 days of data

    def __getitem__(self, idx):

        group_name, group_start_idx = self.group_indices[idx]

        group = self.instrument_groups.get_group(group_name)
        group = group.fillna(method='ffill')
        group = group.fillna(0)

        data = group.iloc[group_start_idx:group_start_idx + self.sequence_length]
        
        input_data = data.drop('LABEL0', axis=1).values 
        label = data['LABEL0'].values 

        return input_data, label

        #### BUG ####
        # input_data = []
        # label_data = []
        
        # idx_list = [300*i + idx for i in range(self.sequence_length) if 300*i + idx < len(self.df)]

        # data = self.df.iloc[idx_list, :-1].values #(seq_len, character)
        # label = self.df.iloc[idx_list, -1].values #(seq_len, 1)
        
        # input_data.append(data)
        # input_data = np.concatenate(input_data, axis=0)
        # label_data.append(label)        
        
        # # label 도출 값은 반드시 잘라서 써야함.
        # return input_data, label
