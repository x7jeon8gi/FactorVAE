import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, Sampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import datetime
import random
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import torch
import bisect
from torch.utils.data import Dataset



def np_ffill(arr):
    """
    NumPy-based forward fill function.
    Fills NaN values in an array with the previous value.

    Parameters:
    - arr: np.array

    Returns:
    - np.array, forward fill이 적용된 배열
    """
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = arr[idx]
    return out

class TSDataSampler:
    """
    Time-Series DataSampler for tabular data.
    """

    def __init__(
        self, data: pd.DataFrame, start, end, step_len: int, fillna_type: str = "none", dtype=None, flt_data=None
    ):
        """
        Parameters
        ----------
        data : pd.DataFrame
            The raw tabular data with a MultiIndex (datetime, instrument).
        start :
            The indexable start time.
        end :
            The indexable end time.
        step_len : int
            The length of the time-series step.
        fillna_type : str
            How to handle missing data:
            - none: fill with np.nan
            - ffill: forward fill with previous samples
            - ffill+bfill: forward fill then backward fill
        dtype : type
            The data type to which the numpy array should be cast.
        flt_data : pd.Series
            A boolean Series to filter data.
        """
        self.start = start
        self.end = end
        self.step_len = step_len
        self.fillna_type = fillna_type

        # Ensure that the index is sorted by datetime
        assert data.index.names == ["datetime", "instrument"]
        self.data = data.sort_index()

        # Convert DataFrame to numpy array for better performance
        self.data_arr = data.to_numpy(dtype=dtype)
        self.data_arr = np.append(
            self.data_arr, np.full((1, self.data_arr.shape[1]), np.nan, dtype=self.data_arr.dtype), axis=0
        )
        self.nan_idx = -1  # The last line is all NaN

        # Build indices
        self.idx_df, self.idx_map = self.build_index(self.data)
        self.data_index = self.data.index

        if flt_data is not None:
            self.flt_data = flt_data.reindex(self.data_index).fillna(False).astype(bool)
            self.idx_map = self.flt_idx_map(self.flt_data, self.idx_map)
            self.data_index = self.data_index[self.flt_data]

        self.idx_map = self.idx_map2arr(self.idx_map)

        self.start_idx, self.end_idx = self.data_index.slice_locs(
            start=pd.Timestamp(start), end=pd.Timestamp(end)
        )
        self.idx_arr = np.array(self.idx_df.values, dtype=np.float64)

    @staticmethod
    def idx_map2arr(idx_map):
        dtype = np.int32
        no_existing_idx = (np.iinfo(dtype).max, np.iinfo(dtype).max)

        max_idx = max(idx_map.keys())
        arr_map = []
        for i in range(max_idx + 1):
            arr_map.append(idx_map.get(i, no_existing_idx))
        arr_map = np.array(arr_map, dtype=dtype)
        return arr_map

    @staticmethod
    def flt_idx_map(flt_data, idx_map):
        idx = 0
        new_idx_map = {}
        for i, exist in enumerate(flt_data):
            if exist:
                new_idx_map[idx] = idx_map[i]
                idx += 1
        return new_idx_map

    def get_index(self):
        return self.data_index[self.start_idx: self.end_idx]

    @staticmethod
    def build_index(data: pd.DataFrame) -> tuple:
        idx_df = pd.Series(range(data.shape[0]), index=data.index, dtype=object).unstack()
        idx_df = idx_df.sort_index().sort_index(axis=1)

        idx_map = {}
        for i, (_, row) in enumerate(idx_df.iterrows()):
            for j, real_idx in enumerate(row):
                if not np.isnan(real_idx):
                    idx_map[real_idx] = (i, j)
        return idx_df, idx_map

    def _get_indices(self, row: int, col: int) -> np.array:
        indices = self.idx_arr[max(row - self.step_len + 1, 0): row + 1, col]

        if len(indices) < self.step_len:
            indices = np.concatenate([np.full((self.step_len - len(indices),), np.nan), indices])

        if self.fillna_type == "ffill":
            indices = np_ffill(indices)
        elif self.fillna_type == "ffill+bfill":
            indices = np_ffill(np_ffill(indices)[::-1])[::-1]
        else:
            assert self.fillna_type == "none"
        return indices

    def _get_row_col(self, idx) -> tuple:
        if isinstance(idx, (int, np.integer)):
            real_idx = self.start_idx + idx
            if self.start_idx <= real_idx < self.end_idx:
                i, j = self.idx_map[real_idx]
            else:
                raise KeyError(f"{real_idx} is out of bounds [{self.start_idx}, {self.end_idx})")
        elif isinstance(idx, tuple):
            date, inst = idx
            date = pd.Timestamp(date)
            i = bisect.bisect_right(self.idx_df.index, date) - 1
            j = bisect.bisect_left(self.idx_df.columns, inst)
        else:
            raise NotImplementedError(f"This type of input is not supported")
        return i, j

    def __getitem__(self, idx):
        if isinstance(idx, (list, np.ndarray)):
            indices = [self._get_indices(*self._get_row_col(i)) for i in idx]
            indices = np.concatenate(indices)
        else:
            indices = self._get_indices(*self._get_row_col(idx))

        indices = np.nan_to_num(indices.astype(np.float64), nan=self.nan_idx).astype(int)
        data = self.data_arr[indices]
        actual_indices = self.data_index[indices]
        if isinstance(idx, (list, np.ndarray)):
            data = data.reshape(-1, self.step_len, *data.shape[1:])
        return data, actual_indices

    def __len__(self):
        return self.end_idx - self.start_idx


class TSDatasetH(Dataset):
    DEFAULT_STEP_LEN = 20

    def __init__(self, data, step_len=DEFAULT_STEP_LEN, **kwargs):
        self.step_len = step_len
        self.data = data
        self.sampler = TSDataSampler(data = data, step_len = step_len, **kwargs)

    def __getitem__(self, idx):
        return self.sampler[idx]

    def __len__(self):
        return len(self.sampler)

    def config(self, **kwargs):
        if "step_len" in kwargs:
            self.step_len = kwargs.pop("step_len")
        self.sampler.config(**kwargs)


class DateGroupedBatchSampler(Sampler):
    """Sampler that groups data by date and returns entire groups as batches."""
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.grouped_indices = self._group_indices_by_date()

    def _group_indices_by_date(self):
        # Create an index that takes into account start and end dates
        start_idx = self.data_source.sampler.start_idx
        end_idx = self.data_source.sampler.end_idx

        # Use only data within a specified start and end date range
        data_index = self.data_source.sampler.data_index[start_idx:end_idx]

        # Group indexes by date
        indices = pd.Series(range(len(data_index)), index=data_index.get_level_values('datetime'))
        grouped = indices.groupby(level='datetime').apply(list).values
        return grouped

    def __iter__(self):
        # Set whether to randomly shuffle groups
        if self.shuffle:
            np.random.shuffle(self.grouped_indices)
        
        # Return indexes grouped by each date in a batch
        for group in self.grouped_indices:
            yield group

    def __len__(self):
        # Number of groups equals number of dates
        return len(self.grouped_indices)
    


def custom_collate_fn(batch):
    data, indices = zip(*batch)  # Separate data and indexes
    data = torch.utils.data.dataloader.default_collate(data)  # Process data with the default collate function
    
    # Convert MultiIndex to List
    indices = [list(index) for index in indices]
    
    return data, indices


def init_data_loader(df, step_len, shuffle, start, end, select_feature=None):
    """
    DataLoader Initialization Functions

    Parameters:
    - dataset: StockDataset 
    - shuffle: Whether to shuffle data

    Returns:
    - DataLoader 
    """
    if select_feature is not None:
        df = df[select_feature]

    dataset = TSDatasetH(df, step_len=step_len, start=start, end=end, fillna_type='ffill+bfill')
    sampler = DateGroupedBatchSampler(dataset, shuffle=shuffle)
    data_loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=custom_collate_fn,  
        pin_memory=True,
    )
    return data_loader

if __name__ == "__main__":
    df = pd.read_pickle('data/csi_data.pkl')
    step_len = 1  # Time Series Length

    # Initializing the DataLoader
    data_loader = init_data_loader(df, step_len, 
                                   shuffle=False, start='2010-01-01', end='2015-01-01', 
                                   select_feature=None)

    # Iterating over data using DataLoader
    for batch, indices in data_loader:
        input_data, labels = batch[:,:,:-1], batch[:,-1,-1].unsqueeze(-1)
        
        # Index information is converted to a list
        # print("Batch Indices:", np.array(indices)[:, -1])
        print(input_data.shape, labels.shape)
    print("Done")