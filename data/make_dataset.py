"""
Generating data using Qlib
Alpha158 is an off-the-shelf dataset provided by Qlib.
"""

import qlib
import pandas as pd
from qlib.constant import REG_CN, REG_US
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset import DatasetH, TSDatasetH, TSDataSampler
from qlib.contrib.data.handler import Alpha158
import argparse


if __name__ == "__main__":

    # Argument parser 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/qlib_data/cn_data")
    parser.add_argument("--freq", type=str, default="day")
    parser.add_argument('--start_time', type=str, default='2008-01-01')
    parser.add_argument('--end_time', type=str, default='2020-12-31')
    parser.add_argument('--fit_end_time', type=str, default='2017-12-31')
    parser.add_argument('--val_start_time', type=str, default='2018-01-01')
    parser.add_argument('--val_end_time', type=str, default='2018-12-31')
    parser.add_argument('--test_start_time', type=str, default='2019-01-01')
    parser.add_argument('--seq_len', type=int, default=21)
    args = parser.parse_args()

    # Qlib을 이용한 데이터 생성
    if args.data_path.split('/')[-1] == "cn_data":
        qlib.init(provider_uri=args.data_path, region=REG_CN)
        benchmark = "SH000300"
        market = "csi300"
    elif args.data_path.split('/')[-1] == "us_data":
        qlib.init(provider_uri=args.data_path, region=REG_US)
        benchmark = "^gspc"
        market = "sp500"

    provider_uri = args.data_path
    print(f"provider_uri: {provider_uri}")
    print(f"freq: {args.freq}")

    data_handler_config = {
        "start_time": args.start_time,
        "end_time": args.end_time,
        "fit_start_time": "2009-01-01", 
        "fit_end_time": args.fit_end_time,
        "instruments": market,
        "infer_processors": [
            # {"class" : "FilterCol", "kwargs" : {"fields_group" : "feature"},},
            {"class" : "RobustZScoreNorm","kwargs" : {"fields_group" : "feature", "clip_outlier" : True}},
            {"class" : "Fillna", "kwargs" : {"fields_group" : "feature"}}],
        "learn_processors": [
            {"class" : "DropnaLabel",}, 
            {"class" : "CSRankNorm", "kwargs" : {"fields_group" : "label"}}, # ! CSZScoreNorm 에서 CSRankNorm으로 변경
            ],
        "label": ["Ref($close, -2)/Ref($close, -1) - 1"],
    }

    segments = {
        'train': (args.start_time, args.fit_end_time),
        'valid': (args.val_start_time, args.val_end_time),
        'test': (args.test_start_time, args.end_time)
    }
    dataset = Alpha158(**data_handler_config)

    dataframe_L = dataset.fetch(col_set=["feature","label"], data_key=DataHandlerLP.DK_L) 
    dataframe_L.columns = dataframe_L.columns.droplevel(0)

    dataframe_I = dataset.fetch(col_set=["feature","label"], data_key=DataHandlerLP.DK_I)
    dataframe_I.columns = dataframe_I.columns.droplevel(0)

    if args.data_path.split('/')[-1] == "cn_data":
        #? market info not included in the dataset
        dataframe_LM = dataframe_L
        dataframe_IM = dataframe_I
        dataframe_LM.to_pickle('csi_data.pkl')
    
    elif args.data_path.split('/')[-1] == "us_data":
        dataframe_LM = dataframe_L
        dataframe_IM = dataframe_I
        dataframe_LM.to_pickle('sp500_data.pkl')

    ## TEST ##
    segments = {
        'train': (args.start_time, args.fit_end_time),
        'valid': (args.val_start_time, args.val_end_time),
        'test': (args.test_start_time, args.end_time)
    }

    handler = DataHandlerLP.from_df(dataframe_LM)
    dic =  {
            'train': ("2009-01-01", "2019-06-30"),
            'valid': ("2019-07-01", "2019-12-31",),
            'test': ("2020-01-01", "2022-12-31",),
        }
    QlibTSDatasetH = TSDatasetH(handler=handler, segments=dic, step_len=20)
    temp = QlibTSDatasetH.prepare(segments="train", data_key=DataHandlerLP.DK_L)

    print("------------------ Test QlibTSDatasetH ------------------")
    print(next(iter(temp)))