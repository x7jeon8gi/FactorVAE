I used Qlib's scripts to download the data.

See the address below.
https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo

example
1. download data to csv <br>
python collector.py download_data --source_dir ~/.qlib/stock_data/source/cn_data --start 2020-01-01 --end 2020-12-31 --delay 1 --interval 1d --region CN --max_workers 8 --max_collector_count 2
2. normalize data <br>
python collector.py normalize_data --source_dir ~/.qlib/stock_data/source/cn_data --normalize_dir ~/.qlib/stock_data/source/cn_1d_nor --region CN --interval 1d
3. dump data from scripts folder <br>
python dump_bin.py dump_all --csv_path ~/.qlib/stock_data/source/cn_1d_nor --qlib_dir ~/.qlib/qlib_data/cn_data --freq day --exclude_fields date,symbol