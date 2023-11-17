import pandas as pd
import wfdb

pd.set_option('display.max_columns', None)
mb_artm_directory = './databases/mit-bih-arrhythmia-database-1.0.0/'
mb_nsrdb_directory = './databases/mit-bih-normal-sinus-rhythm-database-1.0.0/'
dataframes_directory = './Results/dataframes/'
figures_directory = './Results/figures/'


def load_record(record: int, files_directory=mb_artm_directory, **kwargs):
    record_path = f'{files_directory}{record}'
    record = wfdb.rdrecord(record_path, **kwargs)
    ann = wfdb.rdann(record_path, 'atr', **kwargs)
    return record, ann