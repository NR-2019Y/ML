import os
import sys
import pandas as pd

def get_proj_root_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

module_path = os.path.join(get_proj_root_path(), 'mnist')
if module_path not in sys.path:
    sys.path.insert(0, module_path)
import adaboost.my_get_data as my_get_data

def load_xigua_df():
    fpath = os.path.join(get_proj_root_path(), 'datasets/xigua/xigua2.0.csv')
    return pd.read_csv(fpath, delimiter=',', index_col=0)
