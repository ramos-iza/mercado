import pandas as pd 

def read_csv(path):
    df = pd.read_csv(path, index_col='Date')
    return df 

