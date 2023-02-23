import pandas as pd
import numpy as np

def read_SCE(depends_on):
    df = pd.read_stata(depends_on)
    df.set_index(['userid','date'],inplace=True)
    return df

def restrict_sample(df):
    # drop if we don't have weights/no expectations 
    df.dropna(subset=['weight'],inplace=True)
    #df.dropna(subset=['find_job_3mon','find_job_12mon'],how='all',inplace=True)
    # age restriction 20-65
    df = df.loc[df.query('age >= 20 & age <= 65').index.get_level_values(0).unique()]
    df['x'] = 1
    return df

def prep_data(depends_on):
    df = read_SCE(depends_on)
    df = restrict_sample(df)
    return df