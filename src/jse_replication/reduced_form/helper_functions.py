import pandas as pd
import numpy as np

def read_auth_data(depends_on):
    """Read in data from Mueller, Spinnewejn and Topda (2021)."""
    df = pd.read_stata(depends_on)
    df.set_index(['userid','date'],inplace=True)
    return df

def restrict_sample(df):
    """Make prelimanry data restrictions based on age and the existence of survey weights."""
    # drop if we don't have weights/no expectations 
    df.dropna(subset=['weight'],inplace=True)
    #df.dropna(subset=['find_job_3mon','find_job_12mon'],how='all',inplace=True)
    # age restriction 20-65
    df = df.loc[df.query('age >= 20 & age <= 65').index.get_level_values(0).unique()]
    df['x'] = 1
    return df

def prep_auth_data(depends_on):
    df = read_auth_data(depends_on)
    df = restrict_sample(df)
    return df

def read_SCE_data(depends_on):
    """Read in data from the website."""
    df = pd.read_csv(depends_on)
    df.set_index(['userid','date'],inplace=True)
    return df

def prep_SCE_data(depends_on):
    df = read_SCE_data(depends_on)
    # Have to be unemployed atleast once
    df = df.loc[(df['total_unemp']>0)]
    # Standard sample restrictions
    df = restrict_sample(df)
    return df

def export_tab(tab,produces):
    """Export a string-valued table as tex-file."""
    tex_file = open(produces, "w")
    tex_file.write(tab)
    tex_file.close() 