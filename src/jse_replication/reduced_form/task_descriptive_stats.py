import pandas as pd
import pytask 
import matplotlib.pyplot as plt
import numpy as np
#from reduced_form.weighted_moments import *
from jse_replication.reduced_form.weighted_moments import *
from jse_replication.reduced_form.helper_functions import *
from scipy import stats

#deps = "../bld/author_data/sce_datafile.dta"

from jse_replication.config import BLD,SRC

@pytask.mark.depends_on(BLD / "author_data" / "sce_datafile.dta")
@pytask.mark.produces(BLD / "tables" / "tab1_summary_statistics_sce.tex")
def task_make_tab1(depends_on,produces):
    df = prep_data(depends_on)
    rslt = _combine_summary_stats(df)
    rslt.to_latex(produces)


def _combine_summary_stats(df):   
    pap_rslt = make_paper_tab1(df)
    real_rslt = make_tab1(df)
    rslt = pd.DataFrame({'Own calculations':real_rslt,'Paper':pap_rslt})
    return rslt


def make_tab1(df):
    # TODO: Put all of this into a nice format
    # Ideally LaTeX or HTML
    # Number of observations
    obs = len(df)
    # Number of individual respondents
    sub_obs = df.index.get_level_values(0).nunique()
    # Respondents with more than one observation
    sub_obs_more2 = (df.groupby('userid')['x'].sum()>1).sum()
    # Make summary for age bins
    res_age = _age_bins(df) # NOT replicated => They do it for the full sample mistakenly
    # Education statistics
    res_edu = _edu_stats(df) # NOT replicated => They do it for the full sample mistakenly
    # Female
    fem = weighted_value_counts(df.groupby('userid')['female','weight'].first(),'female').loc[1]
    # Black 
    black = weighted_value_counts(df.groupby('userid')['black','weight'].first(),'black').loc[1]
    # Hispanic
    hispanic = weighted_value_counts(df.groupby('userid')['hispanic','weight'].first(),'hispanic').loc['Yes']
    # 1 month job finding rate
    
    ue_jf_1mon = weighted_mean(df.loc[df['UE_trans_1mon'].notna() & df['weight'].notna()], 'UE_trans_1mon')
    # longterm unemployed
    df_temp = df.loc[df['UE_trans_1mon'].notna() & df['weight'].notna() & (df['longterm_unemployed']==1)]
    ue_jf_1mon_lt = weighted_mean(df_temp, 'UE_trans_1mon')
    # not longterm unemployed
    df_temp = df.loc[df['UE_trans_1mon'].notna() & (df['longterm_unemployed']==0)]
    ue_jf_1mon_not_lt = weighted_mean(df_temp, 'UE_trans_1mon')

    
    rslt = pd.concat([res_edu, res_age], ignore_index=False)
    rslt.loc['Female'] = fem
    rslt.loc['Black'] = black
    rslt.loc['Hispanic'] = hispanic
    # Job finding rate
    rslt.loc['Full sample'] = ue_jf_1mon
    rslt.loc['Duration (0-6)'] = ue_jf_1mon_not_lt
    rslt.loc['Duration 7+'] = ue_jf_1mon_lt
    
    rslt = (rslt*100).astype(float).round(1)
    
    # Observations
    rslt.loc['# respondent'] = int(sub_obs)
    rslt.loc['# respondents w/ atleast 2 surveys'] = int(sub_obs_more2)
    rslt.loc['# survey responses'] = int(obs)
    
    return rslt

def make_paper_tab1(df):
    ## TODO: Make it also nice
    # Number of observations
    obs = len(df)
    # Number of individual respondents
    sub_obs = df.index.get_level_values(0).nunique()
    # Respondents with more than one observation
    sub_obs_more2 = (df.groupby('userid')['x'].sum()>1).sum()
    # Make summary for age bins
    df['age_bins'] =  pd.cut(df['age'],bins=[20,34,49,65],labels=['Ages 20-34','Ages 35-49','Ages 50-65'],include_lowest=True)
    res_age = weighted_value_counts(df,'age_bins') 
    res_edu = weighted_value_counts(df.loc[df['edu_cat'] !=''],'edu_cat')
    # Female
    fem = weighted_value_counts(df,'female').loc[1]
    # Black 
    black = weighted_value_counts(df,'black').loc[1]
    # Hispanic
    hispanic = weighted_value_counts(df,'hispanic').loc['Yes']
    # 1 month job finding rate
    
    ue_jf_1mon = weighted_mean(df.loc[df['UE_trans_1mon'].notna() & df['weight'].notna()], 'UE_trans_1mon')
    # longterm unemployed
    df_temp = df.loc[df['UE_trans_1mon'].notna() & df['weight'].notna() & (df['longterm_unemployed']==1)]
    ue_jf_1mon_lt = weighted_mean(df_temp, 'UE_trans_1mon')
    # not longterm unemployed
    df_temp = df.loc[df['UE_trans_1mon'].notna() & (df['longterm_unemployed']==0)]
    ue_jf_1mon_not_lt = weighted_mean(df_temp, 'UE_trans_1mon')

    
    rslt = pd.concat([res_edu, res_age], ignore_index=False)
    rslt.loc['Female'] = fem
    rslt.loc['Black'] = black
    rslt.loc['Hispanic'] = hispanic
    # Job finding rate
    rslt.loc['Full sample'] = ue_jf_1mon
    rslt.loc['Duration (0-6)'] = ue_jf_1mon_not_lt
    rslt.loc['Duration 7+'] = ue_jf_1mon_lt
    
    rslt = (rslt*100).astype(float).round(1)
    
    # Observations
    rslt.loc['# respondent'] = int(sub_obs)
    rslt.loc['# respondents w/ atleast 2 surveys'] = int(sub_obs_more2)
    rslt.loc['# survey responses'] = int(obs)
    return rslt
    
def _age_bins(df):
    age_bins =pd.cut(df.groupby('userid')['age'].first(),bins=[20,34,49,65],labels=['Ages 20-34','Ages 35-49','Ages 50-65'],include_lowest=True)
    weight = df.groupby('userid')['weight'].first()
    res = pd.DataFrame({'age_bins': age_bins, 'weight': weight})
    return weighted_value_counts(res, 'age_bins')
    
def _edu_stats(df):
    res = df.loc[df['edu_cat'] != '']
    return weighted_value_counts(res.groupby('userid')['edu_cat','weight'].first(),'edu_cat')
    