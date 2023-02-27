import pandas as pd
import pytask 
import matplotlib.pyplot as plt
import numpy as np
from jse_replication.reduced_form.weighted_moments import *
from jse_replication.reduced_form.helper_functions import *
from scipy import stats

from jse_replication.config import BLD,SRC

@pytask.mark.depends_on(
    [BLD / "author_data" / "sce_datafile.dta",
     BLD / "data" / "new_sce_data.csv"])
@pytask.mark.produces(BLD / "tables" / "tab1_summary_statistics_sce.tex")
def task_make_tab1(depends_on,produces):
    auth_df = prep_auth_data(depends_on[0])
    df = prep_SCE_data(depends_on[1])
    
    #auth_df = auth_df.loc[auth_df['in_sample_1']==1]
    
    df['edu_cat'] = df['edu_cat'].replace(['Some college, including associates degree'], 'Some college')
    df['edu_cat'].replace({'College':'College grad plus','Some College':'Some college','High School':'Up to HS grad'},inplace=True)
    
    # Calculation used in the paper
    pap_rslt = _make_paper_tab1(auth_df)
    # Error corrected version
    real_pap_rslt = _make_tab1(auth_df)
    # Own data
    real_rslt = _make_real_tab1(df)
    
    tab = _write_tab1(real_rslt,pap_rslt,real_pap_rslt)
    
    _export_tab(tab,produces)
    #rslt = _combine_summary_stats(auth_df,df)
    #rslt.to_latex(produces)

def _write_tab1(real_rslt,pap_rslt,real_pap_rslt):
    tab = rf"""
    \begin{{tabular}}{{lccc}}
    \toprule
     &  Own Data &   \multicolumn{{2}}{{c}}{{Paper Data}} \\
     &   (1) & (2) & (3) \\
         \midrule
        High-School Degree or Less     &   {real_rslt.loc['Up to HS grad']:.1f} & {real_pap_rslt.loc['Up to HS grad']:.1f} &  {pap_rslt.loc['Up to HS grad']:.1f} \\
        Some College Education         &   {real_rslt.loc['Some college']:.1f} & {real_pap_rslt.loc['Some college']:.1f} &  {pap_rslt.loc['Some college']:.1f} \\
        College Degree or More         &    {real_rslt.loc['College grad plus']:.1f} & {real_pap_rslt.loc['College grad plus']:.1f} &  {pap_rslt.loc['College grad plus']:.1f} \\
        Ages 20-34                         &    {real_rslt.loc['Ages 20-34']:.1f} & {real_pap_rslt.loc['Ages 20-34']:.1f} &  {pap_rslt.loc['Ages 20-34']:.1f} \\
        Ages 35-49                         &    {real_rslt.loc['Ages 35-49']:.1f} & {real_pap_rslt.loc['Ages 35-49']:.1f} &  {pap_rslt.loc['Ages 35-49']:.1f} \\
        Ages 50-65                         &    {real_rslt.loc['Ages 50-65']:.1f} & {real_pap_rslt.loc['Ages 50-65']:.1f} &  {pap_rslt.loc['Ages 50-65']:.1f} \\
        Female                             &    {real_rslt.loc['Female']:.1f} & {real_pap_rslt.loc['Female']:.1f} &  {pap_rslt.loc['Female']:.1f} \\
        Black                              &    {real_rslt.loc['Black']:.1f} & {real_pap_rslt.loc['Black']:.1f} &  {pap_rslt.loc['Black']:.1f} \\
        Hispanic                           &    {real_rslt.loc['Hispanic']:.1f} & {real_pap_rslt.loc['Hispanic']:.1f} &  {pap_rslt.loc['Hispanic']:.1f} \\
        \midrule
        Monthly job finding rate & & \\
        ...Full sample                        &   {real_rslt.loc['Full sample']:.1f} & {real_pap_rslt.loc['Full sample']:.1f} &  {pap_rslt.loc['Full sample']:.1f} \\
        ...Duration (0-6)                     &   {real_rslt.loc['Duration (0-6)']:.1f} & {real_pap_rslt.loc['Duration (0-6)']:.1f} &  {pap_rslt.loc['Duration (0-6)']:.1f} \\
        ...Duration 7+                        &   {real_rslt.loc['Duration 7+']:.1f} & {real_pap_rslt.loc['Duration 7+']:.1f} &  {pap_rslt.loc['Duration 7+']:.1f} \\
        \midrule
        \# of respondents                       &    {real_rslt.loc['# respondent']:.0f} & {real_pap_rslt.loc['# respondent']:.0f} &  {pap_rslt.loc['# respondent']:.0f} \\
        \# of respondents w/ atleast 2 surveys &    {real_rslt.loc['# respondents w/ atleast 2 surveys']:.0f} & {real_pap_rslt.loc['# respondents w/ atleast 2 surveys']:.0f} &  {pap_rslt.loc['# respondents w/ atleast 2 surveys']:.0f} \\
        \# of survey responses                 &    {real_rslt.loc['# survey responses']:.0f} & {real_pap_rslt.loc['# survey responses']:.0f} &  {pap_rslt.loc['# survey responses']:.0f} \\
        \bottomrule
    \end{{tabular}}
    """
    return tab

def _export_tab(tab,produces):
    tex_file = open(produces, "w")
    tex_file.write(tab)
    tex_file.close() 

def _combine_summary_stats(auth_df,df):   
    # Calculation used in the paper
    pap_rslt = _make_paper_tab1(auth_df)
    # Error corrected version
    real_pap_rslt = _make_tab1(auth_df)
    # Own data
    real_rslt = _make_real_tab1(df)
    rslt = pd.DataFrame({'Own Data': real_rslt,r'Own calculations, \\ Paper Data':real_pap_rslt,'Paper':pap_rslt})
    return rslt

def _make_real_tab1(df):
    df = df.loc[(df['find_job_3mon'].notna()|df['find_job_12mon'].notna())& df['weight'].notna()]
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
    hispanic = weighted_value_counts(df.groupby('userid')['hispanic','weight'].first(),'hispanic').loc[1]
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
    

def _make_tab1(df):
    df = df.loc[(df['find_job_3mon'].notna()|df['find_job_12mon'].notna())& df['weight'].notna()]
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
    
    rslt.loc['Some college'] = rslt.loc['Some college, including associates degree']
    return rslt

def _make_paper_tab1(df):
    df = df.loc[(df['find_job_3mon'].notna()|df['find_job_12mon'].notna())& df['weight'].notna()]
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
    
    rslt.loc['Some college'] = rslt.loc['Some college, including associates degree']
    return rslt
    
def _age_bins(df):
    age_bins =pd.cut(df.groupby('userid')['age'].first(),bins=[20,34,49,65],labels=['Ages 20-34','Ages 35-49','Ages 50-65'],include_lowest=True)
    weight = df.groupby('userid')['weight'].first()
    res = pd.DataFrame({'age_bins': age_bins, 'weight': weight})
    return weighted_value_counts(res, 'age_bins')
    
def _edu_stats(df):
    res = df.loc[df['edu_cat'] != '']
    return weighted_value_counts(res.groupby('userid')['edu_cat','weight'].first(),'edu_cat')
    