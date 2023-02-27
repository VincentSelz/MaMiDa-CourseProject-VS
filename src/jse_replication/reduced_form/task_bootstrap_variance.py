import pandas as pd
import pytask
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from jse_replication.reduced_form.weighted_moments import *
from jse_replication.reduced_form.helper_functions import *

from jse_replication.config import BLD,SRC


@pytask.mark.depends_on(
    [BLD / "author_data" / "sce_datafile.dta",
     BLD / "data" / "new_sce_data.csv"])
@pytask.mark.produces(BLD / "tables" / "tab3_bootstrap_lower_bound_variance.tex")
def task_make_tab3(depends_on,produces):
    auth_df = prep_auth_data(depends_on[0])
    df = prep_SCE_data(depends_on[1])
    df = df.loc[df['in_sample_2']== 1]  
    auth_df = auth_df.loc[auth_df['in_sample_2']== 1]  
    auth_df = _gen_columns(auth_df)
    df = _gen_columns(df)
    
    # Get the variance values
    lb_z1,lb_z12 = _lb_variance_base(df)
    cov_z_pred1, cov_z_pred2 = _lb_variance_reg(df)    
    # Bootstrap to obtain the standard errors
    se_s = _bootstrap_variance(df,num_bootsim=2000,seed=858)
    
    # Get the variance values
    auth_lb_z1,auth_lb_z12 = _lb_variance_base(auth_df)
    auth_cov_z_pred1, auth_cov_z_pred2 = _lb_variance_reg(auth_df)    
    # Bootstrap to obtain the standard errors
    auth_se_s = _bootstrap_variance(auth_df,num_bootsim=2000,seed=89)
    
    # Make table
    _make_latex_table_from_scratch(lb_z1,lb_z12,cov_z_pred1, cov_z_pred2,se_s,auth_lb_z1,auth_lb_z12,auth_cov_z_pred1, auth_cov_z_pred2,auth_se_s,produces)
    
def _make_latex_table_from_scratch(lb_z1,lb_z12,cov_z_pred1, cov_z_pred2,se_s,auth_lb_z1,auth_lb_z12,auth_cov_z_pred1, auth_cov_z_pred2,auth_se_s,produces):  
    #open text file
    lb_z1_se = se_s.loc['lb_z1']
    lb_z12_se = se_s.loc['lb_z12']
    cov_z_pred1_se = se_s.loc['cov_z_pred1']
    cov_z_pred2_se = se_s.loc['cov_z_pred2']
    auth_lb_z1_se = auth_se_s.loc['lb_z1']
    auth_lb_z12_se = auth_se_s.loc['lb_z12']
    auth_cov_z_pred1_se = auth_se_s.loc['cov_z_pred1']
    auth_cov_z_pred2_se = auth_se_s.loc['cov_z_pred2']
    
    text_file = open(produces, "w")
    table = rf"""\begin{{table}}[!htbp] \centering 
    \caption{{Lower Bound Variance}}
    \label{{tab:lb_variance_table3}} 
    \begin{{tabular}}{{lcccc}}
    \toprule
    & \multicolumn{{2}}{{c}}{{\textbf{{Own Data}}}} & \multicolumn{{2}}{{c}}{{\textbf{{Author Data}}}} \\
    Lower bound based on: &  Value &   SE & Value & SE \\
    \midrule
    ... 3-month elicitations only & {lb_z1:.3f} & {lb_z1_se:.3f}  & {auth_lb_z1:.3f} & {auth_lb_z1_se:.3f}  \\
    ... 3- and 12-month elicitations & {lb_z12:.3f} & {lb_z12_se:.3f}  &  {auth_lb_z12:.3f} & {auth_lb_z12_se:.3f}  \\
    ... only control & {cov_z_pred1:.3f} & {cov_z_pred1_se:.3f}  & {auth_cov_z_pred1:.3f} &  {auth_cov_z_pred1_se:.3f}  \\
    ... controls and both elicitations & {cov_z_pred2:.3f} & {cov_z_pred2_se:.3f}  & {auth_cov_z_pred2:.3f} & {auth_cov_z_pred2_se:.3f} \\
    \bottomrule
    \footnotesize{{\textit{{Notes:}} Standard errors are bootstrapped with 2,000 samples.}}
    \end{{tabular}}
    \end{{table}}
    """
    #write string to file
    text_file.write(table)
 
    #close file
    text_file.close()
    
def _bootstrap_variance(df,num_bootsim=2000,seed=345):
    """
   Calculates the bootstrap variance for a given DataFrame. 

   Args:
   - df (pandas.DataFrame): DataFrame containing the data to be analyzed. Must have a multi-level index with 
     'userid' and 'spell_id' as the first two levels.
   - num_bootsim (int): number of bootstrap samples to generate (default=2000)
   - seed (int): random seed to use for the random number generator (default=345)

   Returns:
   - pandas.Series: standard deviation of the bootstrap results for the lower bound of z_1, lower bound of z_1 and z_2,
     covariance of z_pred and z_1, and covariance of z_pred and z_2.
   """
    np.random.seed(seed)
    users = df.index.get_level_values(0).unique()
    rslt = pd.DataFrame(columns=['lb_z1','lb_z12','cov_z_pred1','cov_z_pred2'])
    # Bootstrap 2000 times
    for boot in range(num_bootsim):
        boot_users = np.random.choice(users, size=len(users), replace=True)
        df_boot = df.loc[boot_users]
        lb_z1,lb_z12 = _lb_variance_base(df_boot)
        cov_z_pred1, cov_z_pred2 = _lb_variance_reg(df_boot)
        rslt.loc[boot,'lb_z1'] = lb_z1
        rslt.loc[boot,'lb_z12'] = lb_z12
        rslt.loc[boot,'cov_z_pred1'] = cov_z_pred1
        rslt.loc[boot,'cov_z_pred2'] = cov_z_pred2
        #print(boot)
    return rslt.std()

def _gen_columns(df):
    df = df.copy()
    # Generate useful columns
    df['age2'] = df['age']*df['age']
    df['userid'] = df.index.get_level_values(0)
    # Generate imputed values
    ones = pd.Series(np.ones(len(df)),index=df.index)
    df['imputed_3mon'] = (ones - (ones - df['find_job_12mon']).pow(0.25))
    # Compute annualized probability on the basis of the 3month elicited rate
    df['imputed_12mon'] = df['find_job_3mon']+(ones-df['find_job_3mon'])*df['find_job_3mon']+(ones-df['find_job_3mon']).pow(2)*df['find_job_3mon']+(ones-df['find_job_3mon']).pow(3)*df['find_job_3mon']    
    return df

def _lb_variance_base(df):
    df = df.copy()
    # Compute the variance
    v_z1 = w_var(df,'find_job_3mon')
    # Compute the covariance
    cov_z1t = w_cov(df, 'find_job_3mon','UE_trans_3mon')
    # Compute the lower bound using the formula
    lb_z1 = ((cov_z1t)**2)/v_z1
    # Compute covariance with 12 month too
    cov_z2t = w_cov(df,'imputed_3mon', 'UE_trans_3mon')
    # Compute covariance between measurements
    cov_z1z2 = w_cov(df, 'find_job_3mon','imputed_3mon')
    # Compute the lower bound using hte formula for two measurements
    lb_z1z2 = (cov_z1t*cov_z2t)/cov_z1z2    
    return lb_z1,lb_z1z2

def _lb_variance_reg(df):
    df = df.copy()
    # Only use controls for the regression
    controls = 'female + black + hispanic + age + age2 + r_asoth + other_race + education_2 + education_3 + education_4 + education_5 + education_6 + hhinc_2 + hhinc_3 + hhinc_4'
    dep_control = 'UE_trans_3mon ~ '+controls
    # Restrict sample to valid observations
    df1 = df.loc[df['UE_trans_3mon'].notna()]
    # Only use controls
    mod1 = smf.wls(dep_control,data=df1, weights=df1['weight']).fit(cov_type='cluster', cov_kwds={'groups': df1['userid']})
    df1['pred1'] = mod1.predict()
    cov_z_pred1 = w_cov(df1,'pred1','UE_trans_3mon')
    df2 = df.loc[df['UE_trans_3mon'].notna() & df['find_job_12mon'].notna() & df['weight'].notna()]
    dep_job_control = 'UE_trans_3mon ~ '+' find_job_3mon + imputed_3mon +'+controls
    mod2 = smf.wls(dep_job_control,data=df2, weights=df2['weight']).fit(cov_type='cluster', cov_kwds={'groups': df2['userid']})
    df2['pred2'] = mod2.predict()
    cov_z_pred2 = w_cov(df2,'pred2','UE_trans_3mon')
    return cov_z_pred1,cov_z_pred2
    

#df = read_SCE(deps)
#df = restrict_sample(df)
#df = df.loc[df['in_sample_2']== 1]
#df = _gen_columns(df)


#### We can reject a linear trend in beliefs on a 5% level
#base_formula = 'find_job_3mon ~ imputed_3mon -1'
#mod1 = smf.wls(base_formula, data=df.loc[df['imputed_3mon'].notna()], weights= df.loc[df['imputed_3mon'].notna(),'weight']).fit(cov_type='cluster', cov_kwds={'groups': df.loc[df['imputed_3mon'].notna(),'userid']})
#mod1.summary()
# Imputed annualized expected transition rate > Elicited 12-month expected transition rate
# They know they have duration dependence OR they have problems with math
