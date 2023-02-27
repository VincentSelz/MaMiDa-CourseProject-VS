import pandas as pd
import pytask
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer
#from reduced_form.task_descriptive_stats import *
from jse_replication.reduced_form.helper_functions import *
from linearmodels.panel import PanelOLS

from jse_replication.config import BLD,SRC

@pytask.mark.depends_on(
    [BLD / "author_data" / "sce_datafile.dta",
     BLD / "data" / "new_sce_data.csv"])
@pytask.mark.produces(BLD / "tables" / "tab_2_transition_rate_percep.tex")
def task_make_tab2(depends_on,produces):
    auth_df = prep_auth_data(depends_on[0])
    df = prep_SCE_data(depends_on[1])
    mod1, mod3, mod5, mod7 = _make_tab2_panel_a(df)
    mod2, mod4, mod6, mod8 = _make_tab2_panel_a(auth_df)
    tab_a = _write_tab2_panel_a(mod1,mod2,mod3,mod4,mod5,mod6,mod7,mod8)
    mod1, mod3, mod5, mod7 = _make_tab2_panel_b(df)
    mod2, mod4, mod6, mod8 = _make_tab2_panel_b(auth_df)
    tab_b = _write_tab2_panel_b(mod1,mod2,mod3,mod4,mod5,mod6,mod7,mod8)
    tab = tab_a+tab_b
    export_tab(tab, produces)
    
@pytask.mark.depends_on(
    [BLD / "author_data" / "sce_datafile.dta",
     BLD / "data" / "new_sce_data.csv"])
@pytask.mark.produces(BLD / "tables" / "tab_4_percep_udur.tex")
def task_make_tab4(depends_on,produces):
    auth_df = prep_auth_data(depends_on[0])
    df = prep_SCE_data(depends_on[1])
    auth_df = _prep_data_for_tab4(auth_df)
    df = _prep_data_for_tab4(df)
    mod1, mod3, mod5, mod7 = _tab4_regressions(df)
    mod2, mod4, mod6, mod8 = _tab4_regressions(auth_df)
    tab = _write_tab4(mod1,mod2,mod3,mod4,mod5,mod6,mod7,mod8)
    export_tab(tab, produces) 
    
def _write_tab2_panel_a(mod1,mod2,mod3,mod4,mod5,mod6,mod7,mod8):
    #tex_file = open(produces, "w")
    tab = rf"""
    \begin{{table}}[!htbp] \centering 
    \tiny
    \caption{{Linear Regressions of Realized Job Finding Rates on Elicitations}}
    \label{{tab:realized_perc_table2}} 
    \begin{{tabular}}{{lcccccccc}}
    \toprule
    \textbf{{Panel A.}} Dependent Variable: & & & & & & & & \\
    3-Month UE Transition Rate    & (1) & (2) & (3) & (4) & (5) & (6) & (7) & (8)\\
    \midrule
    Prob(Find Job in 3 Months) & ${mod1.params.loc['find_job_3mon']:.3f}^{{***}}$ & ${mod2.params.loc['find_job_3mon']:.3f}^{{***}}$ & & & ${mod5.params.loc['find_job_3mon']:.3f}^{{***}}$ & ${mod6.params.loc['find_job_3mon']:.3f}^{{***}}$ & ${mod7.params.loc['find_job_3mon']:.3f}^{{***}}$ & ${mod8.params.loc['find_job_3mon']:.3f}^{{***}}$     \\
                                & ({mod1.bse.loc['find_job_3mon']:.3f}) & ({mod2.bse.loc['find_job_3mon']:.3f}) & & & ({mod5.bse.loc['find_job_3mon']:.3f}) & ({mod6.bse.loc['find_job_3mon']:.3f}) & ({mod7.bse.loc['find_job_3mon']:.3f}) & ({mod8.bse.loc['find_job_3mon']:.3f}) \\
    Prob(Find Job in 3 Months)  & & & & & & & ${mod7.params.loc['find_job_3mon:longterm_unemployed']:.3f}^{{**}}$ & ${mod8.params.loc['find_job_3mon:longterm_unemployed']:.3f}^{{*}}$           \\
    x LT Unemployed                           & & & & & & & ({mod7.bse.loc['find_job_3mon:longterm_unemployed']:.3f}) & ({mod8.bse.loc['find_job_3mon:longterm_unemployed']:.3f})           \\
    LT Unemployed  & & & & & & & {mod7.params.loc['longterm_unemployed']:.3f} & {mod8.params.loc['longterm_unemployed']:.3f}           \\
                                & & & & & & & ({mod7.bse.loc['longterm_unemployed']:.3f}) & ({mod8.bse.loc['longterm_unemployed']:.3f})           \\
    \hline
    Controls & & & x & x & x & x & x & x \\
    Observations & {mod1.nobs:.0f} & {mod2.nobs:.0f} & {mod3.nobs:.0f} & {mod4.nobs:.0f} & {mod5.nobs:.0f} & {mod6.nobs:.0f} & {mod7.nobs:.0f} & {mod8.nobs:.0f} \\
    $R^{{2}}$    & {mod1.rsquared:.3f} & {mod2.rsquared:.3f} & {mod3.rsquared:.3f} & {mod4.rsquared:.3f} & {mod5.rsquared:.3f} & {mod6.rsquared:.3f} & {mod7.rsquared:.3f} & {mod8.rsquared:.3f} \\     
    \hline
    """
    #write string to file
    #tex_file.write(tab)
    #close file
    #tex_file.close()
    return tab

def _write_tab2_panel_b(mod1,mod2,mod3,mod4,mod5,mod6,mod7,mod8):
    #tex_file = open(produces, "w")
    tab = rf"""
    \hline \\
    \textbf{{Panel B.}} Dependent Variable: & & & & & & & & \\
    3-Month UE Transition Rate    & (1) & (2) & (3) & (4) & (5) & (6) & (7) & (8)\\
    \midrule
    Prob(Find Job in 3 Months)   & ${mod1.params.loc['tplus3_percep_3mon']:.3f}^{{**}}$ & ${mod2.params.loc['tplus3_percep_3mon']:.3f}^{{**}}$ & & & & & &  \\
                                & ({mod1.bse.loc['tplus3_percep_3mon']:.3f}) & ({mod2.bse.loc['tplus3_percep_3mon']:.3f}) & & & & & &  \\
    3-Month Lag    & & & & & ${mod5.params.loc['find_job_3mon']:.3f}^{{***}}$ & ${mod6.params.loc['find_job_3mon']:.3f}^{{***}}$ & ${mod7.params.loc['find_job_3mon']:.3f}^{{***}}$ & ${mod8.params.loc['find_job_3mon']:.3f}^{{***}}$           \\
    of Prob(Find Job in 3 Months) & & & & & ({mod5.bse.loc['find_job_3mon']:.3f}) & ({mod6.bse.loc['find_job_3mon']:.3f}) & ({mod7.bse.loc['find_job_3mon']:.3f}) & ({mod8.bse.loc['find_job_3mon']:.3f})          \\
    \hline
    Controls & & & x & x &  &  & x & x \\
    Observations & {mod1.nobs:.0f} & {mod2.nobs:.0f} & {mod3.nobs:.0f} & {mod4.nobs:.0f} & {mod5.nobs:.0f} & {mod6.nobs:.0f} & {mod7.nobs:.0f} & {mod8.nobs:.0f} \\
    $R^{{2}}$    & {mod1.rsquared:.3f} & {mod2.rsquared:.3f} & {mod3.rsquared:.3f} & {mod4.rsquared:.3f} & {mod5.rsquared:.3f} & {mod6.rsquared:.3f} & {mod7.rsquared:.3f} & {mod8.rsquared:.3f} \\     
    \hline 
    \end{{tabular}}
    \end{{table}}
    """
    notes = """
    \multicolumn{{9}}{{l}}{{\tablenotes \textit{{Notes:}} All regression use survey weights. The even columns are using the authors data and the uneven columns the own data. 
    Standard errors (in parentheses) are clustered on the individual level. *, **, and *** denote significance at the 10, 5, and 1 percent level.}} \\ 
    """
    #write string to file
    #tex_file.write(tab)
    #close file
    #tex_file.close()
    return tab

def _make_tab2_panel_a(df):
    # TODO: Make it look nice
    df = df.loc[df['in_sample_2']== 1]
    df['userid'] = df.index.get_level_values(0)
    # Baseline
    base_formula = 'UE_trans_3mon ~ find_job_3mon'
    mod1 = smf.wls(base_formula,data=df, weights=df['weight']).fit(cov_type='cluster', cov_kwds={'groups': df['userid']})
    mod1.summary()
    # Only Controls
    df['age2'] = df['age']**2
    controls = 'female + black + hispanic + age + age2 + r_asoth + other_race + education_2 + education_3 + education_4 + education_5 + education_6 + hhinc_2 + hhinc_3 + hhinc_4'
    dep_control = 'UE_trans_3mon ~'+' + '+controls
    mod2 = smf.wls(dep_control,data=df, weights=df['weight']).fit(cov_type='cluster', cov_kwds={'groups': df['userid']})
    mod2.summary()
    # Var of Interest and Control
    base_control = base_formula+' + '+controls
    mod3 = smf.wls(base_control,data=df, weights=df['weight']).fit(cov_type='cluster', cov_kwds={'groups': df['userid']})
    mod3.summary()
    # Add interaction of LT unemployed
    inter_control = 'UE_trans_3mon ~ find_job_3mon*longterm_unemployed + female + black + hispanic + age + age2 + r_asoth + other_race + education_2 + education_3 + education_4 + education_5 + education_6 + hhinc_2 + hhinc_3 + hhinc_4'
    mod4 = smf.wls(inter_control,data=df, weights=df['weight']).fit(cov_type='cluster', cov_kwds={'groups': df['userid']})
    mod4.summary()

    # How to access the relevant variables
    #mod1.params.loc['find_job_3mon']
    #mod1.rsquared
    #mod1.nobs
    return mod1, mod2, mod3, mod4

def make_tab3():
    """\begin{tabular}{lcccc}
    \textbf{Panel A.} Dependent Variable: & & & & \\
    3-Month UE Transition Rate    & (1) & (2) & (3) & (4) \\
    \hline
    Item 1        & \$10           & 3                 & \$30           \\
    Item 2        & \$15           & 2                 & \$30           \\
    Item 3        & \$20           & 1                 & \$20           \\
    \end{tabular}
    """

def _make_tab2_panel_b(df):
    # TODO: Make it look nice
    df = df.loc[df['in_sample_2']== 1]
    df = df.loc[(df['tplus3_UE_trans_3mon'].notna()& df['tplus3_percep_3mon'].notna())]
    df['userid'] = df.index.get_level_values(0)
    # Baseline
    base_formula = 'tplus3_UE_trans_3mon ~ tplus3_percep_3mon'
    mod1 = smf.wls(base_formula,data=df, weights=df['weight']).fit(cov_type='cluster', cov_kwds={'groups': df['userid']})
    mod1.summary()
    # Only Controls
    df['age2'] = df['age']**2
    controls = 'female + black + hispanic + age + age2 + r_asoth + other_race + education_2 + education_3 + education_4 + education_5 + education_6 + hhinc_2 + hhinc_3 + hhinc_4'
    dep_control = 'tplus3_UE_trans_3mon ~'+' + '+controls
    mod2 = smf.wls(dep_control,data=df, weights=df['weight']).fit(cov_type='cluster', cov_kwds={'groups': df['userid']})
    mod2.summary()
    # Var of Interest and Control
    lag = 'tplus3_UE_trans_3mon ~ find_job_3mon'
    mod3 = smf.wls(lag,data=df, weights=df['weight']).fit(cov_type='cluster', cov_kwds={'groups': df['userid']})
    mod3.summary()
    # Add interaction of LT unemployed
    lag_control = lag+' + '+controls
    mod4 = smf.wls(lag_control,data=df, weights=df['weight']).fit(cov_type='cluster', cov_kwds={'groups': df['userid']})
    mod4.summary()
    return mod1, mod2, mod3, mod4

def _make_tab4(depends_on):
    df = read_SCE(deps)
    df = restrict_sample(df)
    df = _prep_data_for_tab4(df)

def _prep_data_for_tab4(df):
    df = df.loc[df['in_sample_1']== 1]
    # Generate dummy variables for each year and month
    date_dummies = pd.get_dummies(pd.to_datetime(df.index.get_level_values(1)).strftime('%Y-%m'), prefix='dd')
    date_dummies = date_dummies.iloc[:, 1:]  # drop first dummy variable
    date_dummies.index = df.index
    # Add date_dummies to data
    df = df.join(date_dummies)

    # Create olf2 indicator variable
    df['olf2'] = (df['lfs'] == 3).astype(int)

    # Create emp3 indicator variable
    df['emp3'] = (df['lfs'] == 1).astype(int)

    # Create i3m indicator variable
    df['i3m'] = np.where(df['lfs'] != 1, df['find_job_3mon'].notnull().astype(int), np.nan)

    # Create next1lfs variable
    df = df.sort_values(['userid', 'date'])
    df['next1lfs'] = df.groupby('userid')['lfs'].shift(-1)
    df['first_unemp_survey'] = df.groupby('userid').cumcount() + 1

    # Create spelln and spellN variables
    df = df.sort_values(['spell_id', 'date'])
    df['spelln'] = df.groupby('spell_id').cumcount() + 1
    df['spellN'] = df.groupby('spell_id')['spell_id'].transform('count')

    # Create n_f3m_spell and N_f3m_spell variables
    df['n_f3m_spell'] = np.where(df['lfs'] != 1, df.groupby('spell_id')['i3m'].cumsum(), np.nan)
    df['N_f3m_spell'] = np.where(df['lfs'] != 1, df.groupby('spell_id')['n_f3m_spell'].transform('last'), np.nan)

    # Create n_olf_spell and N_olf_spell variables
    df['n_olf_spell'] = np.where(df['lfs'] != 1, df.groupby('spell_id')['olf2'].cumsum(), np.nan)
    df['N_olf_spell'] = np.where(df['lfs'] != 1, df.groupby('spell_id')['n_olf_spell'].transform('last'), np.nan)

    # Generate agesq variable
    df['age2'] = df['age'] ** 2
    df['userid'] = df.index.get_level_values('userid')
    df['date'] = df.index.get_level_values('date')
    return df

def _tab4_regressions(df):
    ## TODO: Standard Errors of mod 4 are wrong, else looks good
    # Grab relevant info and put it into a nice table
    df1 = df.loc[df['first_unemp_survey'] == 1]
    # Only first interview
    base_formula = 'find_job_3mon ~ udur'
    mod1 = smf.wls(base_formula,data=df1, weights=df1['weight']).fit(cov_type='cluster', cov_kwds={'groups': df1['userid']})
    mod1.summary()
    # Full Sample
    mod2 = smf.wls(base_formula,data=df, weights=df['weight']).fit(cov_type='cluster', cov_kwds={'groups': df['userid']})
    mod2.summary()
    # Add Controls
    controls = 'female + black + hispanic + age + age2 + r_asoth + other_race + education_2 + education_3 + education_4 + education_5 + education_6 + hhinc_2 + hhinc_3 + hhinc_4'
    form_control = base_formula+' + '+controls
    mod3 = smf.wls(form_control,data=df, weights=df['weight']).fit(cov_type='cluster', cov_kwds={'groups': df['userid']})
    mod3.summary()
    df_new = df.set_index(['userid','spell_id'])
    form_spells = base_formula+' + TimeEffects +EntityEffects'
    mod4 = PanelOLS.from_formula(form_spells, data=df_new,drop_absorbed=True, weights=df_new['weight']).fit(cov_type='clustered',cluster_time=True)
    return mod1, mod2, mod3, mod4
    
def _write_tab4(mod1,mod2,mod3,mod4,mod5,mod6,mod7,mod8):
    tab = rf"""
    \begin{{table}}[!htbp] \centering 
    \tiny
    \caption{{Linear Regressions of Elicitations on Unemployment Duration}}
    \label{{tab:realized_perc_table4}} 
    \begin{{tabular}}{{lcccccccc}}
    \toprule
    Dependent Variable: & & & & & & & & \\
    Elicited 3-Month Probability    & (1) & (2) & (3) & (4) & (5) & (6) & (7) & (8)\\
    \midrule
        Unemployment Duration,   & ${mod1.params.loc['udur']:.4f}^{{***}}$ & ${mod2.params.loc['udur']:.4f}^{{***}}$ & ${mod3.params.loc['udur']:.4f}^{{***}}$ & ${mod4.params.loc['udur']:.4f}^{{***}}$ &  ${mod5.params.loc['udur']:.4f}^{{***}}$& ${mod6.params.loc['udur']:.4f}^{{***}}$ & ${mod7.params.loc['udur']:.4f}$ & ${mod8.params.loc['udur']:.4f}$  \\
    in Months                          & ({mod1.bse.loc['udur']:.4f}) & ({mod2.bse.loc['udur']:.4f}) & ({mod3.bse.loc['udur']:.4f}) & ({mod4.bse.loc['udur']:.4f}) & ({mod5.bse.loc['udur']:.4f}) & ({mod6.bse.loc['udur']:.4f}) & ({mod7.std_errors.loc['udur']:.4f}) & ({mod8.std_errors.loc['udur']:.4f}) \\
    \midrule 
    Demographic Controls & & &  &  & x & x &  &  \\
    Spell Fixed Effects & & &  &  &  &  & x & x \\
    Observations & {mod1.nobs:.0f} & {mod2.nobs:.0f} & {mod3.nobs:.0f} & {mod4.nobs:.0f} & {mod5.nobs:.0f} & {mod6.nobs:.0f} & {mod7.nobs:.0f} & {mod8.nobs:.0f} \\
    $R^{{2}}$    & {mod1.rsquared:.3f} & {mod2.rsquared:.3f} & {mod3.rsquared:.3f} & {mod4.rsquared:.3f} & {mod5.rsquared:.3f} & {mod6.rsquared:.3f} & {mod7.rsquared:.3f} & {mod8.rsquared:.3f} \\     
    \hline
    \end{{tabular}}
    \end{{table}}
    """
    return tab
    
    