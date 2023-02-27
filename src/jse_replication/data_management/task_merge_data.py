"""Read in the SCE data, rename columns and harmonize values."""

import pytask
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from jse_replication.config import BLD,SRC
from jse_replication.reduced_form.helper_functions import read_auth_data

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

@pytask.mark.depends_on([
    SRC / "data" / "FRBNY-SCE-Public-Microdata-Complete-13-16.xlsx",
    SRC / "data" / "FRBNY-SCE-Public-Microdata-Complete-17-19.xlsx",
    ])
@pytask.mark.produces(BLD / "data" / "merged_SCE_data.csv")
def task_merge_SCE_data(depends_on,produces):
    """Reads and merges two Excel files, cleans and exports the data."""
    # Read the two Excel files into separate dataframes
    df1 = pd.read_excel(depends_on[0],header=1)
    df2 = pd.read_excel(depends_on[1],header=1)
    for df in [df1,df2]:
        # Save as proper datetime format
        df['date'] = df['date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m'))
        # Set index
        df.set_index(['userid','date'],inplace=True)
    # merge  data
    df = pd.concat([df1, df2])
    # Rename variables
    df = _rename_variables(df)
    # export data
    df.to_csv(produces)


@pytask.mark.depends_on([
    BLD / "data" / "merged_SCE_data.csv",
    SRC / "data" / "urjr.dta",
    SRC / "data" / "stateur.dta",
    SRC / "data" / "rgdp.dta"
    ])
@pytask.mark.produces(BLD / "data" / "new_SCE_data.csv")
def task_add_more_agg_data(depends_on,produces):

    df = _read_data(depends_on[0])
    df = _rename_variables(df)
    urjr, stateur, rgdp = _read_aux_files(depends_on)
    df = _merge_econ_data(df,urjr,stateur,rgdp)

    df = fill_in_missings(df)

    # Make sample code for quick access
    df = _sample_codes(df)

    # Fill with demographics from the first date of the respondents
    # Note: This is a little faulty since it does not update age or income etc.
    df = _merge_demographics(df)


    df.to_csv(produces)

depends_on = [
    BLD / "data" / "new_SCE_data.csv",
    BLD / "author_data" / "sce_datafile.dta"
    ]

@pytask.mark.depends_on(depends_on)
@pytask.mark.produces(BLD / "figures" / "data_comparison_over_time.png")
def task_comp_data(depends_on,produces):
    df = _read_data(depends_on[0])
    auth_df = read_auth_data(depends_on[1])
    real_count, auth_count = _data_comparison(df, auth_df)
    _visualize_data_discrepancies(real_count,auth_count,produces)

def _data_comparison(df, auth_df):
    auth_df['datem'] = auth_df['datem'].apply(lambda x: pd.to_datetime(str(x))).dt.strftime('%Y-%m')

    # Raw data appears to already exclude people without unemployment spell
    # Read raw data? =>not necessary since their datafile does not exclude obs
    assert (auth_df['total_unemp']==0).sum() == 0
    # Not the case for the raw data
    assert (df['total_unemp']==0).sum() > 0
    # Exclude people with no unemployment spells
    df = df.loc[(df['total_unemp']>0)]
    ## Problem we even have too many sometimes

    # Differences in dates
    A = set(auth_df['datem'].unique())
    B = set(df.index.get_level_values('date').unique())
    diff = A - B
    # They are able to include data until december 2012 whereas the available data dates back only to mid 2013

    real_count = df.reset_index().groupby('date')['userid'].count()
    auth_count = auth_df.reset_index().groupby('datem')['userid'].count()
    return real_count, auth_count

def _visualize_data_discrepancies(real_count,auth_count,produces):
    new_dates = []
    frequency = 12
    for date in auth_count.index:
        if date.endswith('12'):
            new_dates.append(date)
        else:
            new_dates.append('')
    fig, ax = plt.subplots()
    fig.supylabel('# of Respondents',fontsize = 11.0)
    fig.supxlabel('Date',fontsize = 11.0)
    ax.plot(auth_count,label ='Authors Data',color='#FF7F50')
    ax.plot(real_count,label='Own Data',color='#800000')
    #ax.set_xticklabels([x.strftime('%Y-%m') for x in xticks])
    ax.set_xticklabels(new_dates)
    ax.legend()
    plt.xticks(auth_count.index[::frequency], auth_count.index[::frequency])
    #plt.show()
    fig.savefig(produces)

def fill_in_missings(df):
    # LFS coding
    df = _gen_lfs_data(df)
    # Unemployment Spells
    df = _gen_spells(df)
    # Topcode self-reported unemployment by 60 month
    df = _clean_selfreported_unemployment(df)
    # Add more
    df = _gen_lfs_checks(df)
    # Compute unemployment duration
    df = _fill_in_missing_self_reports(df)
    # Add longterm unemployed dummy
    df = _gen_more_vars(df)
    # Generate dummies whether the UE transition was successful
    df = _ue_transitions(df)
    # Make dummies for unemployment duration
    df = _unemp_duration_dummies(df)
    df['find_job_3mon'] = df['find_job_3mon'].div(100)
    df['find_job_12mon'] = df['find_job_12mon'].div(100)
    # Unemployment transition within 3 month in 3 month
    df = _ue_3_mon_horizon_plus3(df)
    # Perception in 3 month from now
    df = _job_find_3mon_plus(df)

    return df


def _read_data(depends_on):
    df = pd.read_csv(depends_on)
    df.set_index(['userid','date'],inplace=True)
    return df

def _rename_variables(df):
    """Renames variables in the given dataframe and remove unnecessary columns.

    Args:
        df (DataFrame): The input dataframe to rename variables in.

    Returns:
        DataFrame: The output dataframe with renamed variables.

    """
    rename_dict = {
        'Q17new':'find_job_12mon',
        'Q18new': 'find_job_3mon',
        'Q22new': 'find_new_job_3mon',
        'Q6new': 'USstocks_higher',
        'Q4new': 'USunemployment_higher',
        'Q10_1': 'working_ft',
        'Q10_2': 'working_pt',
        'Q10_3': 'not_working_wouldlike',
        'Q10_4': 'temp_laid_off',
        'Q10_5': 'sick_leave',
        'Q15': 'looking_for_job',
        'Q16': 'unemployment_duration',
        'Q12new': 'selfemployed',
        '_STATE': 'state',
        'Q36': 'education',
        'Q47': 'hh_inc',
        '_EDU_CAT': 'edu_cat',
        '_HH_INC_CAT': 'hh_inc_cat',
        '_AGE_CAT': 'age_cat',
        'Q35_1': 'white',
        'Q35_2': 'black',
        'Q35_3': 'am_indian_nat_alaskan',
        'Q35_4': 'asian',
        'Q35_5': 'nat_hawaii',
        'Q35_6': 'other_race',
        'Q33': 'female',
        'Q34': 'hispanic',
        'Q32': 'age'
        }
    df = df.rename(columns = rename_dict)
    cols = list(rename_dict.values())+['tenure','weight','survey_date']
    df = df[cols]
    return df

def _read_aux_files(depends_on):
    """Reads auxiliary data files into separate dataframes.

   Args:
       depends_on (list): A list of file paths of the auxiliary data files to read.

   Returns:
       tuple: A tuple of three dataframes.

   """
    # BLS unemployment rate and job openings rate data (not seasonally adjusted, downloaded from BLS webpage)
    urjr = pd.read_stata(depends_on[1])
    # BLS state-level unemployment rate (not seasonally adjusted, downloaded from BLS webpage)
    stateur = pd.read_stata(depends_on[2])
    # Real GDP growth rate (quarterly frequency, annualized) and S&P500 Index (index value and quarterly growth rate) (downloaded from FRED webpage)
    rgdp = pd.read_stata(depends_on[3])
    return urjr, stateur, rgdp

def _merge_econ_data(df,urjr,stateur,rgdp):
    """Merges economic data with a given dataframe.

   Args:
       df (pandas.DataFrame): The dataframe to merge with.
       urjr (pandas.DataFrame): The dataframe containing BLS unemployment rate and job openings rate data.
       stateur (pandas.DataFrame): The dataframe containing BLS state-level unemployment rate data.
       rgdp (pandas.DataFrame): The dataframe containing real GDP growth rate and S&P500 index data.

   Returns:
       pandas.DataFrame: The merged dataframe.

   """
    df = df.reset_index()
    for aux in [df,urjr,stateur,rgdp]:
        aux['date'] =  aux['date'].apply(lambda x: pd.to_datetime(str(x))).dt.strftime('%Y-%m')
    df = pd.merge(df, urjr, on="date", how="inner")
    df = pd.merge(df, stateur, on=["date", "state"], how="inner")
    df = pd.merge(df, rgdp, on="date", how="inner")
    df = df.drop(columns=["statefips"])
    return df.set_index(['userid','date'])

def _gen_lfs_data(df):
    """Generates Labor Force Status (LFS) variable based on given columns of a dataframe.

    Args:
        df (pandas.DataFrame): The dataframe containing the columns necessary to generate the LFS variable.

    Returns:
        pandas.DataFrame: The updated dataframe with the new LFS variable.

    """
    # generate LFS variable
    cond1 = df[['working_ft','working_pt','sick_leave','selfemployed']].any(axis='columns')
    cond2 = ((df["temp_laid_off"] == 1) | ((df["not_working_wouldlike"] == 1) & (df["looking_for_job"] == 1)) & ~cond1)
    cond3 = (~cond1 & ~cond2)
    # LFS values
    values = [1, 2, 3]
    # Create the new LFS variable using numpy's select() function and the defined conditions and values
    df["lfs"] = np.select([cond1,cond2,cond3], values)
    return df

def _gen_spells(df):
    """
    Generates spell ID for non-employment spells in the input dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing LFS status for each user at each time period.

    Returns:
    --------
    pandas.DataFrame
        Returns a new dataframe with the same columns as the input dataframe, as well as a new column called `spell_id` that
        contains the spell ID for each non-employment spell for each user. The `spell_id` column is set to `pd.NA` for rows where the
        LFS status is 1, indicating that there is no non-employment spell associated with that row.
    """
    # generate spell id for nonemployment spells
    df= df.sort_index(level=["userid", "date"])
    change = ((df["lfs"].eq(2)) | (df["lfs"].eq(3))) & (df["lfs"].shift(1).eq(1))
    df["number"] = change.groupby("userid").transform('cumsum')
    df["spell_id"] = df.groupby(["number", "userid"]).ngroup()
    df["spell_id"] = df["spell_id"].where(df["lfs"] != 1, other=pd.NA)
    return df

def _clean_selfreported_unemployment(df):
    """Topdcode self-reported unemployment duration."""
    df['udur_self'] = df['unemployment_duration']
    df.loc[df["udur_self"] <= -1, "udur_self"] = pd.NA
    df['udur_self_top'] = df['udur_self']
    df.loc[df["udur_self"] > 60, "udur_self_top"] = 60
    return df

def _gen_lfs_checks(df):
    """
    Generate variables to identify labor force status changes for each user in the input dataframe.

    Args:
    - df: pandas dataframe containing survey responses for multiple users

    Returns:
    - df: pandas dataframe with the following additional columns:
        - 'olf': binary variable indicating if user is out of labor force in the current period
        - 'total_olf': cumulative sum of 'olf' for each user
        - 'emp': binary variable indicating if user is employed in the current period
        - 'total_emp': cumulative sum of 'emp' for each user
        - 'emp_so_far': cumulative sum of 'emp' up to the current period for each user
        - 'lag_emp': binary variable indicating if user was employed in the previous period
        - 'unemp': binary variable indicating if user is unemployed in the current period
        - 'total_unemp': cumulative sum of 'unemp' for each user
        - 'lag_unemp': binary variable indicating if user was unemployed in the previous period
        - 'x': a counter variable
        - 'obs': cumulative count of survey responses for each user
        - 'spell_obs': cumulative count of survey responses within each spell (i.e. consecutive observations of the same LFS status)
    """
    # sort by user ID and date of survey
    df = df.reset_index().sort_values(['userid', 'survey_date'])

    # test whether user is out of labor force
    df['olf'] = np.where(df['lfs'] == 3,1,0)
    df['total_olf'] = df.groupby('userid')['olf'].transform('sum')

    # test whether user is employed
    df['emp'] = np.where(df['lfs'] == 1,1,0)
    df['total_emp'] = df.groupby('userid')['emp'].transform('sum')
    df['emp_so_far'] = df.groupby('userid')['emp'].transform('cumsum')

    # test if user was employed or unemployed before current period
    df['lag_emp'] = df.groupby('userid')['emp'].shift(1)
    #df.loc[df.groupby('userid').cumcount() == 0, 'emp'] = np.nan

    df['unemp'] = np.where(df['lfs'] == 2,1,0)
    df['total_unemp'] = df.groupby('userid')['unemp'].transform('sum')
    df['lag_unemp'] = df.groupby('userid')['unemp'].shift(1)
    df.loc[df.groupby('userid').cumcount() == 0, 'unemp'] = np.nan
    # Count number of instances we have for each respondent
    df['x'] = 1
    df['obs'] = df.groupby('userid')['x'].transform('cumsum')
    df['spell_obs'] = df.groupby('spell_id')['x'].transform('cumsum')
    return df

def _fill_in_missing_self_reports(df):
    """
    Fills in missing values for self-reported duration data and computes the actual duration of employment spells.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe containing columns representing relevant variables such as survey date, spell id, user id,
        employment status, self-reported duration, etc.

    Returns:
    --------
    df : pandas.DataFrame
        The updated dataframe with additional columns for the day of survey, presence of self-reported data, cumulative
        sum of non-missing self-reported values, duration between survey dates, actual duration of employment spells,
        etc. The missing values for self-reported duration data are filled in based on certain conditions such as the
        first date at which we observe it, the date at which a person is not missing the self-reported value, etc. The
        negative values of actual duration are set to missing.
    """
    df['day_of_surv'] = (pd.to_datetime(df['survey_date']) - pd.Timestamp('1960-01-01')).dt.days.astype(int)

    # find whether there is self-reported duration data for that spell
    df['selfdata'] = df['udur_self_top'].notna()
    df['ever_selfdata'] = df.groupby('spell_id')['selfdata'].transform('max')

    # compute running sum of non-missing self-reported values
    df['nonmissing_so_far'] = df.groupby('userid')['selfdata'].transform('cumsum')

    # pull the self-reported value at the first date at which we observe it
    df['first_selfreport'] = df.groupby(['userid', 'spell_id'])['udur_self_top'].transform('first')
    #tag first date at which a person is not missing the self-reported value
    df["nonmissdate"] = np.where(df["first_selfreport"].notna(),df['day_of_surv'], np.nan)
    df['nonmissdate'] = df.groupby('userid')['nonmissdate'].transform('first')

    # Diff survey days
    df['diff'] = (df['day_of_surv'] - df.groupby('userid')['day_of_surv'].shift(1)) / 30.3
    # compute duration between survey dates in month
    df['udur'] = (df['day_of_surv'] - df.groupby('userid')['day_of_surv'].shift(1)) / 30.3
    # If we have a change only use half the measure (implicit assumption that the job finding prop is uniform distributed between the two dates)
    cond = (df['lfs'] != 1) & (df.groupby('userid')['lfs'].shift(1) == 1)
    df.loc[cond, 'udur'] = ((df['day_of_surv'] - df.groupby('userid')['day_of_surv'].shift(1)) / 30.3 / 2)
    df.loc[cond, 'diff'] = ((df['day_of_surv'] - df.groupby('userid')['day_of_surv'].shift(1)) / 30.3 / 2)

    # Remove if we have no data at all and no employment spell
    mask = (df['emp_so_far']==0) & (df['ever_selfdata'] == False)
    df.loc[mask, 'udur'] = np.nan
    # Also do if someone is employed
    df.loc[(df['emp']==1), 'udur'] = np.nan

    # add up running sum of duration for each userid
    df['add_time'] = df.groupby('userid')['diff'].transform('cumsum')
    df['spell_time'] = df.groupby('spell_id')['diff'].transform('cumsum')
    # Front fill for the self-reported
    df.loc[(df["lfs"] != 1) & (df['obs'] != 1) & df['emp_so_far']==0, "udur"] = df['first_selfreport'] + df['add_time']
    # Front fill for the one without self-report
    df.loc[(df["lfs"] != 1) & (df['emp_so_far']>0), "udur"] = df['spell_time']
    # Condition
    mask = (df['emp_so_far']==0) & (df['nonmissdate']==df['day_of_surv'])
    df.loc[mask, 'udur'] = df['udur_self_top']
    # back-fill the duration if there is self-reported data but no observed employment spell before that date
    df.loc[(df["ever_selfdata"] == 1) & (df["emp"] == 0) & (df["udur"].isna()), "udur"] = df['first_selfreport'] - ((df["nonmissdate"] - df["day_of_surv"]) / 30.3)

    # set negative values to missing
    df.loc[df["udur"] < 0, "udur"] = np.nan

    return df

def _gen_more_vars(df):
    """Generate dummy variables such as for longterm unemployment and the interaction with job finding perceptions."""
    # Long term unemployment dummy
    values = [np.nan,1,0]
    df['longterm_unemployed'] = np.select([df['udur'].isna(),df['udur']>6,df['udur']<=6], values)
    df['find_job_3mon_longterm'] = df['find_job_3mon'] * df['longterm_unemployed']
    df['find_job_12mon_longterm'] = df['find_job_3mon'] * df['longterm_unemployed']
    # Unemployment bins
    cut_labels = [1,2,3,4]
    cut_bins = [0, 3, 6, 12,np.inf]
    df['udur_bins'] = pd.cut(df['udur'],bins=cut_bins,labels=cut_labels,include_lowest=True)
    return df

def _ue_transitions(df):
    """
    Calculates employment transitions from the given dataframe for both 1-month and 3-month horizons.
    Adds two columns 'UE_trans_1mon' and 'UE_trans_3mon' to the dataframe indicating whether an individual 
    transitions from unemployment to employment within the specified time horizon. 
    
    Args:
        df (pandas.DataFrame): A pandas dataframe containing employment data. Must have the following columns:
            - 'userid': unique identifier for each individual
            - 'date': employment date in YYYY-MM-DD format
            - 'lfs': labor force status code, 2 represents unemployed, 1 represents employed, and 3 represents inactive.
    
    Returns:
        pandas.DataFrame: The original dataframe with two additional columns indicating employment transitions.
            'UE_trans_1mon': Indicates whether the individual transitioned from unemployment to employment within 
            the 1-month horizon.
            'UE_trans_3mon': Indicates whether the individual transitioned from unemployment to employment within 
            the 3-month horizon.
    """
    df['datem'] = pd.to_datetime(df['date'])
    df['datem'] = df['datem'].dt.to_period('M')

    # 1-month horizon U-to-E (Unemployment to Employment)
    df.sort_values(by=['userid', 'datem'], inplace=True)
    df['UE_trans_1mon'] = None

    # Not employed but employed in the next period
    mask1 = (df['lfs'] == 2) & (df.groupby('userid')['lfs'].shift(-1) == 1) & (df.groupby('userid')['datem'].shift(-1) <= (df['datem'] + 1))
    df.loc[mask1, 'UE_trans_1mon'] = 1
    # Not employed today and also not employed tommorow
    mask2 = (df['lfs'] == 2) & (df.groupby('userid')['lfs'].shift(-1).isin([2,3])) & (df.groupby('userid')['datem'].shift(-1) <= (df['datem'] + 1))
    df.loc[mask2, 'UE_trans_1mon'] = 0

    df = _ue_3_mon_horizon(df)

    return df

def _ue_3_mon_horizon(df):
    """
    Calculates a binary indicator variable "UE_trans_3mon" for each observation in the input DataFrame "df",
    which indicates whether the individual experienced a transition from unemployment to employment within a
    3-month horizon from the current observation.

    Args:
    - df (pandas.DataFrame): a DataFrame containing the input data, with the following columns:
        - 'userid': unique identifier of the individual
        - 'datem': month and year of the observation in YYYY-MM format
        - 'lfs': labor force status, encoded as an integer:
            - 1: employed
            - 2: unemployed
            - 3: not in labor force

    Returns:
    - pandas.DataFrame: the input DataFrame with an additional column 'UE_trans_3mon' that contains the binary
    indicator variable for each observation, indicating whether the individual experienced a transition from
    unemployment to employment within a 3-month horizon from the current observation.
    """
    df.sort_values(by=['userid', 'datem'], inplace=True)
    unemp = [2,3]
    # Don't have a job but found one in the next period which is within 3 month
    df.loc[(df['lfs'] == 2) & (df.groupby('userid')['lfs'].shift(-1) == 1) & \
         (df.groupby('userid')['datem'].shift(-1) <= (df['datem'] +3)), 'UE_trans_3mon'] = 1
    #  Don't have a job but found one in the next two periods which are within 3 month
    df.loc[(df['lfs'] == 2) & ((df.groupby('userid')['lfs'].shift(-2) == 1) | (df.groupby('userid')['lfs'].shift(-1) == 1)) & \
         (df.groupby('userid')['datem'].shift(-2) <= (df['datem'] + 3)) & \
         (df.groupby('userid')['lfs'].shift(-2).notnull()), 'UE_trans_3mon'] = 1

    df.loc[(df['lfs'] == 2) & ((df.groupby('userid')['lfs'].shift(-1) == 1) | (df.groupby('userid')['lfs'].shift(-2) == 1) | \
         (df.groupby('userid')['lfs'].shift(-3) == 1)) & \
         (df.groupby('userid')['datem'].shift(-3) <= (df['datem'] +3)) & \
         (df.groupby('userid')['lfs'].shift(-3).notnull()), 'UE_trans_3mon'] = 1

    df.loc[(df['lfs'] == 2) & (df.groupby('userid')['lfs'].shift(-1).isin(unemp)) & \
         (df.groupby('userid')['datem'].shift(-1) <= (df['datem'] + 3)) & \
         ((df.groupby('userid')['datem'].shift(-2) > (df['datem'] + 3)) | (df.groupby('userid')['lfs'].shift(-2).isnull())) \
         & (df.groupby('userid')['lfs'].shift(-1).notnull()), 'UE_trans_3mon'] = 0

    df.loc[(df['lfs'] == 2) & (df.groupby('userid')['lfs'].shift(-1).isin(unemp)) & (df.groupby('userid')['lfs'].shift(-2).isin(unemp)) & \
         (df.groupby('userid')['datem'].shift(-2) <= (df['datem'] + 3)) & \
         ((df.groupby('userid')['datem'].shift(-3) > (df['datem'] + 3)) | (df.groupby('userid')['lfs'].shift(-3).isnull())) \
         & (df.groupby('userid')['lfs'].shift(-2).notnull()), 'UE_trans_3mon'] = 0

    df.loc[(df['lfs'] == 2) & (df.groupby('userid')['lfs'].shift(-1).isin(unemp)) & (df.groupby('userid')['lfs'].shift(-2).isin(unemp)) & \
         (df.groupby('userid')['lfs'].shift(-3).isin(unemp)) & (df.groupby('userid')['datem'].shift(-3) <= (df['datem'] +3)) & \
         (df.groupby('userid')['lfs'].shift(-3).notnull()) & (df['lfs'].shift(-3).notnull()), 'UE_trans_3mon'] = 0

    return df

def _ue_3_mon_horizon_plus3(df):
    """Indicator for 3-month job finding in 3 months from now (conditional on not being employed in 3 months from now."""
    df.sort_values(by=['userid', 'datem'], inplace=True)
    unemp = [2,3]
    group = df.groupby('userid')
    # Don't have a job in three month but found one in the next period which is within 3 month
    df.loc[(group['lfs'].shift(-3).isin(unemp)) & (group['lfs'].shift(-4) == 1) & \
         (group['datem'].shift(-4) <= (df['datem'] +6)), 'tplus3_UE_trans_3mon'] = 1
    #  Don't have a job but found one in the next two periods which are within 3 month
    df.loc[(group['lfs'].shift(-3).isin(unemp)) & ((group['lfs'].shift(-5) == 1) | (group['lfs'].shift(-4) == 1)) & \
         (group['datem'].shift(-5) <= (df['datem'] + 6)) & \
         (group['lfs'].shift(-5).notnull()), 'tplus3_UE_trans_3mon'] = 1

    df.loc[(group['lfs'].shift(-3).isin(unemp)) & ((group['lfs'].shift(-4) == 1) | (group['lfs'].shift(-5) == 1) | \
         (group['lfs'].shift(-6) == 1)) & \
         (group['datem'].shift(-6) <= (df['datem'] +3)) & \
         (group['lfs'].shift(-6).notnull()), 'tplus3_UE_trans_3mon'] = 1

    df.loc[(group['lfs'].shift(-3).isin(unemp)) & (group['lfs'].shift(-4).isin(unemp)) & \
         (group['datem'].shift(-4) <= (df['datem'] + 6)) & \
         ((group['datem'].shift(-5) > (df['datem'] + 6)) | (group['lfs'].shift(-4).isnull())) \
         & (group['lfs'].shift(-4).notnull()), 'tplus3_UE_trans_3mon'] = 0

    df.loc[(group['lfs'].shift(-3).isin(unemp)) & (group['lfs'].shift(-4).isin(unemp)) & (group['lfs'].shift(-5).isin(unemp)) & \
         (group['datem'].shift(-5) <= (df['datem'] + 6)) & \
         ((group['datem'].shift(-6) > (df['datem'] + 6)) | (group['lfs'].shift(-6).isnull())) \
         & (group['lfs'].shift(-5).notnull()), 'tplus3_UE_trans_3mon'] = 0

    df.loc[(group['lfs'].shift(-3).isin(unemp)) & (group['lfs'].shift(-4).isin(unemp)) & (group['lfs'].shift(-5).isin(unemp)) & \
         (group['lfs'].shift(-6).isin(unemp)) & (group['datem'].shift(-6) <= (df['datem'] +6)) & \
         (group['lfs'].shift(-6).notnull()) & (df['lfs'].shift(-6).notnull()), 'tplus3_UE_trans_3mon'] = 0
    return df

def _job_find_3mon_plus(df):
    """Shift the elicited belief back from the future."""
    df.sort_values(by=['userid', 'datem'], inplace=True)
    df['tplus1_percep_3mon'] = df.groupby('userid')['find_job_3mon'].shift(-1)
    df['tplus3_percep_3mon'] = df.groupby('userid')['find_job_3mon'].shift(-3)
    # Does not work yer
    df['tplus3'] = df['datem']+3
    return df


def _unemp_duration_dummies(df):
    """Create dummies for unemployment duration up to one year."""
    bins = [0,1,2,3,4,5,6,7,8,9,10,11,12]
    labels = [0,1,2,3,4,5,6,7,8,9,10,11]
    df['nedur_1mo_'] = pd.cut(df['udur'],bins=bins,labels=labels)
    df = pd.concat([df, pd.get_dummies(df['nedur_1mo_'],prefix='nedur_1mo')], axis=1)
    return df

def _observe_3survs_within_3(df):
    ## TODO: Does not work
    df.sort_values(by=['userid', 'datem'], inplace=True)
    unemp = [2,3]
    df['UE_trans_3mon_3sur'] = np.nan
    group = df.groupby('userid')
    ## Why do we impose that we observe the lfs in 3 months from now?
    df.loc[((df['lfs']==2) & ((group['lfs'].shift(-1) == 1) | (group['lfs'].shift(-2) == 1) | \
         (group['lfs'].shift(-3) == 1)) & \
         (group['datem'].shift(-3) <= (df['datem'] +3)) & \
         (group['lfs'].shift(-3).notnull())), 'UE_trans_3mon_3sur'] = 1
    df.loc[((df['lfs']==2) & (group['lfs'].shift(-1).isin(unemp) & (group['lfs'].shift(-2).isin(unemp)) & \
         (group['lfs'].shift(-3).isin(unemp))) & \
         (group['datem'].shift(-3) <= (df['datem'] +3)) & \
         (group['lfs'].shift(-3).notnull())), 'UE_trans_3mon_3sur'] = 0
    return df

def _merge_demographics(df):
    """
    Merge demographic information with the main dataset.

    Args:
        df (pd.DataFrame): A Pandas DataFrame containing the main dataset.

    Returns:
        pd.DataFrame: A Pandas DataFrame with demographic information merged with the main dataset.
    """
    # Only go to the first observation
    df_first = df.sort_index(level=["userid", "date"]).groupby('userid').first()
    # Make categories then dummies
    df_first['new_edu_cat'] = pd.cut(df_first['education'],bins=[0,1,2,3,5,8,9],labels=[1,2,3,4,5,6])
    df_first = pd.concat([df_first,pd.get_dummies(df_first['new_edu_cat'].astype('Int64'),prefix='education')],axis=1)
    df_first['new_hhinc_cat'] = pd.cut(df_first['hh_inc'],bins=[0,3,6,8,11],labels=[1,2,3,4])
    df_first = pd.concat([df_first,pd.get_dummies(df_first['new_hhinc_cat'].astype('Int64'),prefix='hhinc')],axis=1)
    # Reset index to userid and merge m:1 back to the original dataframe
    # OR use fill na (or something like that)
    df_first['r_asoth'] = np.where((df_first['am_indian_nat_alaskan']==1)|(df_first['asian']==1)|(df_first['nat_hawaii']==1),1,0)
    for col in ['female','black','hispanic']:
        df_first[col].replace(2,0,inplace=True)
    # All relevant columns
    controls = ['female','black','hispanic','age','r_asoth','other_race','education_1','education_2','education_3','education_4','education_5','education_6','hhinc_1','hhinc_2','hhinc_3','hhinc_4']
    df_first = df_first[controls]
    df = df[[col for col in df.columns if col not in controls]]
    df = df.reset_index().set_index('userid').join(df_first,validate='m:1').reset_index().set_index(['userid','date'])
    return df

def _sample_codes(df):
    """Creates dummies for different sample restriction for use later."""
    df = _observe_3survs_within_3(df)
    # No clear violation and bothj elicitations exist
    condition1 = ((df['find_job_3mon']<=df['find_job_12mon'])&(df['find_job_3mon'].notna() | df['find_job_12mon'].notna()) & df['weight'].notna())
    df['in_sample_1'] = np.where(condition1,1,0)
    # Additionally we observe three surveys within the next three month
    condition2 = ((df['find_job_3mon']<=df['find_job_12mon'])&(df['find_job_3mon'].notna() | df['find_job_12mon'].notna()) & df['weight'].notna()) & (df['UE_trans_3mon_3sur'].notna())
    df['in_sample_2'] = np.where(condition2,1,0)
    # Observe atleast one abstract from rationality constraint
    condition3 = ((df['find_job_3mon'].notna() | df['find_job_12mon'].notna()) & df['weight'].notna())
    df['in_sample_3'] = np.where(condition3,1,0)
    # Observe atleast one abstract from rationality constraint have all three consecutive surveys
    condition4 = ((df['find_job_3mon'].notna() | df['find_job_12mon'].notna()) & df['weight'].notna() & df['UE_trans_3mon_3sur'].notna())
    df['in_sample_4'] = np.where(condition4,1,0)
    return df
