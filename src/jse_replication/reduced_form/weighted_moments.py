import pandas as pd
import numpy as np

## Compute sample descriptives with weights

def weighted_mean(df, variable):
    """
    Compute weighted sample mean for a variable in a Pandas DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame with a column named `weight`.
    variable : str
        The name of the column to compute moments for.

    Returns:
    --------
    float
        A list of the first four weighted moments: [mean, variance, skewness, kurtosis].
    """

    # Get the weighted mean
    w_mean = np.average(df[variable], weights=df['weight'])

    return w_mean

def w_var(df, val):
    """
    Compute weighted sample variance for a variable in a Pandas DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame with a column named `weight`.
    variable : str
        The name of the column to compute moments for.

    Returns:
    --------
    float
        A list of the first four weighted moments: [mean, variance, skewness, kurtosis].
    """
    # Kick out missing values
    df = df.loc[df[val].notna()]
    rel_weight = df['weight']/df['weight'].sum()
    vals = df[val]
    weighted_avg = np.average(vals, weights=rel_weight)
    
    num = np.sum(rel_weight * (vals - weighted_avg)**2)
    denum = ((vals.count()-1)/vals.count())*np.sum(rel_weight)
    return (num/denum)

def w_m(x, w):
    """
    Compute weighted sample mean for a variable in a Pandas DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame with a column named `weight`.
    variable : str
        The name of the column to compute moments for.

    Returns:
    --------
    float
        A list of the first four weighted moments: [mean, variance, skewness, kurtosis].
    """

    # Get the weighted mean
    w_mean = np.average(x, weights=w)

    return w_mean

def w_cov(df,x, y):
    """Weighted Covariance with N-1 degrees of freedom correction"""
    # Kick out missing values
    df = df.loc[df[x].notna() & df[y].notna()]
    w = df['weight']/df['weight'].sum()
    x = df[x]
    y = df[y]
    num = np.sum(w * (x - w_m(x, w)) * (y - w_m(y, w)))
    denum = ((x.count()-1)/x.count())*np.sum(w)
    return (num/denum)

def weighted_median(df, variable):
    """
    Compute the weighted median for a variable in a Pandas DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame with a column named `weight`.
    variable : str
        The name of the column to compute the weighted median for.

    Returns:
    --------
    float
        The weighted median.
    """

    # Sort the data by the variable
    sorted_data = df.sort_values(variable)

    # Compute the cumulative weights
    cum_weights = np.cumsum(sorted_data['weight'])

    # Find the index of the median value
    median_idx = np.searchsorted(cum_weights, 0.5 * cum_weights.iloc[-1])

    # Get the value at the median index
    median_value = sorted_data.iloc[median_idx][variable]

    return median_value

def weighted_value_counts(df, col):
    """
    Returns a weighted value count of a pandas DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        A pandas DataFrame to count the values of.
    col : str
        The name of the column in `df` that contains the series to count the values of.
    
    Returns:
    --------
    pandas.Series
        A pandas series containing the weighted value counts of `col`.
    """
    # Calculate the weighted counts by multiplying the counts by the weights
    weighted_counts = df.groupby(col)['weight'].sum()
    all_weights = weighted_counts.sum()
    
    return weighted_counts / all_weights

def group_weighted_mean(df, col):
    """Calculate the weighted mean for a DataFrame"""
    numerator = (df[col] * df['weight']).sum()
    denominator = df['weight'].sum()
    return numerator / denominator

def weighted_sd(df,val):
    rel_weight = df['weight']/df['weight'].sum()
    vals = df[val]
    weighted_avg = np.average(vals, weights=rel_weight)
    
    num = np.sum(rel_weight * (vals - weighted_avg)**2)
    denum = ((vals.count()-1)/vals.count())*np.sum(rel_weight)
    return np.sqrt(num/denum)


