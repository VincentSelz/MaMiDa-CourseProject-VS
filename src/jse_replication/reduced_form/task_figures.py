import pandas as pd
import pytask
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#from reduced_form.weighted_moments import *
from jse_replication.reduced_form.weighted_moments import *
from jse_replication.reduced_form.helper_functions import *
from scipy import stats

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')


from jse_replication.config import BLD,SRC

@pytask.mark.depends_on(
    [BLD / "author_data" / "sce_datafile.dta",
     BLD / "data" / "new_sce_data.csv"])
@pytask.mark.produces(BLD / "figures" / "histogram_perceptions_figure_1.png")
def task_make_fig1(depends_on,produces):
    auth_df = prep_auth_data(depends_on[0])
    df = prep_SCE_data(depends_on[1])
    _make_fig1(auth_df,df,produces)

@pytask.mark.depends_on(
    [BLD / "author_data" / "sce_datafile.dta",
     BLD / "data" / "new_sce_data.csv"])
@pytask.mark.produces(BLD / "figures" / "jf3mon_per_percbin_figure_2.png")
def task_make_fig2(depends_on,produces):
    auth_df = prep_auth_data(depends_on[0])
    df = prep_SCE_data(depends_on[1])
    _make_fig2(auth_df,df,produces)


@pytask.mark.depends_on(
    [BLD / "author_data" / "sce_datafile.dta",
     BLD / "data" / "new_sce_data.csv"])
@pytask.mark.produces(
    [BLD / "figures" / "perc_vs_real_jf3mon_figure_3_panel_a.png",
     BLD / "figures" / "perc_vs_real_jf3mon_figure_3_panel_b.png"])
def task_make_fig3(depends_on,produces):
    auth_df = prep_auth_data(depends_on[0])
    df = prep_SCE_data(depends_on[1])
    _make_fig3_panel_a(auth_df,produces[0])
    _make_fig3_panel_b(df,produces[1])


def _make_fig1(auth_df,df,produces):
    matplotlib.rcParams['xtick.labelsize'] = 20
    matplotlib.rcParams['ytick.labelsize'] = 20
    for data in [auth_df,df]:
        data = _make_perc_bins(data)
    auth_bin_weight = auth_df.groupby('find_job_3mon_bins')['weight'].sum()
    auth_all_weights = auth_bin_weight.sum()
    bin_weight = df.groupby('find_job_3mon_bins')['weight'].sum()
    all_weights = bin_weight.sum()
    fig, axs = plt.subplots(ncols=2,figsize=(25,10))
    fig.supylabel('Relative Frequency (in %)',fontsize = 30.0)
    fig.supxlabel('Elicited Prob(Find Job in 3 Month)',fontsize = 30.0)
    axs[0].hist(auth_bin_weight.index, weights=(auth_bin_weight/auth_all_weights)*100,color='#FF7F50')
    axs[0].set_title('Author Data',fontsize=35)
    axs[1].hist(bin_weight.index, weights=(bin_weight/all_weights)*100,color='#800000')
    axs[1].yaxis.set_label_coords(-0.15, 0.5)
    axs[1].set_title('Own Data',fontsize=35)
    # Save it
    fig.savefig(produces)

def _make_fig2(auth_df,df,produces):
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['ytick.labelsize'] = 10
    df = _make_perc_bins(df)
    df_fig = df.loc[df['in_sample_2']== 1]
    # Weighted mean
    w_mean = df_fig.groupby('find_job_3mon_bins').apply(lambda x: np.average(x['UE_trans_3mon'], weights=x.loc[x.index,'weight']))
    # Weighted standard deviation
    w_std = df_fig.groupby('find_job_3mon_bins').apply(weighted_sd,'UE_trans_3mon')
    n = df_fig.groupby('find_job_3mon_bins')['UE_trans_3mon'].count()
    # get t-value for the respective degress of freedom in each bin
    t_perc = n.apply(lambda x: stats.t(df=x-1).ppf((0.975)))
    # Author data
    auth_df = _make_perc_bins(auth_df)
    df_fig2 = auth_df.loc[auth_df['in_sample_2']== 1]
    # Weighted mean
    w_mean2 = df_fig2.groupby('find_job_3mon_bins').apply(lambda x: np.average(x['UE_trans_3mon'], weights=x.loc[x.index,'weight']))
    # Weighted standard deviation
    w_std2 = df_fig2.groupby('find_job_3mon_bins').apply(weighted_sd,'UE_trans_3mon')
    n2 = df_fig2.groupby('find_job_3mon_bins')['UE_trans_3mon'].count()
    # get t-value for the respective degress of freedom in each bin
    t_perc2 = n.apply(lambda x: stats.t(df=x-1).ppf((0.975)))

    # Make plot
    fig, ax = plt.subplots()
    fig.supylabel('Realized 3-Month U-E transition rate',fontsize = 12.0)
    fig.supxlabel('Elicited Prob(Find Job in 3 Month)',fontsize = 12.0)
    # 45 degree line
    ax.plot(np.arange(-0.1,1.1,0.1),np.arange(-0.1,1.1,0.1),color='grey',label='Rational Expectation')
    # Plot auth data
    ax.errorbar(w_mean2.index,w_mean2,yerr=(w_std2/np.sqrt(n2))*t_perc2,fmt='o',color='#FF7F50',capsize=4, label='Author Data')
    # Plot own data
    ax.errorbar(pd.Series(w_mean.index).astype(float)+0.01,w_mean,yerr=(w_std/np.sqrt(n))*t_perc,fmt='o',color='#800000',capsize=4, label='Own Data')
    # Make it look nicer
    ax.set_yticks(np.arange(0,1.1,0.1))
    ax.set_xticks(np.arange(0,1.1,0.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left')
    # Save it
    fig.savefig(produces)

def _make_fig3_panel_a(df,produces):
    matplotlib.rcParams['xtick.labelsize'] = 12
    matplotlib.rcParams['ytick.labelsize'] = 12
    ## Also clean up the x axis
    df_fig = df.loc[df['in_sample_2']== 1]
    # Weighted mean
    perc_w_mean = df_fig.groupby('udur_bins').apply(lambda x: np.average(x['find_job_3mon'], weights=x.loc[x.index,'weight']))
    # Weighted standard deviation
    perc_w_std = df_fig.groupby('udur_bins').apply(weighted_sd,'find_job_3mon')
    perc_n = df_fig.groupby('udur_bins')['find_job_3mon'].count()
    # get t-value for the respective degress of freedom in each bin
    perc_t = perc_n.apply(lambda x: stats.t(df=x-1).ppf((0.975)))
    perc_err = (perc_w_std/np.sqrt(perc_n))*perc_t


    real_w_mean = df_fig.groupby('udur_bins').apply(lambda x: np.average(x['UE_trans_3mon'], weights=x.loc[x.index,'weight']))
    # Weighted standard deviation
    real_w_std = df_fig.groupby('udur_bins').apply(weighted_sd,'UE_trans_3mon')
    real_n = df_fig.groupby('udur_bins')['UE_trans_3mon'].count()
    # get t-value for the respective degress of freedom in each bin
    real_t = real_n.apply(lambda x: stats.t(df=x-1).ppf((0.975)))
    real_err = (real_w_std/np.sqrt(real_n))*real_t

    fig, ax = plt.subplots()
    fig.supylabel('Prob(Find Job in 3 Month)',fontsize = 15.0)
    fig.supxlabel('Length of Unemployment Spell',fontsize = 15.0)
    # Plot Perceived job finding rate
    ax.errorbar(perc_w_mean.index,perc_w_mean,yerr=perc_err,fmt='-o',color='#FF7F50',capsize=4, label='Perceived Job Finding Rate')
    ax.errorbar((real_w_mean.index)+0.1,real_w_mean,yerr=real_err,fmt='-o',color='grey',capsize=4, label='Realized Job Finding Rate')
    ax.set_xticks(range(1,5),['(0-3) Months','(3-6) Months','(7-12) Months','13+ Months'])
    # Make it look nicer
    ax.legend(loc='lower left')
    # Save it
    fig.savefig(produces)

def _make_fig3_panel_b(df,produces):
    matplotlib.rcParams['xtick.labelsize'] = 12
    matplotlib.rcParams['ytick.labelsize'] = 12
    ## Also clean up the x axis
    df_fig = df.loc[df['in_sample_2']== 1]
    # Weighted mean
    perc_w_mean = df_fig.groupby('udur_bins').apply(lambda x: np.average(x['find_job_3mon'], weights=x.loc[x.index,'weight']))
    # Weighted standard deviation
    perc_w_std = df_fig.groupby('udur_bins').apply(weighted_sd,'find_job_3mon')
    perc_n = df_fig.groupby('udur_bins')['find_job_3mon'].count()
    # get t-value for the respective degress of freedom in each bin
    perc_t = perc_n.apply(lambda x: stats.t(df=x-1).ppf((0.975)))
    perc_err = (perc_w_std/np.sqrt(perc_n))*perc_t


    real_w_mean = df_fig.groupby('udur_bins').apply(lambda x: np.average(x['UE_trans_3mon'], weights=x.loc[x.index,'weight']))
    # Weighted standard deviation
    real_w_std = df_fig.groupby('udur_bins').apply(weighted_sd,'UE_trans_3mon')
    real_n = df_fig.groupby('udur_bins')['UE_trans_3mon'].count()
    # get t-value for the respective degress of freedom in each bin
    real_t = real_n.apply(lambda x: stats.t(df=x-1).ppf((0.975)))
    real_err = (real_w_std/np.sqrt(real_n))*real_t

    fig, ax = plt.subplots()
    fig.supylabel('Prob(Find Job in 3 Month)',fontsize = 15.0)
    fig.supxlabel('Length of Unemployment Spell',fontsize = 15.0)
    # Plot Perceived job finding rate
    ax.errorbar(perc_w_mean.index,perc_w_mean,yerr=perc_err,fmt='-o',color='#800000',capsize=4, label='Perceived Job Finding Rate')
    ax.errorbar((real_w_mean.index)+0.1,real_w_mean,yerr=real_err,fmt='-o',color='grey',capsize=4, label='Realized Job Finding Rate')
    ax.set_xticks(range(1,5),['(0-3) Months','(3-6) Months','(7-12) Months','13+ Months'])
    # Make it look nicer
    ax.legend(loc='lower left')
    # Save it
    fig.savefig(produces)


def _make_perc_bins(df):
    bins = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.1]
    labels = [0.05,.15,.25,.35,.45,.55,.65,.75,.85,.95]
    df['find_job_3mon_bins'] = pd.cut(df['find_job_3mon'],bins=bins,labels=labels,right=False,include_lowest=True)
    return df
