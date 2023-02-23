import pandas as pd 
import pytask
import matplotlib.pyplot as plt
import numpy as np
#from reduced_form.weighted_moments import *
from jse_replication.reduced_form.weighted_moments import *
from jse_replication.reduced_form.helper_functions import *
from scipy import stats

#deps = "../bld/author_data/sce_datafile.dta"
#df = read_SCE(deps)
#df = restrict_sample(df)  


from jse_replication.config import BLD,SRC

@pytask.mark.depends_on(BLD / "author_data" / "sce_datafile.dta")
@pytask.mark.produces(BLD / "figures" / "histogram_perceptions_figure_1.png")
def task_make_fig1(depends_on,produces):
    df = prep_data(depends_on)
    _make_fig1(df,produces)
    
@pytask.mark.depends_on(BLD / "author_data" / "sce_datafile.dta")
@pytask.mark.produces(BLD / "figures" / "jf3mon_per_percbin_figure_2.png")
def task_make_fig2(depends_on,produces):
    df = prep_data(depends_on)
    _make_fig2(df,produces)
   
    
@pytask.mark.depends_on(BLD / "author_data" / "sce_datafile.dta")
@pytask.mark.produces(BLD / "figures" / "perc_vs_real_jf3mon_figure_3.png")
def task_make_fig3(depends_on,produces):
    df = prep_data(depends_on)
    _make_fig3(df,produces) 


def _make_fig1(df,produces): 
    df = _make_perc_bins(df)
    bin_weight = df.groupby('find_job_3mon_bins')['weight'].sum()
    all_weights = bin_weight.sum()
    fig, ax = plt.subplots()
    fig.supylabel('Relative Frequency (in %)',fontsize = 12.0)
    fig.supxlabel('Elicited Prob(Find Job in 3 Month)',fontsize = 12.0)
    ax.hist(bin_weight.index, weights=(bin_weight/all_weights)*100,color='#FF7F50')
    ax.yaxis.set_label_coords(-0.3, 0.5)
    # Save it
    fig.savefig(produces)
   
def _make_fig2(df,produces):
    df = _make_perc_bins(df)
    df_fig = df.loc[df['in_sample_2']== 1]
    # Weighted mean
    w_mean = df_fig.groupby('find_job_3mon_bins').apply(lambda x: np.average(x['UE_trans_3mon'], weights=x.loc[x.index,'weight']))
    # Weighted standard deviation
    w_std = df_fig.groupby('find_job_3mon_bins').apply(weighted_sd,'UE_trans_3mon')
    n = df_fig.groupby('find_job_3mon_bins')['UE_trans_3mon'].count()
    # get t-value for the respective degress of freedom in each bin
    t_perc = n.apply(lambda x: stats.t(df=x-1).ppf((0.975)))
    # Compute upper and lower bound of the confidence interval at 95%
    lower_conf = w_mean - (w_std/np.sqrt(n))*t_perc
    upper_conf = w_mean + (w_std/np.sqrt(n))*t_perc
    # Width of the confidence interval
    conf_band = upper_conf-lower_conf
    # Make plot
    fig, ax = plt.subplots()
    fig.supylabel('Realized 3-Month U-E transition rate',fontsize = 12.0)
    fig.supxlabel('Elicited Prob(Find Job in 3 Month)',fontsize = 12.0)
    # 45 degree line
    ax.plot(np.arange(-0.1,1.1,0.1),np.arange(-0.1,1.1,0.1),color='grey',label='Rational Expectation')
    # Plot
    ax.errorbar(w_mean.index,w_mean,yerr=(w_std/np.sqrt(n))*t_perc,fmt='o',color='#FF7F50',capsize=4, label='Mean of Realized 3−Month U−E Transition Rate')
    # Make it look nicer
    ax.set_yticks(np.arange(0,1.1,0.1))
    ax.set_xticks(np.arange(0,1.1,0.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc=(0.1,-.35))
    # Save it
    fig.savefig(produces)
    
def _make_fig3(df,produces):
    ## TODO: Some discrepancies in the Confidence Intervals -> Check Footnote 29
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
    fig.supylabel('Prob(Find Job in 3 Month)',fontsize = 12.0)
    fig.supxlabel('Length of Unemployment Spell',fontsize = 12.0)
    # Plot Perceived job finding rate
    ax.errorbar(perc_w_mean.index,perc_w_mean,yerr=perc_err,fmt='-o',color='#FF7F50',capsize=4, label='Perceived Job Finding Rate')
    ax.errorbar((real_w_mean.index)+0.1,real_w_mean,yerr=real_err,fmt='-o',color='#2e86c1',capsize=4, label='Realized Job Finding Rate')
    # Make it look nicer
    ax.legend(loc=(0,-.25),ncol=2)
    # Save it
    fig.savefig(produces)
    
    
def _make_perc_bins(df):
    bins = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.1]
    labels = [0.05,.15,.25,.35,.45,.55,.65,.75,.85,.95]
    df['find_job_3mon_bins'] = pd.cut(df['find_job_3mon'],bins=bins,labels=labels,right=False,include_lowest=True)
    return df 

