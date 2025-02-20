import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
from matplotlib.patches import Patch
import scipy.stats as stats
import random
import traceback
from scipy.stats import chi2
from abc_functions import *
import matplotlib.patches as patches
from scipy.stats import gaussian_kde


@njit
def numba_observed_vafs_alpha_vaf_array(pur,S,C,cov,binss,min_reads,xdata,y_prob):
    cov_C, alt_C = clonal(pur, C, cov, min_reads)
    cov_S, alt_S = subclonal(pur, S, cov, min_reads, xdata, y_prob)
    vaf = np.empty(len(cov_C) + len(cov_S)) 
    for i in range(len(cov_C)):
        vaf[i] = alt_C[i] / cov_C[i]
    for i in range(len(cov_C), len(cov_C) + len(cov_S)):
        vaf[i] = alt_S[i - len(cov_C)] / cov_S[i - len(cov_C)] 
    return vaf


def longest_non_zero_underestimation(vaf2,lower_vaf):

    longest_sequence = 0
    current_sequence = 1

    indices = np.where((vaf2 != 0) & (lower_vaf == 0))[0]
    for i in range(1, len(indices)):
        if indices[i] == indices[i-1] + 1:
            current_sequence += 1
        else:
            if current_sequence > longest_sequence:
                longest_sequence = current_sequence
            current_sequence = 1

    if current_sequence > longest_sequence:
        longest_sequence = current_sequence
    return longest_sequence


def longest_false_sequence(within_limits):
    max_false_len = 0
    current_false_len = 0

    for value in within_limits:
        if value == False:
            current_false_len += 1
            if current_false_len > max_false_len:
                max_false_len = current_false_len
        else:
            current_false_len = 0

    return max_false_len



df=pd.read_csv('../results/abc_results.txt')
plots_folderpath='../results/vafs_predicted/'
scored_results_folderpath='../results/abc_scored.csv'
classified_scored_results_folderpath='../results/abc_scored_classified.csv'

ngauss_mean_error_variance_exp = []
gauss_mean_error_variance_exp = []
zeros_predicted_exp = []
zeros_observed_exp = []
lower_score_exp = []
upper_score_exp = []
total_score_exp = []

gauss_absolute_exp = []
ngauss_absolute_exp = []

ngauss_mean_error_variance_wf = []
gauss_mean_error_variance_wf = []
zeros_predicted_wf = []
zeros_observed_wf = []
lower_score_wf = []
upper_score_wf = []
total_score_wf = []

gauss_absolute_wf = []
ngauss_absolute_wf = []

longest_false_estimation_exp=[]
longest_nonzero_exp=[]
longest_false_estimation_wf=[]
longest_nonzero_wf=[]
exp_vaf_first_peak=[]
wf_vaf_first_peak=[]
real_vaf_first_peak=[]

exp_peak_count=[]
wf_peak_count=[]
real_peak_count=[]

wf_peak_difference=[]
exp_peak_difference=[]

rdonors=[]
rcohorts=[]

#df.drop_duplicates(subset="donor", keep="first", inplace=True)#in case you did accidentally two predictions on one donor
used_df=df

for donor in used_df.donor.unique():
    try:

        donor_df=used_df[used_df['donor']==donor]

        pred_purity_wf,pred_purity_exp=donor_df.pred_purity_wf.item(),donor_df.pred_purity_exp.item()
        pred_Sexp0,pred_Swf0=donor_df.pred_S_exp0.item(),donor_df.pred_S_wf0.item()
        pred_C_wf,pred_C_exp=donor_df.pred_C_wf.item(),donor_df.pred_C_exp.item()

        vaf_donor=donor

        for cohort in ['MC3', 'PCAWG', 'HMF']:
            file_path = '../data_filtered/' +cohort+ '/'+vaf_donor + '.bed.gz'
            if os.path.isfile(file_path):
                break

        whole_df= pd.read_csv(file_path, sep = ',', compression ='gzip')
        vaf=whole_df['VAF']
        observed_hist=np.histogram(vaf, bins = 100,range=(0,1))[0]
        cov=whole_df['coverage']
        cov=np.array(cov)
        cov_val=np.mean(cov)
        cov_str=str(np.round(np.mean(cov)))
        measure_cov=cov_str
        min_reads=whole_df["AD_ALT"].min() 

        score_wf0=donor_df.score_wf0.item()
        score_exp0=donor_df.score_exp0.item()

        #################################################
        #Find peaks for REAL
        # estimate the density function
        density = gaussian_kde(list(vaf), bw_method='scott')
        # evaluate the density in 1000 points (arbitrary)
        freq_range = np.linspace(min(vaf), max(vaf), 1000)
        dens_range = density(freq_range)
        # calculate the differences of adjacent points
        diff_y = np.diff(dens_range)
        # find the peaks
        peak_indices = np.where((diff_y[:-1] > 0) & (diff_y[1:] < 0))[0] + 1 
        freq_peak = freq_range[peak_indices]
        dens_peak = dens_range[peak_indices]
        # find the valleys
        valley_indices = np.where((diff_y[:-1] < 0) & (diff_y[1:] > 0))[0] + 1 
        valley_freq = freq_range[valley_indices]
        low_thresh = 0.01 # Threshold to remove peaks with low density
        # remove peaks with a very low density (noise): 1% of the highest peak
        freq_peak = freq_peak[dens_peak > low_thresh*max(dens_peak)]
        dens_peak = dens_peak[dens_peak > low_thresh*max(dens_peak)]
        real_pc=len(freq_peak)
        real_vaffp=freq_peak[0]
        real_freq_range=freq_range
        real_dens_range=dens_range
        real_freq_peak=freq_peak
        real_dens_peak=dens_peak
        orignial_vaf=vaf
        ###########################################
        alpha=1
        discr_cov_mean=1000
        discretization_cov=np.array([discr_cov_mean])
        lowest_frequency_allowed=1/np.max(discretization_cov)
        nr_of_bins=int(1/lowest_frequency_allowed)
        xdata_wf=np.linspace(lowest_frequency_allowed,1,nr_of_bins+1)
        y_model = f_alpha(xdata_wf, 1.0,alpha)
        y_prob_wf=y_model/np.sum(y_model)
        alpha=2
        exp_discr_cov_mean=np.max([int(np.max(cov)*1.05),100])
        lowest_frequency_allowed=1/np.max(exp_discr_cov_mean)
        nr_of_bins=int(1/lowest_frequency_allowed)
        xdata_exp=np.linspace(lowest_frequency_allowed,1,nr_of_bins+1)
        y_model = f_alpha(xdata_exp, 1.0,alpha)
        y_prob_exp=y_model/np.sum(y_model)
        binss = np.linspace(0, 1, 101)
        x_range_scatter=binss[1:]
        model="WF"


        gauss_absolute=0.0
        ngauss_absolute=0.0
        ngauss_binwise=np.zeros(100)
        gauss_binwise=np.zeros(100)
        
        used_vafs=[]
        nr_of_repeats=10 #number of times we average over simulations
        vaf2=observed_hist
        vaf2n=vaf2/sum(vaf2)
        for repeat in range(nr_of_repeats):    
            vaf1=simulated_vaf(pred_purity_wf,pred_Swf0,pred_C_wf,cov,binss,min_reads,xdata_wf,y_prob_wf)
            vaf1n=vaf1/sum(vaf1)

            gauss_absolute+=np.sum((vaf1-vaf2)**2)/nr_of_repeats
            ngauss_absolute+=np.sum((vaf1n-vaf2n)**2)/nr_of_repeats
            
            ngauss_binwise+=((vaf2n - vaf1n)**2)/nr_of_repeats
            gauss_binwise+=((vaf2 - vaf1)**2)/nr_of_repeats
            used_vafs.append(vaf1)

        ngauss_binwise_cum=np.cumsum(ngauss_binwise)
        gauss_binwise_cum=np.cumsum(gauss_binwise)

        ngauss_mean_error_variance=np.var(ngauss_binwise_cum)
        gauss_mean_error_variance=np.var(gauss_binwise_cum)

        upper_vaf = np.max(used_vafs, axis=0)
        lower_vaf = np.min(used_vafs, axis=0)
        mean_vaf= np.mean(used_vafs, axis=0)

        #already three new things to measure

        zeros_predicted=sum(mean_vaf==0)
        zeros_observed=sum(vaf2==0)
        lower_score=sum(vaf2-lower_vaf<0)
        upper_score=sum(upper_vaf-vaf2<0)
        total_score=upper_score+lower_score

        ngauss_mean_error_variance_wf.append(ngauss_mean_error_variance)
        gauss_mean_error_variance_wf.append(gauss_mean_error_variance)
        zeros_predicted_wf.append(zeros_predicted)
        zeros_observed_wf.append(zeros_observed)
        lower_score_wf.append(lower_score)
        upper_score_wf.append(upper_score)
        total_score_wf.append(total_score)

        gauss_absolute_wf.append(gauss_absolute)      
        ngauss_absolute_wf.append(ngauss_absolute)
        upper_vaf_bool=upper_vaf>=vaf2
        lower_vaf_bool=lower_vaf<=vaf2
        within_limits = [lower_vaf_bool[i] and upper_vaf_bool[i] for i in range(len(vaf2))]
        longest_false_estim_wf=longest_false_sequence(within_limits)
        longest_false_estimation_wf.append(longest_false_estim_wf )
        nonzero=longest_non_zero_underestimation(vaf2,lower_vaf)
        longest_nonzero_wf.append(nonzero)
        


        #################################################
        #Find peaks for WF
        vaf=numba_observed_vafs_alpha_vaf_array(pred_purity_wf,pred_Swf0,pred_C_wf,cov,binss,min_reads,xdata_wf,y_prob_wf)      

        # estimate the density function
        density = gaussian_kde(list(vaf), bw_method='scott')

        # evaluate the density in 1000 points (arbitrary)
        freq_range = np.linspace(min(vaf), max(vaf), 1000)
        dens_range = density(freq_range)

        # calculate the differences of adjacent points
        diff_y = np.diff(dens_range)

        # find the peaks
        peak_indices = np.where((diff_y[:-1] > 0) & (diff_y[1:] < 0))[0] + 1 
        freq_peak = freq_range[peak_indices]
        dens_peak = dens_range[peak_indices]

        # find the valleys
        valley_indices = np.where((diff_y[:-1] < 0) & (diff_y[1:] > 0))[0] + 1 
        valley_freq = freq_range[valley_indices]
        low_thresh = 0.01 # Threshold to remove peaks with low density
        # remove peaks with a very low density (noise): 1% of the highest peak
        freq_peak = freq_peak[dens_peak > low_thresh*max(dens_peak)]
        dens_peak = dens_peak[dens_peak > low_thresh*max(dens_peak)]
        wf_pc=len(freq_peak)
        wf_vaffp=freq_peak[0]

        ###########################################
    

        
        fig = plt.figure(figsize=(20, 8), dpi=120)
        pred_S=pred_Swf0
        pred_C=pred_C_wf
        pred_purity=pred_purity_wf

        model="WF"
        c='purple'
        ax2 = fig.add_subplot(1, 2,1)
        #ax2.bar(x_range_scatter, observed_hist/sum(observed_hist), alpha=0.4, label="observed", edgecolor='grey', width=0.01, color='blue')
        #ax2.bar(x_range_scatter, mean_vaf/sum(mean_vaf), alpha=0.4, label=model + "\nS=" + str(pred_S) + "\nC=" + str(pred_C) + "\npur=" + str(np.round(pred_purity, 2)), edgecolor='grey', width=0.01, color=c)
        ax2.hist(x=orignial_vaf, bins=100, range=(0, 1), density=True,alpha=0.4,edgecolor='grey',color='blue',label='observed')
        ax2.hist(x=vaf, bins=100, range=(0, 1), density=True,alpha=0.5,edgecolor='grey',color='purple', label=model + "\nS=" + str(pred_S) + "\nC=" + str(pred_C) + "\npur=" + str(np.round(pred_purity, 2)))
        ax2.plot(real_freq_range, real_dens_range, color='blue') # plot the probability density function
        ax2.scatter(real_freq_peak, real_dens_peak, color='blue') # plot the detected peaks
        ax2.plot(freq_range, dens_range, color='purple') # plot the probability density function
        ax2.scatter(freq_peak, dens_peak, color='purple')
        distance_of_first_peaks=wf_vaffp-real_freq_peak[0]
        
        ax2.set_title("gauss= " + str(np.round(gauss_absolute, 5)) + ",\n ngauss= " + str(np.round(ngauss_absolute, 5))+",\n nonzero= "+str(nonzero) +",\n fpd= "+str(np.round(distance_of_first_peaks,2)) )
        ax2.legend()

        gauss_absolute_wf_val=gauss_absolute
        ngauss_absolute_wf_val=ngauss_absolute
        nonzero_wf=nonzero
        distance_of_first_peaks_wf=distance_of_first_peaks


        ###########################################
        model="EXP"
        
        gauss_absolute=0.0
        ngauss_absolute=0.0

        ngauss_binwise=np.zeros(100)
        gauss_binwise=np.zeros(100)

        used_vafs=[]
        nr_of_repeats=10

        vaf2=observed_hist
        vaf2n=vaf2/sum(vaf2)
        for repeat in range(nr_of_repeats):    
            vaf1=simulated_vaf(pred_purity_exp,pred_Sexp0,pred_C_exp,cov,binss,min_reads,xdata_exp,y_prob_exp)
            vaf1n=vaf1/sum(vaf1)

            gauss_absolute+=np.sum((vaf1-vaf2)**2)/nr_of_repeats
            ngauss_absolute+=np.sum((vaf1n-vaf2n)**2)/nr_of_repeats
            ngauss_binwise+=((vaf2n - vaf1n)**2)/nr_of_repeats
            gauss_binwise+=((vaf2 - vaf1)**2)/nr_of_repeats

            used_vafs.append(vaf1)

        ngauss_binwise_cum=np.cumsum(ngauss_binwise)
        gauss_binwise_cum=np.cumsum(gauss_binwise)
        ngauss_mean_error_variance=np.var(ngauss_binwise_cum)
        gauss_mean_error_variance=np.var(gauss_binwise_cum)

        upper_vaf = np.max(used_vafs, axis=0)
        lower_vaf = np.min(used_vafs, axis=0)
        mean_vaf= np.mean(used_vafs, axis=0)

        #already three new things to measure

        zeros_predicted=sum(mean_vaf==0)
        zeros_observed=sum(vaf2==0)
        lower_score=sum(vaf2-lower_vaf<0)
        upper_score=sum(upper_vaf-vaf2<0)
        total_score=upper_score+lower_score
        ngauss_mean_error_variance_exp.append(ngauss_mean_error_variance)
        gauss_mean_error_variance_exp.append(gauss_mean_error_variance)
        zeros_predicted_exp.append(zeros_predicted)
        zeros_observed_exp.append(zeros_observed)
        lower_score_exp.append(lower_score)
        upper_score_exp.append(upper_score)
        total_score_exp.append(total_score)
        gauss_absolute_exp.append(gauss_absolute)
        ngauss_absolute_exp.append(ngauss_absolute)
        upper_vaf_bool=upper_vaf>=vaf2
        lower_vaf_bool=lower_vaf<=vaf2
        within_limits = [lower_vaf_bool[i] and upper_vaf_bool[i] for i in range(len(vaf2))]
        longest_false_estim_exp=longest_false_sequence(within_limits)
        longest_false_estimation_exp.append( longest_false_estim_exp)
        nonzero=longest_non_zero_underestimation(vaf2,lower_vaf)
        longest_nonzero_exp.append(nonzero)

    
         #################################################
        #Find peaks for EXP
        vaf=numba_observed_vafs_alpha_vaf_array(pred_purity_exp,pred_Sexp0,pred_C_exp,cov,binss,min_reads,xdata_exp,y_prob_exp)
        # estimate the density function
        density = gaussian_kde(list(vaf), bw_method='scott')
        # evaluate the density in 1000 points (arbitrary)
        freq_range = np.linspace(min(vaf), max(vaf), 1000)
        dens_range = density(freq_range)
        # calculate the differences of adjacent points
        diff_y = np.diff(dens_range)
        # find the peaks
        peak_indices = np.where((diff_y[:-1] > 0) & (diff_y[1:] < 0))[0] + 1 
        freq_peak = freq_range[peak_indices]
        dens_peak = dens_range[peak_indices]
        # find the valleys
        valley_indices = np.where((diff_y[:-1] < 0) & (diff_y[1:] > 0))[0] + 1 
        valley_freq = freq_range[valley_indices]
        low_thresh = 0.01 # Threshold to remove peaks with low density
        # remove peaks with a very low density (noise): 1% of the highest peak
        freq_peak = freq_peak[dens_peak > low_thresh*max(dens_peak)]
        dens_peak = dens_peak[dens_peak > low_thresh*max(dens_peak)]
        exp_pc=len(freq_peak)
        exp_vaffp=freq_peak[0]

        
        
        #######################################
        
        model="EXP"
        pred_S=pred_Sexp0
        pred_C=pred_C_exp
        pred_purity=pred_purity_exp
        c='yellow'
        ax3 = fig.add_subplot(1, 2,2)
        ax3.hist(x=orignial_vaf, bins=100, range=(0, 1), density=True,alpha=0.4,edgecolor='grey',color='blue',label='observed')
        ax3.hist(x=vaf, bins=100, range=(0, 1), density=True,alpha=0.5,edgecolor='grey',color='yellow', label=model + "\nS=" + str(pred_S) + "\nC=" + str(pred_C) + "\npur=" + str(np.round(pred_purity, 2)))
        #ax3.bar(x_range_scatter, observed_hist/sum(observed_hist), alpha=0.4, label="observed", edgecolor='grey', width=0.01, color='blue')
        #ax3.bar(x_range_scatter, mean_vaf/sum(mean_vaf), alpha=0.4, label=model + "\nS=" + str(pred_S) + "\nC=" + str(pred_C) + "\npur=" + str(np.round(pred_purity, 2)), edgecolor='grey', width=0.01, color=c)
        ax3.plot(real_freq_range, real_dens_range, color='blue') # plot the probability density function
        ax3.scatter(real_freq_peak, real_dens_peak, color='blue')# plot the detected peaks
        ax3.plot(freq_range, dens_range, color='black') # plot the probability density function
        ax3.scatter(freq_peak, dens_peak, color='black')
        distance_of_first_peaks=exp_vaffp-real_freq_peak[0]
        
        gauss_absolute_exp_val=gauss_absolute
        ngauss_absolute_exp_val=ngauss_absolute
        nonzero_exp=nonzero
        distance_of_first_peaks_exp=distance_of_first_peaks
        
        
        ax3.set_title("gauss= " + str(np.round(gauss_absolute, 5)) + ",\n ngauss= " + str(np.round(ngauss_absolute, 5))+",\n nonzero= "+str(nonzero) +",\n fpd= "+str(np.round(distance_of_first_peaks,2)) )
        ax3.legend()
        fig.suptitle(donor)
        plt.savefig(plots_folderpath+donor+'.png')
        #plt.show()

        ###########################################
        exp_vaf_first_peak.append(exp_vaffp)
        wf_vaf_first_peak.append(wf_vaffp)
        real_vaf_first_peak.append(real_vaffp)

        exp_peak_count.append(exp_pc)
        wf_peak_count.append(wf_pc)
        real_peak_count.append(real_pc)

        exp_peak_difference.append(distance_of_first_peaks_exp)
        wf_peak_difference.append(distance_of_first_peaks_wf)

            
        rdonors.append(donor)
        rcohorts.append(cohort)
        plt.close(fig)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        
data = {'donor':rdonors,
        'cohort':rcohorts,
        'ngauss_mean_error_variance_exp': ngauss_mean_error_variance_exp,
        'gauss_mean_error_variance_exp': gauss_mean_error_variance_exp,
        'zeros_predicted_exp': zeros_predicted_exp,
        'zeros_observed_exp': zeros_observed_exp,
        'lower_score_exp': lower_score_exp,
        'upper_score_exp': upper_score_exp,
        'total_score_exp': total_score_exp,
        'gauss_absolute_exp': gauss_absolute_exp,
        'ngauss_absolute_exp': ngauss_absolute_exp,
        'ngauss_mean_error_variance_wf': ngauss_mean_error_variance_wf,
        'gauss_mean_error_variance_wf': gauss_mean_error_variance_wf,
        'zeros_predicted_wf': zeros_predicted_wf,
        'zeros_observed_wf': zeros_observed_wf,
        'lower_score_wf': lower_score_wf,
        'upper_score_wf': upper_score_wf,
        'total_score_wf': total_score_wf,
        'gauss_absolute_wf': gauss_absolute_wf,
        'ngauss_absolute_wf': ngauss_absolute_wf,
        'longest_false_estimation_exp':longest_false_estimation_exp,
        'longest_nonzero_exp':longest_nonzero_exp,
        'longest_false_estimation_wf':longest_false_estimation_wf,
        'longest_nonzero_wf':longest_nonzero_wf, 
        'exp_vaf_first_peak':exp_vaf_first_peak,
        'wf_vaf_first_peak':wf_vaf_first_peak,
        'real_vaf_first_peak':real_vaf_first_peak,
        'exp_peak_count':exp_peak_count,
        'wf_peak_count':wf_peak_count,
        'real_peak_count':real_peak_count,
        'wf_peak_difference':wf_peak_difference,
        'exp_peak_difference':exp_peak_difference,}
df = pd.DataFrame(data)

df.to_csv(scored_results_folderpath)

all_df=df

all_df['normalized_total_score_exp']=all_df['total_score_exp']/(100-all_df['zeros_observed_exp'])
all_df['normalized_total_score_wf']=all_df['total_score_wf']/(100-all_df['zeros_observed_wf'])
all_df['ngauss_class']=(all_df['ngauss_absolute_wf']>all_df['ngauss_absolute_exp'])*1
all_df['gauss_class']=(all_df['gauss_absolute_wf']>all_df['gauss_absolute_exp'])*1


all_df['ngauss_absolute_wf_minus_exp']=all_df['ngauss_absolute_wf']-all_df['ngauss_absolute_exp']
all_df['gauss_absolute_wf_minus_exp']=all_df['gauss_absolute_wf']-all_df['gauss_absolute_exp']


bad_fits_1=all_df[(all_df.longest_nonzero_wf>10) & (all_df.longest_nonzero_exp>10)]

all_df=all_df[(all_df.longest_nonzero_wf<=10)  | (all_df.longest_nonzero_exp<=10)] 

bad_fits_2=all_df[(all_df.longest_false_estimation_wf>15)  & (all_df.exp_peak_difference>15)] 

all_df=all_df[(all_df.longest_false_estimation_wf<=15)  | (all_df.longest_false_estimation_exp<=15)]
used_df=all_df
wf_df = used_df[ (used_df['ngauss_absolute_wf_minus_exp'] < 0) & (used_df['gauss_absolute_wf_minus_exp'] < 0)]
exp_df = used_df[ (used_df['ngauss_absolute_wf_minus_exp'] > 0) & (used_df['gauss_absolute_wf_minus_exp'] > 0)]
wf_exp_donors = pd.concat([wf_df.donor, exp_df.donor]).unique()
neither_wf_nor_exp_df1 = used_df[~used_df.donor.isin(wf_exp_donors)]

exp_df_discarded=exp_df[exp_df['ngauss_absolute_wf_minus_exp']<=0.0005]
exp_df=exp_df[exp_df['ngauss_absolute_wf_minus_exp']>0.0005]
wf_df_discarded=wf_df[wf_df['ngauss_absolute_wf_minus_exp']>=-0.006]
wf_df=wf_df[wf_df['ngauss_absolute_wf_minus_exp']<-0.006]


bad_fits_1['selection_1_neutral_0']=1
bad_fits_2['selection_1_neutral_0']=1
exp_df_discarded['selection_1_neutral_0']=1
wf_df_discarded['selection_1_neutral_0']=1
exp_df['selection_1_neutral_0']=0
wf_df['selection_1_neutral_0']=0

bad_fits_1['neutral_wf']=0
bad_fits_2['neutral_wf']=0
exp_df_discarded['neutral_wf']=0
wf_df_discarded['neutral_wf']=0
exp_df['neutral_wf']=0
wf_df['neutral_wf']=1

bad_fits_1['neutral_exp']=0
bad_fits_2['neutral_exp']=0
exp_df_discarded['neutral_exp']=0
wf_df_discarded['neutral_exp']=0
exp_df['neutral_exp']=1
wf_df['neutral_exp']=0

bad_fits_1['classified_as']='bad_fits_1'
bad_fits_2['classified_as']='bad_fits_2'
exp_df_discarded['classified_as']='exp_df_discarded'
wf_df_discarded['classified_as']='wf_df_discarded'
exp_df['classified_as']='exp_df'
wf_df['classified_as']='wf_df'


#and add someting for wf_peak_difference and exp_peak_difference if it is needed.
#bad_fits_3=all_df[(all_df.wf_peak_difference>x)  & (all_df.exp_peak_difference>x)]
# maybe first need to adjust the minimum peak height that it detects


classified_df=pd.concat([bad_fits_1,bad_fits_2,exp_df_discarded,wf_df_discarded,exp_df,wf_df])
classified_df.to_csv(scored_results_folderpath)

