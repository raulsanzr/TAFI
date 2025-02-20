import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import sys
import scipy.stats as st
from scipy.stats import multivariate_normal, wasserstein_distance
from scipy.special import rel_entr
from scipy.spatial.distance import cosine
from numba import njit
from numba import jit
import random

@njit
def clonal(pur, C, cov, min_reads):
    '''
    Simulates the number of clonal alternative reads and coverage.
    '''
    coverage_dist_emp_C = np.random.choice(cov, C, replace=True)
    clonal_alt_reads = np.empty(C, dtype=np.int64)
    for i in range(C):
        clonal_alt_reads[i] = np.random.binomial(coverage_dist_emp_C[i], pur*1/2.0)
    return coverage_dist_emp_C[clonal_alt_reads>min_reads], clonal_alt_reads[clonal_alt_reads>min_reads] # NICO: remove condition > min_reads if you want to change the filter

@njit
def random_choice_with_probabilities(xdata, p, size):
    '''
    Random sampling from xdata based on given probabilities p.
    '''
    cum_prob = np.cumsum(p)
    samples = np.empty(size, dtype=xdata.dtype)
    for i in range(size):
        r = np.random.rand()
        idx = np.searchsorted(cum_prob, r)
        samples[i] = xdata[idx]
    return samples

@njit
def subclonal(pur, S, cov, min_reads, xdata, y_prob):
    '''
    Simulates the number of subclonal alternative reads and coverage.
    '''
    coverage_dist_emp_S = np.random.choice(cov, S, replace = True)
    subclonal_frequencies =  random_choice_with_probabilities(xdata, y_prob, S)
    subclonal_alt_reads = np.empty(S, dtype=np.int64)
    for i in range(S):
        p = pur * 0.5 * subclonal_frequencies[i]
        subclonal_alt_reads[i] = np.random.binomial(coverage_dist_emp_S[i], p)
    return coverage_dist_emp_S[subclonal_alt_reads>=min_reads], subclonal_alt_reads[subclonal_alt_reads>=min_reads]

@njit
def simulated_vaf(pur,S,C,cov,binss,min_reads,xdata,y_prob):
    '''
    Builds the histogram for the simulated variant allele frequencies.
    '''
    cov_C, alt_C = clonal(pur, C, cov, min_reads)
    cov_S, alt_S = subclonal(pur, S, cov, min_reads, xdata, y_prob)
    vaf = np.empty(len(cov_C) + len(cov_S)) 
    for i in range(len(cov_C)):
        vaf[i] = alt_C[i] / cov_C[i]
    for i in range(len(cov_C), len(cov_C) + len(cov_S)):
        vaf[i] = alt_S[i - len(cov_C)] / cov_S[i - len(cov_C)] 
    return np.histogram(vaf, bins=binss, range=(0, 1))[0]

@njit
def new_parameters(current_purity,current_S, current_C, proposal_sd):
    '''
    Defines the new parameters for the next iteration.
    '''
    new_purity = min(1.0, max(0.0, current_purity + np.random.normal()  * 0.1))
    new_S = int(max(0, current_S + np.random.normal() * current_S*0.5*proposal_sd))
    new_C = int(max(0, current_C + np.random.normal() * current_C*0.5*proposal_sd))
    return new_purity, new_S, new_C

# ---- DISTANCES --- #

# Sum of squares
@njit
def ss_dist(vaf1,vaf2): 
    return abs(np.sum((vaf1-vaf2)**2)) 

# Chi-Square
@njit
def chi_dist(vaf1, vaf2):
    return 0.5 * np.sum((vaf1 - vaf2) ** 2 / (vaf1 + vaf2 + 1e-10))

# Kolmogorov-Smirnov
@njit
def ks_dist(vaf1, vaf2): 
    return np.max(np.abs(np.cumsum(vaf1) - np.cumsum(vaf2)))

# Wasserstein: Measures the effort to transform one histogram into another.
@njit
def wasserstein_dist(vaf1, vaf2):
    return wasserstein_distance(range(len(vaf1)), range(len(vaf2)), vaf1, vaf2)

# Kullback-Leibler: Measures how much information is lost when using one distribution to approximate another.
@njit
def kl_dist(vaf1, vaf2):
    return np.sum(rel_entr(vaf1 + 1e-10, vaf2 + 1e-10))  # Avoid log(0) errors

# Bhattacharyya: Measures the overlap between two probability distributions.
@njit
def bhattacharyya_dist(vaf1, vaf2):
    return -np.log(np.sum(np.sqrt(vaf1 * vaf2)))

# Cosine Similarity: Measures the cosine of the angle between two histogram vectors.
@njit
def cosine_distance(vaf1, vaf2):
    return cosine(vaf1, vaf2)

#  ----------------- # 

@njit
def compute_distance(vaf2,nr_of_repeats,purity,S,C,cov,binss,min_reads,xdata,y_prob):
    '''
    Computes the distance between the observed and simulated VAFs. The distance is computed nr_of_repeats times and the average is returned.
    '''
    r_nd5 = np.zeros(nr_of_repeats)
    for repeat in range(nr_of_repeats):
        vaf1 = simulated_vaf(purity,S,C,cov,binss,min_reads,xdata,y_prob)
        r_nd5[repeat] = ss_dist(vaf1, vaf2)
    return np.mean(r_nd5)

@njit
def adjust_S(model,first_bins,observed_hist,cov,binss,min_reads,xdata,y_prob): 
    '''
    Adjusts the initial S depending on the model and the observed data.
    '''
    observed_max_tmb=np.max(observed_hist[:first_bins])
    purity=1.0
    C=0
    i=0
    if model=="WF":
        S_range=np.array([10**2,10**3,10**4,10**5,10**6,10**7])
        measured_tmb=np.zeros(len(S_range))
        max_tmb=np.zeros(len(S_range))
        for S in S_range:
            sim_hist= simulated_vaf(purity,S,C,cov,binss,min_reads,xdata,y_prob)
            measured_tmb[i]=np.sum(sim_hist[:first_bins])
            max_tmb[i]=np.max(sim_hist[:first_bins])
            i+=1
        ratio = observed_max_tmb / max_tmb
        S_estimate=S_range[(ratio > 0.05) & (ratio < 50)]
    elif model=="EXP":
        S_range=np.array([10**2,10**3,10**4,10**5,10**6,10**7])
        measured_tmb=np.zeros(len(S_range))
        max_tmb=np.zeros(len(S_range))
        for S in S_range:
            sim_hist= simulated_vaf(purity,S,C,cov,binss,min_reads,xdata,y_prob)
            measured_tmb[i]=np.sum(sim_hist[:first_bins])
            max_tmb[i]=np.max(sim_hist[:first_bins])
            i+=1
        ratio = observed_max_tmb / max_tmb
        S_estimate=S_range[(ratio > 0.05) & (ratio < 50)]
    return S_estimate

def f_alpha(x,a, alpha):
    return a*(1/(x**alpha))

def run_abc(test_model,cov,min_reads,observed_hist,collected_data_size,xdata,y_prob):
    binss = np.linspace(0, 1, 101)
    
    pur_init= np.random.rand(collected_data_size)
    first_bins=25 # NICO: before 15
    S_estim=adjust_S(test_model,first_bins,observed_hist,cov,binss,min_reads,xdata,y_prob)
    current_S=int(10**np.mean(np.log10(S_estim)))
    S_init=(np.ones(collected_data_size)*current_S).astype(int)
    C_min, C_max = 100, 100000                                                                                      # NICO: C_options=[10,100,1000,10000,100000]
    C_init = np.exp(np.random.uniform(low=np.log(C_min), high=np.log(C_max), size=collected_data_size)).astype(int) # NICO: C_init=np.random.choice(C_options, size=collected_data_size)

    proposal_sd=1
    nr_of_repeats=1 # NICO: try with just one repeat to speedup
    scores_init=np.ones(collected_data_size)

    for counter,current_purity,current_S,current_C in zip(range(0,collected_data_size),pur_init,S_init,C_init):
        current_dist=compute_distance(observed_hist,nr_of_repeats,current_purity,current_S,current_C,cov,binss,min_reads,xdata,y_prob)
        scores_init[counter]=current_dist
        new_purity,new_S,new_C=new_parameters(current_purity,current_S, current_C, proposal_sd)        
        proposed_dist=compute_distance(observed_hist,nr_of_repeats,new_purity,new_S,new_C,cov,binss,min_reads,xdata,y_prob)
        if proposed_dist < current_dist:
            pur_init[counter], S_init[counter], C_init[counter], scores_init[counter] = new_purity, new_S, new_C, proposed_dist

        prob=1./scores_init/sum(1./scores_init)
        indices = np.arange(len(scores_init))
        sampled_indices = np.random.choice(indices, size=collected_data_size, p=prob)
        sampled_values = scores_init[sampled_indices]
        pur_init,S_init,C_init,scores_init=pur_init[sampled_indices],S_init[sampled_indices],C_init[sampled_indices],scores_init[sampled_indices]

    return (pur_init,S_init,C_init,scores_init)

def run_fit_smc_with_WF_initial(test_model,cov,min_reads,observed_hist,collected_data_size,xdata,y_prob,pred_purity,pred_C):
    binss = np.linspace(0, 1, 101)
    first_bins=25 # NICO: before 15

    S_estim=adjust_S(test_model,first_bins,observed_hist,cov,binss,min_reads,xdata,y_prob)
    current_S=int(10**np.mean(np.log10(S_estim)))
    S_init=(np.ones(collected_data_size)*current_S).astype(int)
    
    C_options=[pred_C]
    C_init = np.random.choice(C_options, size=collected_data_size)

    p_options=[pred_purity]
    pur_init= np.random.choice(p_options, size=collected_data_size)

    # NICO: C_init = np.exp(np.random.uniform(low=np.log(C_min), high=np.log(C_max), size=collected_data_size)).astype(int)

    proposal_sd=1
    nr_of_repeats=1 # NICO: try with just one repeat to speedup
    scores_init=np.ones(collected_data_size)
    for counter,current_purity,current_S,current_C in zip(range(0,collected_data_size),pur_init,S_init,C_init):
        current_dist=compute_distance(observed_hist,nr_of_repeats,current_purity,current_S,current_C,cov,binss,min_reads,xdata,y_prob)
        
        scores_init[counter]=current_dist

        (new_purity,new_S,new_C)=new_parameters(current_purity,current_S, current_C, proposal_sd)
        proposed_dist=compute_distance(observed_hist,nr_of_repeats,new_purity,new_S,new_C,cov,binss,min_reads,xdata,y_prob)

        if proposed_dist < current_dist:
            pur_init[counter], S_init[counter], C_init[counter], scores_init[counter] = new_purity, new_S, new_C, proposed_dist

        prob=1./scores_init/sum(1./scores_init)
        indices = np.arange(len(scores_init))
        sampled_indices = np.random.choice(indices, size=collected_data_size, p=prob)
        sampled_values = scores_init[sampled_indices]
        pur_init,S_init,C_init,scores_init=pur_init[sampled_indices],S_init[sampled_indices],C_init[sampled_indices],scores_init[sampled_indices]

    return (pur_init,S_init,C_init,scores_init)