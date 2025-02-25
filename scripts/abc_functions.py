import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import sys
import scipy.stats as st
from scipy.stats import wasserstein_distance
from numba import njit
from numba import jit
import random

@njit
def simulate_clonal(pur, C, cov, min_reads):
    '''
    Simulates the number of clonal alternative reads and coverage.
    '''
    coverage_dist_emp_C = np.random.choice(cov, C, replace=True)
    clonal_alt_reads = np.empty(C, dtype=np.int64)
    for i in range(C):
        clonal_alt_reads[i] = np.random.binomial(coverage_dist_emp_C[i], pur*1/2.0)
    return coverage_dist_emp_C[clonal_alt_reads>min_reads], clonal_alt_reads[clonal_alt_reads>min_reads] #remove condition > min_reads if you want to change the filter

@njit
def random_choice_with_probabilities(xdata, p, size):
    cum_prob = np.cumsum(p)
    samples = np.empty(size, dtype=xdata.dtype)
    for i in range(size):
        r = np.random.rand()
        idx = np.searchsorted(cum_prob, r)
        samples[i] = xdata[idx]
    return samples

@njit
def generate_subclonal_alt_reads(coverage_dist_emp_S, pur, subclonal_frequencies, S):
    subclonal_alt_reads = np.empty(S, dtype=np.int64)
    for i in range(S):
        p = pur * 0.5 * subclonal_frequencies[i]
        subclonal_alt_reads[i] = np.random.binomial(coverage_dist_emp_S[i], p)
    return subclonal_alt_reads

@njit
def simulate_subclonal(pur, S, cov, min_reads, xdata, y_prob):
    '''
    Simulates the number of subclonal alternative reads and coverage.
    '''
    coverage_dist_emp_S = np.random.choice(cov, S, replace = True)
    subclonal_frequencies =  random_choice_with_probabilities(xdata, y_prob, S)
    subclonal_alt_reads = generate_subclonal_alt_reads(coverage_dist_emp_S, pur, subclonal_frequencies, S)
    return coverage_dist_emp_S[subclonal_alt_reads>=min_reads], subclonal_alt_reads[subclonal_alt_reads>=min_reads]

@njit
def simulate_vafs(pur,S,C,cov,min_reads,xdata,y_prob):
    '''
    Builds the VAF histogram for the simulated clonal and subclonal mutations.
    '''
    cov_C, alt_C = simulate_clonal(pur, C, cov, min_reads)
    cov_S, alt_S = simulate_subclonal(pur, S, cov, min_reads, xdata, y_prob)
    vaf = np.empty(len(cov_C) + len(cov_S)) 
    for i in range(len(cov_C)):
        vaf[i] = alt_C[i] / cov_C[i]
    for i in range(len(cov_C), len(cov_C) + len(cov_S)):
        vaf[i] = alt_S[i - len(cov_C)] / cov_S[i - len(cov_C)] 
    return vaf

@njit
def propose_new_parameters(current_purity,current_S, current_C, proposal_sd):
    '''
    Proposes the new parameters for the next iteration.
    '''
    new_purity = min(1.0, max(0.0, current_purity + np.random.normal()  * 0.1))
    new_S = int(max(0, current_S + np.random.normal() * current_S*0.5*proposal_sd))
    new_C = int(max(0, current_C + np.random.normal() * current_C*0.5*proposal_sd))

    return new_purity, new_S, new_C

def wasserstein(vaf1, vaf2):
    '''
    Calculates the Wasserstein distance from the variant allele frequencies
    '''
    return wasserstein_distance(vaf1, vaf2)

def f_alpha(x,a, alpha):
    return a*(1/(x**alpha))

@njit
def distance(vaf2,nr_of_repeats,purity,S,C,cov,min_reads,xdata,y_prob):
    '''
    Calculates the distance (averaged across nr_of_repeats times) between the observed and simulated VAFs.
    '''
    r_nd5 = np.zeros(nr_of_repeats)
    for repeat in range(nr_of_repeats):
        vaf1 = simulate_vafs(purity,S,C,cov,min_reads,xdata,y_prob)
        r_nd5[repeat] = wasserstein(vaf1, vaf2)
    return np.mean(r_nd5)

@njit
def s_adjustment(first_bins, observed_vaf, cov, min_reads, xdata, y_prob): # TODO: I changed observed_vaf, now is a vaf
    '''
    Estimates a suitable range for the parameter S by comparing observed and simulated data
    '''
    observed_max_tmb = np.max(observed_vaf[:first_bins]) #
    purity = 1.0
    C = 0
    S_range = np.array([10**2, 10**3, 10**4, 10**5, 10**6, 10**7])
    
    measured_tmb = np.zeros(len(S_range))
    max_tmb = np.zeros(len(S_range))
    
    for i, S in enumerate(S_range):
        sim_hist = simulate_vafs(purity, S, C, cov, min_reads, xdata, y_prob)
        measured_tmb[i] = np.sum(sim_hist[:first_bins])
        max_tmb[i] = np.max(sim_hist[:first_bins])

    ratio = observed_max_tmb / max_tmb
    S_estimate = S_range[(ratio > 0.05) & (ratio < 50)]

    return S_estimate

def run_fit(test_model,cov,min_reads,max_steps,observed_vaf,collected_data_size,xdata,y_prob,pred_purity=None,pred_C=None):
    '''
    Runs the fitting process for the Wright-Fisher and Exponential models.
    '''
    first_bins=25 

    S_estim=s_adjustment(first_bins,observed_vaf,cov,min_reads,xdata,y_prob) # TODO: check this
    current_S=int(10**np.mean(np.log10(S_estim)))
    S_init=(np.ones(collected_data_size)*current_S).astype(int)
    
    if test_model=='WF':
        C_min = 100
        C_max = 100000
        C_init = np.exp(np.random.uniform(low=np.log(C_min), high=np.log(C_max), size=collected_data_size)).astype(int)
        pur_init= np.random.rand(collected_data_size)
    elif test_model=='EXP': # C and purity are informed from WF model
        C_options=[pred_C]
        C_init = np.random.choice(C_options, size=collected_data_size)
        p_options=[pred_purity]
        pur_init= np.random.choice(p_options, size=collected_data_size)

    proposal_sd=1
    scores_init=np.ones(collected_data_size)
    for i in range(0,max_steps):
        for counter,current_purity,current_S,current_C in zip(range(0,collected_data_size),pur_init,S_init,C_init):

            proposal_sd=1
            nr_of_repeats=1 #try with just one repeat to speedup
            current_dist=distance(observed_vaf,nr_of_repeats,current_purity,current_S,current_C,cov,min_reads,xdata,y_prob)
            
            scores_init[counter]=current_dist

            (new_purity,new_S,new_C)=propose_new_parameters(current_purity,current_S, current_C, proposal_sd)
            proposed_dist=distance(observed_vaf,nr_of_repeats,new_purity,new_S,new_C,cov,min_reads,xdata,y_prob)

            if proposed_dist < current_dist:
                pur_init[counter]=new_purity
                S_init[counter]=new_S
                C_init[counter]=new_C
                scores_init[counter]=proposed_dist

        prob=1./scores_init/sum(1./scores_init)
        indices = np.arange(len(scores_init))
        sampled_indices = np.random.choice(indices, size=collected_data_size, p=prob)
        pur_init,S_init,C_init,scores_init=pur_init[sampled_indices],S_init[sampled_indices],C_init[sampled_indices],scores_init[sampled_indices]

    return (pur_init,S_init,C_init,scores_init)