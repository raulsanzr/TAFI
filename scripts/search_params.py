import numpy as np
from scipy.stats import wasserstein_distance
from numba import njit
from simulations import *

@njit
def new_params(pur, S, C, proposal_sd):
    '''
    Proposes new candidate parameters by sampling from a symmetric Gaussian distribution centered on the current values.
    '''
    new_pur=min(1.0, max(0.0, pur + np.random.normal()  * 0.1))
    new_S=int(max(0, S + np.random.normal() * S*0.5*proposal_sd))
    new_C=int(max(0, C + np.random.normal() * C*0.5*proposal_sd))
    return new_pur, new_S, new_C

def dist(real_vaf, nr_of_repeats, purity, S, C, cov,min_reads, xdata, y_prob):
    '''
    Calculates the wasserstein distance between the observed and simulated VAFs.
    '''
    r_nd5=np.zeros(nr_of_repeats)
    for repeat in range(nr_of_repeats):
        vaf1=sim_vafs(purity,S,C,cov,min_reads,xdata,y_prob)
        if len(vaf1) > 10: # if there are at leat 10 mutations simulated
            r_nd5[repeat]=wasserstein_distance(vaf1, real_vaf)
        else:
            r_nd5[repeat]=1
        return np.mean(r_nd5)

def s_adjustment(first_bins, observed_vaf, cov, min_reads, xdata, y_prob): # TODO: I changed observed_vaf, now is a vaf
    '''
    Estimates a suitable range for the parameter S by comparing observed and simulated data
    '''
    observed_hist, _=np.histogram(observed_vaf, bins=100, range=(0, 1), density=True)
    observed_max_tmb=np.max(observed_hist[:first_bins])
    purity=1.0
    C=0
    S_range=np.array([10**2, 10**3, 10**4, 10**5, 10**6, 10**7])
    
    measured_tmb=np.zeros(len(S_range))
    max_tmb=np.zeros(len(S_range))
    
    for i, S in enumerate(S_range):
        sim_hist=sim_vafs(purity, S, C, cov, min_reads, xdata, y_prob)
        measured_tmb[i]=np.sum(sim_hist[:first_bins])
        max_tmb[i]=np.max(sim_hist[:first_bins])

    ratio=observed_max_tmb / max_tmb
    S_estimate=S_range[(ratio > 0.05) & (ratio < 50)]

    return S_estimate

def run_fit(test_model, cov, min_reads, max_steps, observed_vaf, collected_data_size, xdata, y_prob, pred_purity, pred_C):
    '''
    
    '''
    first_bins=25
    proposal_sd=1
    nr_of_repeats=5

    # Initialize the parameters. collected_data (100) sets of parameters that are adjusted max_steps (100) times
    S_estim=s_adjustment(first_bins,observed_vaf,cov,min_reads,xdata,y_prob) # TODO: check this
    current_S=int(10**np.mean(np.log10(S_estim)))
    S_init=(np.ones(collected_data_size)*current_S).astype(int)
    scores_init=np.ones(collected_data_size)
 
    if test_model=='WF':
        C_min, C_max=100, 100000
        C_init=np.exp(np.random.uniform(low=np.log(C_min), high=np.log(C_max), size=collected_data_size)).astype(int)
        pur_init=np.random.rand(collected_data_size)
    elif test_model=='EXP': # C and purity are informed (re-used) from the WF model (should be the same)
        C_init=pred_C
        pur_init=pred_purity

    for i in range(0,max_steps):
        for counter,current_purity,current_S,current_C in zip(range(0,collected_data_size),pur_init,S_init,C_init):
            current_dist=scores_init[counter]
            new_purity,new_S,new_C=new_params(current_purity,current_S, current_C, proposal_sd)
            proposed_dist=dist(observed_vaf,nr_of_repeats,new_purity,new_S,new_C,cov,min_reads,xdata,y_prob)
            if proposed_dist < current_dist:
                pur_init[counter]=new_purity
                S_init[counter]=new_S
                C_init[counter]=new_C
                scores_init[counter]=proposed_dist
    return (pur_init, S_init, C_init, scores_init)