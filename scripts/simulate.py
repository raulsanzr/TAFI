import numpy as np
from numba import njit

@njit
def sim_clonal(pur, C, cov, min_reads):
    '''
    Simulates the number of clonal alternative reads and coverage.
    '''
    coverage_dist_emp_C=np.random.choice(cov, C, replace=True)
    clonal_alt_reads=np.empty(C, dtype=np.int64)
    for i in range(C):
        clonal_alt_reads[i]=np.random.binomial(coverage_dist_emp_C[i], pur*1/2.0)
    return coverage_dist_emp_C[clonal_alt_reads>min_reads], clonal_alt_reads[clonal_alt_reads>min_reads]

@njit
def random_choice_with_probabilities(xdata, p, size):
    cum_prob=np.cumsum(p)
    samples=np.empty(size, dtype=xdata.dtype)
    for i in range(size):
        r=np.random.rand()
        idx=np.searchsorted(cum_prob, r)
        samples[i]=xdata[idx]
    return samples

@njit
def generate_subclonal_alt_reads(coverage_dist_emp_S, pur, subclonal_frequencies, S):
    subclonal_alt_reads=np.empty(S, dtype=np.int64)
    for i in range(S):
        p=pur * 0.5 * subclonal_frequencies[i]
        subclonal_alt_reads[i]=np.random.binomial(coverage_dist_emp_S[i], p)
    return subclonal_alt_reads

@njit
def sim_subclonal(pur, S, cov, min_reads, xdata, y_prob):
    '''
    Simulates the number of subclonal alternative reads and coverage.
    '''
    coverage_dist_emp_S=np.random.choice(cov, S, replace=True)
    subclonal_frequencies= random_choice_with_probabilities(xdata, y_prob, S)
    subclonal_alt_reads=generate_subclonal_alt_reads(coverage_dist_emp_S, pur, subclonal_frequencies, S)
    return coverage_dist_emp_S[subclonal_alt_reads>=min_reads], subclonal_alt_reads[subclonal_alt_reads>=min_reads]

@njit
def sim_vafs(pur, S, C, cov,min_reads, xdata, y_prob):
    '''
    Builds the VAF histogram for the simulated clonal and subclonal mutations.
    '''
    cov_C, alt_C=sim_clonal(pur, C, cov, min_reads)
    cov_S, alt_S=sim_subclonal(pur, S, cov, min_reads, xdata, y_prob)
    vaf=np.empty(len(cov_C) + len(cov_S)) 
    for i in range(len(cov_C)):
        vaf[i]=alt_C[i] / cov_C[i]
    for i in range(len(cov_C), len(cov_C) + len(cov_S)):
        vaf[i]=alt_S[i - len(cov_C)] / cov_S[i - len(cov_C)] 
    return vaf

def f_alpha(x, a, alpha):
    return a*(1/(x**alpha))
