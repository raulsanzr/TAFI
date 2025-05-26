import numpy as np
from scipy.stats import wasserstein_distance
from numba import njit

def f_alpha(x, alpha):
    '''
    Computes the frequency of observing a mutation at a given coverage.
    '''
    return 1/(x**alpha)

@njit
def clonal_mutations(pur, C, cov, min_reads, ploidy=2):
    '''
    Simulates the sampling of clonal mutations.
    '''
    coverage_dist_emp_C = np.random.choice(cov, C, replace=True)
    clonal_alt_reads = np.empty(C, dtype=np.int64)
    for i in range(C):
        clonal_alt_reads[i] = np.random.binomial(coverage_dist_emp_C[i], pur/ploidy)
    return coverage_dist_emp_C[clonal_alt_reads>min_reads], clonal_alt_reads[clonal_alt_reads>min_reads]

@njit
def random_choice(xdata, p, size):
    '''
    Numba version of np.random.choice.
    '''
    cum_prob = np.cumsum(p)
    samples = np.empty(size, dtype=xdata.dtype)
    for i in range(size):
        r = np.random.rand()
        idx = np.searchsorted(cum_prob, r)
        samples[i] = xdata[idx]
    return samples

@njit
def random_binomial(coverage_dist_emp_S, pur, subclonal_frequencies, S):
    '''
    Numba version of np.random.binomial.
    '''
    subclonal_alt_reads = np.empty(S, dtype=np.int64)
    for i in range(S):
        p = pur*0.5*subclonal_frequencies[i]
        subclonal_alt_reads[i] = np.random.binomial(coverage_dist_emp_S[i], p)
    return subclonal_alt_reads

@njit
def subclonal_mutations(pur, S, cov, min_reads, xdata, y_prob):
    '''
    Simulates the sampling of subclonal mutations.
    '''
    coverage_dist_emp_S = np.random.choice(cov, S, replace=True)
    subclonal_frequencies = random_choice(xdata, y_prob, S)
    subclonal_alt_reads = random_binomial(coverage_dist_emp_S, pur, subclonal_frequencies, S)
    return coverage_dist_emp_S[subclonal_alt_reads>=min_reads], subclonal_alt_reads[subclonal_alt_reads>=min_reads]

@njit
def simulate_vaf(pur, S, C, cov, min_reads, xdata, y_prob):
    '''
    Builds the VAF histogram for the simulated clonal and subclonal mutations.
    '''
    cov_C, alt_C = clonal_mutations(pur, C, cov, min_reads)
    cov_S, alt_S = subclonal_mutations(pur, S, cov, min_reads, xdata, y_prob)
    vaf = np.empty(len(cov_C)+len(cov_S)) 
    for i in range(len(cov_C)):
        vaf[i] = alt_C[i]/cov_C[i]
    for i in range(len(cov_C), len(cov_C)+len(cov_S)):
        vaf[i] = alt_S[i -len(cov_C)]/cov_S[i-len(cov_C)] 
    return vaf

def distance(observed_vaf, repeats, purity, S, C, cov, min_reads, xdata, y_prob, lam):
    '''
    Calculates the wasserstein distance between the observed and simulated VAFs.
    '''
    scores = np.zeros(repeats)
    for repeat in range(repeats):
        simulated_vaf = simulate_vaf(purity, S, C, cov, min_reads, xdata, y_prob) # Simulate the vafs for the given parameters
        observed_n, simulated_n = len(observed_vaf), len(simulated_vaf) # Get the number of mutations in the observed and simulated vafs
        max_w = np.mean(np.maximum(observed_vaf, 1-observed_vaf)) # Maximum possible wasserstein distance given the observed vaf
        try:
            w_dist = wasserstein_distance(simulated_vaf, observed_vaf)
            w_dist_norm = w_dist/max_w # Normalized wasserstein distance
            n_diff_norm = min(abs(simulated_n-observed_n)/observed_n, 1) # Normalized count difference (or 1 if the difference is too large)
            scores[repeat] = w_dist_norm+lam*n_diff_norm
        except:
            return np.inf # If the wasserstein distance cannot be computed, return infinity
    return np.mean(scores)

@njit
def new_params(pur, S, C, proposal_sd=1):
    '''
    Proposes new candidate parameters.
    '''
    new_pur = min(1.0, max(0.0, pur+np.random.normal()*0.1))
    new_S = int(max(0, S+np.random.normal()*S*0.5*proposal_sd))
    new_C = int(max(0, C+np.random.normal()*C*0.5*proposal_sd))
    return new_pur, new_S, new_C

def mcmc(model, cov, min_reads, iter, observed_vaf, starting_points, xdata, y_prob, pur_informed, C_informed, lam=0.2, repeats=5, S_min=10e2, S_max=10e5, C_min=10e2, C_max=10e5):
    '''
    Performs the MCMC fitting process to find the set of parameters that best fit the observed variant allele frequencies.
    '''
    # Initialize the parameters for starting_points (default = 100) dimensions
    S_init = np.exp(np.random.uniform(low=np.log(S_min), high=np.log(S_max), size=starting_points)).astype(int) # S is sampled from a log-uniform distribution

    if model == 'WF':
        C_init = np.exp(np.random.uniform(low=np.log(C_min), high=np.log(C_max), size=starting_points)).astype(int) # C is sampled from a log-uniform distribution
        pur_init = np.random.rand(starting_points) # Purity is sampled from a uniform distribution
    elif model == 'EXP': # C and purity are informed (re-used) from the WF model previously fitted
        C_init = C_informed
        pur_init = pur_informed
    scores_init = np.inf*np.ones(starting_points) # Initialize the scores

    for i in range(0, iter):
        for point, current_pur, current_S, current_C in zip(range(0, starting_points), pur_init, S_init, C_init):
            current_score = scores_init[point]
            new_pur, new_S, new_C = new_params(current_pur, current_S, current_C)
            new_score = distance(observed_vaf, repeats, new_pur, new_S, new_C, cov, min_reads, xdata, y_prob, lam)
            if new_score < current_score:
                pur_init[point], S_init[point], C_init[point], scores_init[point] = new_pur, new_S, new_C, new_score
        
    return (pur_init, S_init, C_init, scores_init)