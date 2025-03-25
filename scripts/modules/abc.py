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

def distance(observed_vaf, repeats, purity, S, C, cov, min_reads, xdata, y_prob, lam=1):
    '''
    Calculates the wasserstein distance between the observed and simulated VAFs.
    '''
    scores = np.zeros(repeats)
    for repeat in range(repeats):
        simulated_vaf = simulate_vaf(purity, S, C, cov, min_reads, xdata, y_prob) # Simulate the vafs for the given parameters
        observed_n, simulated_n = len(observed_vaf), len(simulated_vaf) # Get the number of mutations in the observed and simulated vafs
        if simulated_n > 10: # If there are at leat 10 mutations simulated
            w_dist = wasserstein_distance(simulated_vaf, observed_vaf)
            n_diff_norm = abs(simulated_n-observed_n)/observed_n # Normalized count difference
            # scores[repeat] = w_dist # option 0
            scores[repeat] = w_dist+lam*n_diff_norm # option 1
        else:
            return np.inf
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

# def initial_s(first_bins, observed_vaf, cov, min_reads, xdata, y_prob): # TODO: I changed observed_vaf, now is a vaf
#     '''
#     Estimates a suitable range for the parameter S by comparing observed and simulated data.
#     '''
#     observed_hist, _ = np.histogram(observed_vaf, bins=100, range=(0, 1), density=True)
#     observed_max_tmb = np.max(observed_hist[:first_bins])
#     purity = 1.0
#     C = 0
#     S_range = np.array([10**2, 10**3, 10**4, 10**5, 10**6, 10**7])
#     measured_tmb = np.zeros(len(S_range))
#     max_tmb = np.zeros(len(S_range))
#     for i, S in enumerate(S_range):
#         sim_vaf = sim_vafs(purity, S, C, cov, min_reads, xdata, y_prob)
#         sim_hist, _ = np.histogram(sim_vaf, bins=100, range=(0, 1), density=True)
#         measured_tmb[i] = np.sum(sim_hist[:first_bins])
#         max_tmb[i] = np.max(sim_hist[:first_bins])
#     ratio = observed_max_tmb/max_tmb
#     S_estimate = S_range[(ratio > 0.05) & (ratio < 50)]
#     return S_estimate

def mcmc(model, cov, min_reads, iter, observed_vaf, starting_points, xdata, y_prob, pur_informed, C_informed, repeats=1, S_min=10e2, S_max=10e5, C_min=10e2, C_max=10e5):
    '''
    Performs the MCMC fitting process to find the set of parameters that best fit the observed variant allele frequencies.
    '''
    # Initialize the parameters for starting_points (default = 100) dimensions
    S_init = np.exp(np.random.uniform(low=np.log(S_min), high=np.log(S_max), size=starting_points)).astype(int) # S is sampled from a log-uniform distribution

    # S_estim = initial_s(first_bins, observed_vaf, cov, min_reads, xdata, y_prob)
    # current_S = 0 if len(S_estim) == 0 else int(10 ** np.mean(np.log10(S_estim)))
    # S_init = (np.ones(starting_points)*current_S).astype(int)

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
            new_score = distance(observed_vaf, repeats, new_pur, new_S, new_C, cov, min_reads, xdata, y_prob)
            if new_score < current_score:
                pur_init[point], S_init[point], C_init[point], scores_init[point] = new_pur, new_S, new_C, new_score
        
    return (pur_init, S_init, C_init, scores_init)