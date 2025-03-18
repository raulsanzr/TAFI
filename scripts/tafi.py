import os
import sys
import pandas as pd
import numpy as np
from modules.abc import *

# Search parameters
iter = 100 # Number of iterations
starting_points = 100 # Number of starting points

# Create results directory if it doesn't exist
results_dir = '../results/individual/'
os.makedirs(results_dir, exist_ok=True)

# Read the data
input_file = sys.argv[1]
donor_id = input_file.split('/')[-1].split('.')[0] # Extract donor ID from the file path
print(donor_id)
donor_bed = pd.read_csv(input_file, sep=',', compression='gzip') # Read the donor's data file (compressed CSV)

# Extract Variant Allele Frequency (VAF) and coverage data from the donor's data
observed_vaf = donor_bed['VAF']
cov=np.array(donor_bed['coverage']) # Coverage data
min_reads = donor_bed['AD_ALT'].min() # Minimum number of reads for the alternate allele

# Adjust discretization coverage for Wright-Fisher and Exponential models
discretization_cov = np.max([int(np.max(cov)*1.05), 100])
lowest_frequency_allowed = 1/discretization_cov
xdata = np.linspace(lowest_frequency_allowed, 1, discretization_cov+1)
y_wf = f_alpha(xdata, 1.0, 1.0)
y_exp = f_alpha(xdata, 1.0, 2.0)
# Normalize to probabilities
y_prob_exp = y_exp/np.sum(y_exp)
y_prob_wf = y_wf/np.sum(y_wf) 

# Initialize a DataFrame to store final results
results_df=pd.DataFrame([{'donor': donor_id,
                          'observed_n': len(observed_vaf), # Observed number of mutations
                          'cov': np.mean(cov), # Mean coverage
                          'min_reads': min_reads}])

# No informed values for purity and C for the WF model
pur_informed, C_informed = None, None

# Iterate over each model (WF and EXP)
for model in ['WF', 'EXP']:
    print(f'-> Fitting {model} model...')

    y_prob = y_prob_wf if model=='WF' else y_prob_exp
        
    # for i in range(4): # Run the fitting process 4 times for each model
    pur_pred, S_pred, C_pred, scores_pred = mcmc(model, cov, min_reads, iter, observed_vaf, starting_points, xdata, y_prob, pur_informed, C_informed)
    
    # Provide the informed values of purity and C from the WF model to the EXP one
    pur_informed = pur_pred
    C_informed = C_pred

    # Get the index and parameters of the best run based on the score
    index_best = np.argmin(scores_pred) 
    pur_best, S_best, C_best, score_best = pur_pred[index_best], S_pred[index_best], C_pred[index_best], scores_pred[index_best]
    
    # Save the results for each model
    results_df[f'pur_{model}'] = pur_best
    results_df[f'S_{model}'] = S_best
    results_df[f'C_{model}'] = C_best
    results_df[f'score_{model}'] = score_best

# Save the final result to a CSV file
results_df.to_csv(results_dir+donor_id+'.csv', index=False)

print(f'-> Done! :D')