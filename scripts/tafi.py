import os
import sys
import pandas as pd
import numpy as np
from search_params import *

# Search parameters
max_steps=100 # Maximum number of steps for the fitting process
collected_data_size=100 # Define the size of the collected data

# Folder structure
results_dir='../results/individual/'
os.makedirs(results_dir, exist_ok=True) # Create results directory if it doesn't exist

# Read the data
input_file=sys.argv[1]
donor_id=input_file.split('/')[-1].split('.')[0] # Extract donor ID from the file path
print(donor_id)
donor_bed=pd.read_csv(input_file, sep=',', compression='gzip') # Read the donor's data file (compressed CSV)

# Extract Variant Allele Frequency (VAF) and coverage data from the donor's data
real_vaf=donor_bed['VAF']
cov=np.array(donor_bed['coverage']) # Coverage data
cov_val=np.mean(cov)  # Mean coverage value
min_reads=donor_bed['AD_ALT'].min() # Minimum number of reads for the alternate allele

# Adjust discretization coverage for Wright-Fisher and Exponential models
discretization_cov=np.array([np.max([int(np.max(cov)*1.05), 100])])
lowest_frequency_allowed=1/np.max(discretization_cov)
nr_of_bins=int(1 / lowest_frequency_allowed)
xdata=np.linspace(lowest_frequency_allowed, 1, nr_of_bins+1)
y_wf=f_alpha(xdata, 1.0, 1.0)
y_exp=f_alpha(xdata, 1.0, 2.0)
# Normalize to probabilities
y_prob_exp=y_exp / np.sum(y_exp)
y_prob_wf=y_wf / np.sum(y_wf) 

# Initialize a DataFrame to store final results
final_results=pd.DataFrame([{'donor':donor_id,
                             'cov':cov_val,
                             'min_reads':min_reads}])

# No informed values for purity and C for the WF model
informed_pur, informed_C=None, None

# Iterate over each model (WF and EXP)
for test_model in ['WF', 'EXP']:

    y_prob = y_prob_wf if test_model == 'WF' else y_prob_exp
        
    # for i in range(4): # Run the fitting process 4 times for each model
    pur_pred, S_pred, C_pred, scores_pred=run_fit(test_model, cov,min_reads, max_steps, real_vaf, 
                                                  collected_data_size, xdata, y_prob, informed_pur, informed_C)
    
    # Provide the informed values of purity and C from the WF model to the EXP one
    informed_pur=pur_pred
    informed_C=C_pred

    # Get the index and parameters of the best run based on the score
    best_index=np.argmin(scores_pred) 
    best_pur, best_S, best_C, best_score=pur_pred[best_index], S_pred[best_index], C_pred[best_index], scores_pred[best_index]
    
    # Save the results for each model
    final_results[f'pur_{test_model}']=best_pur
    final_results[f'S_{test_model}']=best_S
    final_results[f'C_{test_model}']=best_C
    final_results[f'score_{test_model}']=best_score

# Save the final result to a CSV file
final_results.to_csv(results_dir+donor_id+'.csv', index=False)