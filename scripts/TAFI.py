import sys
import numpy as np
import pandas as pd
from abc_functions import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# --- DIRECTORIES --- #
results_dir = '../results/'
os.makedirs(results_dir, exist_ok=True)  # Create results directory if it doesn't exist

# --- READ THE DATA --- #
input_file= sys.argv[1]
donor_id = input_file.split('/')[-1].split('.')[0]  # Extract donor ID from the file path
print(donor_id)
donor_bed = pd.read_csv(input_file, sep=',', compression='gzip') # Read the donor's data file (compressed CSV)

# Extract Variant Allele Frequency (VAF) and coverage data
vaf = donor_bed['VAF']
# Create a histogram of VAF values (100 bins, range 0 to 1)
real_hist = np.histogram(vaf, bins=100, range=(0, 1))[0]
cov = np.array(donor_bed['coverage'])  # Coverage data
cov_val = np.mean(cov)  # Mean coverage value
min_reads = donor_bed["AD_ALT"].min()  # Minimum number of reads for the alternate allele

# Define parameters for discretization and frequency analysis
discr_cov_mean = 1000

# Define bins for histogram analysis
binss = np.linspace(0, 1, 101)

# Adjust discretization coverage for Wright-Fisher and Exponential models
wf_discr_cov_mean = 1000
exp_discr_cov_mean = np.max([int(np.max(cov) * 1.05), 100])
discretization_cov = np.array([exp_discr_cov_mean])
lowest_frequency_allowed = 1 / np.max(discretization_cov)
nr_of_bins = int(1 / lowest_frequency_allowed)
exp_xdata = np.linspace(lowest_frequency_allowed, 1, nr_of_bins + 1)
xdata = np.linspace(lowest_frequency_allowed, 1, nr_of_bins + 1)

# Calculate theoretical distributions (Wright-Fisher and Exponential)
y_wf = f_alpha(xdata, 1.0, 1.0)  # Wright-Fisher distribution
y_exp = f_alpha(xdata, 1.0, 2.0)  # Exponential distribution
y_prob_exp = y_exp / np.sum(y_exp)  # Normalize to probabilities
y_prob_wf = y_wf / np.sum(y_wf)  # Normalize to probabilities

# --- RUN --- #
max_steps = 100  # Maximum number of steps for the fitting process
collected_data_size = 100 # Define the size of the collected data

# Initialize a DataFrame to store final results
final_results = pd.DataFrame([{
    'donor': donor_id,
    'cov': cov_val,
    'min_reads': min_reads
}])

# Define models to be tested (Wright-Fisher and Exponential) and their specific parameters
models = {
    "WF": {
        "y_prob": y_prob_wf,
        "xdata": xdata,
    },
    "EXP": {
        "y_prob": y_prob_exp,
        "xdata": exp_xdata,
    }
}

# Initialize values of purity, S, and C as None
pred_purity, pred_S, pred_C = None, None, None

# Iterate over each model (WF and EXP)
for test_model, params in models.items():
    results = pd.DataFrame()  # Initialize a DataFrame to store intermediate results
    y_prob = params["y_prob"]
    xdata = params["xdata"]
    run_function = params["run_function"]
    
    # Run the fitting process 4 times for each model
    for i in range(4):
        pur_pred, S_pred, C_pred, scores_pred = run_fit(
            test_model, cov, min_reads, max_steps, real_hist, collected_data_size,
            xdata, y_prob, pred_purity, pred_C)
        
        # Store the results of each run
        ensemble_results = pd.DataFrame({
            'purity_pred': pur_pred,
            'S_pred': S_pred,
            'C_pred': C_pred,
            'scores_final': scores_pred,
            'chain': i})
        results = pd.concat([results, ensemble_results])
    
    df = results[['purity_pred', 'S_pred', 'C_pred', 'scores_final']].copy()

    # scaled_colnames = [col + '_scaled' for col in df.columns]
    # # Scale the results using StandardScaler
    # df[scaled_colnames] = StandardScaler().fit_transform(df)
    
    # # Perform clustering using KMeans to identify the best cluster
    # lowest_cluster_means, highest_cluster_nr = [], []
    # for i in range(2, 4):
    #     kmeans = KMeans(n_clusters=i, random_state=20, n_init=10).fit(df[scaled_colnames])
    #     df[f'cluster{i}'] = kmeans.labels_
    #     group_means = df.groupby(f'cluster{i}')['scores_final'].mean()
    #     lowest_cluster_means.append(group_means.min())
    #     highest_cluster_nr.append(group_means.idxmax())
    
    # # Select the best cluster based on the lowest mean score
    # K = np.argmin(lowest_cluster_means) + 2
    # cluster_index = df[f'cluster{K}'].unique()[np.argmin([df[df[f'cluster{K}'] == c]['scores_final'].mean() for c in df[f'cluster{K}'].unique()])]
    # df = df[df[f'cluster{K}'] == cluster_index]
    
    # # Calculate z-scores to filter out outliers
    # mean_score, std_score = df['scores_final'].mean(), df['scores_final'].std()
    # df['z_score'] = (df['scores_final'] - mean_score) / std_score
    # extra_correction_df = df[(df['z_score'] >= -1.67) & (df['z_score'] <= 1.67)]
    
    # # Use the filtered data if there are enough samples
    # if len(extra_correction_df) >= 3:
    #     df = extra_correction_df
    
    # Calculate final predictions for purity, S, and C
    pred_purity, pred_S, pred_C = df['purity_pred'].mean(), int(df['S_pred'].mean()), int(df['C_pred'].mean())
    final_results[f'pur_{test_model.lower()}'] = pred_purity
    final_results[f'C_{test_model.lower()}'] = pred_C
    final_results[f'S_{test_model.lower()}'] = pred_S
    final_results[f'score_{test_model.lower()}'] = df['scores_final'].mean()

# Save the final results to a CSV file
final_results.to_csv(results_dir + donor_id + '.csv', index=False)