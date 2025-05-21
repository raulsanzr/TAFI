from ast import Try
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import glob
import pandas as pd
import os
import gzip
import sys
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import subprocess

def is_empty(file):
    '''
    Check if a gzipped file is empty.
    '''
    try:
        with gzip.open(file, 'rb') as f:
            return f.seek(0, whence=2) == 0
    except:
        return True

def plot_vaf(vaf, freq_range, dens_range, freq_peak, dens_peak, donor, thresh):
    '''
    Plot the vaf of a donor.
    Args:
        vaf (list): Variant allele frequency.
        freq_range (list): Range of frequencies of the vaf distribution.
        dens_range (list): Density of the vaf distribution.
        freq_peak (list): Position of the detected peaks in the x-axis.
        dens_peak (list): Position of the detected peaks in the y-axis.
        donor (str): Name of the donor.
    '''
    plt.hist(x=vaf, bins=50, range=(0, 1), density=True) # plot the variant allele frequency histogram
    plt.plot(freq_range, dens_range, color='black') # plot the probability density function
    plt.scatter(freq_peak, dens_peak, color='red', label='Peaks') # plot the detected peaks
    plt.axvline(x=thresh, color='r', linestyle='dashed', linewidth=2) # threshold line
    plt.xlabel('Frequency')
    plt.ylabel('Density')
    plt.title(donor)
    plt.xlim(0,1)
    plt.savefig(plots_dir + donor + '.png') # save plot
    plt.close()

def selector(bed_file, cohort):
    '''
    Classify a donor based on its vaf.
    Args: 
        bed_file (str): Path of the TAFI-formatted file containing the mutations of a donor.
        cohort (str): Name of the cohort to which the donor belongs [PCAWG/HMF/MC3].
    '''
    # fixed parameters
    min_reads = 2 
    max_mut = 0.10
    low_thresh = 0.01
    min_mut = 1000
    max_freq = 0.55
    thresh_margin = 0.05

    # extract the patient name from the filename
    donor_name = str(bed_file).split('/')[-1].split('.')[0]

    # Check if file is empty, if so, return discard category
    if is_empty(bed_file):
        return 'discard'
    
    # Read the file into a DataFrame
    df = pd.read_csv(bed_file, sep='\t', compression='gzip', header=0, low_memory=False)

    # Specific filter for MC3 cohort: exclude mitochondrial and sex chromosomes
    if 'MC3' in cohort:
        df = df[df['chr'].isin([f'chr{i}' for i in range(1, 23)])]

    # Filter out mutations with low alternative read count
    df = df[df['AD_ALT'] > min_reads]

    # Discard if number of mutations is below minimum threshold
    if len(df) < min_mut:
        return 'discard'
    
    # Calculate variant allele frequency (VAF) and add a "PASS" filter to each mutation
    df['VAF'] = df['AD_ALT'] / df['coverage']
    df['filter'] = 'PASS'

    # Estimate the density function of VAF
    density = gaussian_kde(list(df['VAF']), bw_method='scott')
    freq_range = np.linspace(min(df['VAF']), max(df['VAF']), 1000)  # Evaluate density at 1000 points
    dens_range = density(freq_range)

    # Calculate the difference in density to identify peaks and valleys
    diff_y = np.diff(dens_range)

    # Identify peaks and valleys based on sign changes in the density difference
    peak_indices = np.where((diff_y[:-1] > 0) & (diff_y[1:] < 0))[0] + 1
    freq_peak = freq_range[peak_indices]
    dens_peak = dens_range[peak_indices]
    valley_indices = np.where((diff_y[:-1] < 0) & (diff_y[1:] > 0))[0] + 1
    valley_freq = freq_range[valley_indices]

    # Remove noise by filtering out peaks with very low density
    freq_peak = freq_peak[dens_peak > low_thresh * max(dens_peak)]
    dens_peak = dens_peak[dens_peak > low_thresh * max(dens_peak)]

    # Check conditions for classifying as segregating subclone or passing sample
    if len(freq_peak) <= 2 and all(i <= max_freq for i in freq_peak):
        coverage = np.mean(df['AD_ALT'] + df['AD_REF'])
        mr = min(df['AD_ALT'])
        thresh = mr / coverage + thresh_margin

        if (len(freq_peak) == 2) and (freq_peak[0] > thresh):
            return 'segregating_subclone'
        else:
            plot_vaf(df['VAF'], freq_range, dens_range, freq_peak, dens_peak, donor_name, thresh)
            df.to_csv(out_dir + donor_name + '.bed.gz', compression='gzip', sep=',', index=False)
            return 'pass'
    
    # Filtered peaks by frequency for further processing
    filtered_freq_peak = [value for value in freq_peak if value < max_freq]
    try:
        last_valley_freq = valley_freq[len(filtered_freq_peak) - 1]
    except IndexError:
        last_valley_freq = 0
    
    # Count mutations above the last valley frequency and filter mutations accordingly
    N = len([mut for mut in df['VAF'] if mut > last_valley_freq])
    df_flt = df
    df_flt['filter'] = np.where(df_flt['VAF'] < last_valley_freq, "PASS", "CNV")
    df_flt = df_flt[df_flt['filter'] == "PASS"]

    # Check if filtered mutations meet the max mutation frequency threshold
    if N / len(df['VAF']) < max_mut:
        freq_peak_flt = []
        dens_peak_flt = []

        # Keep peaks below max frequency for final processing
        for i in range(len(freq_peak)):
            if freq_peak[i] <= max_freq:
                freq_peak_flt.append(freq_peak[i])
                dens_peak_flt.append(dens_peak[i])

        if len(freq_peak_flt) <= 2:
            coverage = np.mean(df_flt['AD_ALT'] + df_flt['AD_REF'])
            mr = min(df_flt['AD_ALT'])
            thresh = mr / coverage + thresh_margin

            if (len(freq_peak_flt) == 2) and (freq_peak_flt[0] > thresh):
                return 'segregating_subclone'
            else:
                plot_vaf(df_flt['VAF'], freq_range, dens_range, freq_peak_flt, dens_peak_flt, donor_name, thresh)
                df_flt.to_csv(out_dir + donor_name + '.bed.gz', compression='gzip', sep=',', index=False)
                return 'pass'
    
    return 'discard'

def process_file(file, cohort):
    '''
    Process a single file and return its classification category.
    '''
    category = selector(file, cohort)
    return category

# Set up directories and arguments
cohort = sys.argv[1]
home_dir = '..'
data_dir = home_dir + '/data/raw/' + cohort + '/'
out_dir = home_dir + '/data/filtered/' + cohort + '/'
plots_dir = home_dir + '/data/filtered/plots/' + cohort + '/'

# Create output directories if they don't exist
os.makedirs(out_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Dictionary to track classifications
classification = {'pass': 0, 'segregating_subclone': 0, 'discard': 0}
files = glob.glob(data_dir + '*')  # List all files in the data directory

# Use all available CPU cores for parallel processing
num_cores = multiprocessing.cpu_count()

# Process files in parallel using a pool of worker processes
with ProcessPoolExecutor(max_workers=num_cores) as executor:
    futures = {executor.submit(process_file, file, cohort): file for file in files}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
        category = future.result()
        classification[category]+=1

# Save classification summary with current date and time
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(home_dir+"/results/classification.txt", "a") as f:
    f.write(f"Cohort: {cohort}\n")
    f.write(f"Date: {current_time}\n")
    f.write(f"Classification: {classification}\n")
    f.write("\n")

# Print out classification counts
for key, value in classification.items():
    print(key, value)