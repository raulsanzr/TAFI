import os
import pandas as pd
import matplotlib.pyplot as plt
from modules.abc import *
from modules.metrics import *

results_dir = '../results/individual_MC3/'
data_dir = '../data/processed/MC3/'
individual_plots_dir = '../results/plots/individual_MC3/'
os.makedirs(individual_plots_dir, exist_ok=True)

results_merged = pd.DataFrame()
for file in os.listdir(results_dir):
    # Read the solution
    solution = pd.read_csv(results_dir+file)
    # Get the parameters of the best run
    pur_wf, s_wf, c_wf, score_wf = solution['pur_WF'][0], solution['S_WF'][0], solution['C_WF'][0], solution['score_WF'][0]
    pur_exp, s_exp, c_exp, score_exp = solution['pur_EXP'][0], solution['S_EXP'][0], solution['C_EXP'][0], solution['score_EXP'][0]

    # Read the donor's real data (needed to get the observed VAF). Same process as in tafi.py
    donor_id = solution['donor'][0]
    donor_bed = pd.read_csv(data_dir+donor_id+'.bed.gz', sep=',', compression='gzip')
    real_vaf = donor_bed['VAF']
    cov = np.array(donor_bed['coverage']) # Coverage data
    cov_val = np.mean(cov) # Mean coverage value
    min_reads = donor_bed['AD_ALT'].min() # Minimum number of reads for the alternate allele
    xdata = np.linspace(1/np.max(cov), 1, np.max(cov)+1) # Generate an array of possible frequency values given the maximum observed coverage
    # Assign the probability of observing a mutation at each frequency value
    y_wf = f_alpha(xdata, 1)
    y_exp = f_alpha(xdata, 2)
    y_prob_exp = y_exp/np.sum(y_exp)
    y_prob_wf = y_wf/np.sum(y_wf) 

    # Simulate the VAFs with the best parameters
    wf_vaf = simulate_vaf(pur_wf, s_wf, c_wf, cov, min_reads, xdata, y_prob_wf)
    exp_vaf = simulate_vaf(pur_exp, s_exp, c_exp, cov, min_reads, xdata, y_prob_exp)

    # Compute more scores
    solution['sos_WF'] = sos_distance(real_vaf, wf_vaf)
    solution['sos_EXP'] = sos_distance(real_vaf, exp_vaf)

    # Plot the results
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    # Wright-Fisher model
    plot_histogram(axs[0, 0], real_vaf, wf_vaf, pred_color='tab:green', model='WF')
    plot_cdf(axs[1, 0], real_vaf, wf_vaf, pred_color='tab:green', model='WF')
    axs[0, 0].set_title(f"Wright-Fisher model\npur={pur_wf:.2f}, S={s_wf}, C={c_wf}\ndistance={score_wf:.5f}")
    # Exponential model
    plot_histogram(axs[0, 1], real_vaf, exp_vaf, pred_color='tab:red', model='EXP')
    plot_cdf(axs[1, 1], real_vaf, exp_vaf, pred_color='tab:red', model='EXP')
    axs[0, 1].set_title(f"Exponential model\npur={pur_exp:.2f}, S={s_exp}, C={c_exp}\ndistance={score_exp:.5f}")
    plt.tight_layout()
    plt.savefig(f'../results/plots/individual_MC3/{donor_id}.png')
    plt.close()

    results_merged = pd.concat([results_merged, solution], ignore_index=True)

results_merged.to_csv('../results/all_MC3.csv', index=False)

# distribution of differences in scores
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.hist(results_merged['score_WF'] - results_merged['score_EXP'], bins=50, color='tab:blue')
ax.set_xlabel('WF score - EXP score')
ax.axvline(x=0, color='red', linestyle='--')
plt.title('Distribution of differences in scores')
plt.tight_layout()
plt.savefig('../results/plots/score_diff.png')
plt.close()

# S vs C scatterplot
# fig, ax = plt.subplots(1, 1, figsize=(5, 3))
# ax.scatter(results_merged['S_WF'], results_merged['C_WF'], c='tab:green', label='WF', alpha=0.5)
# ax.scatter(results_merged['S_EXP'], results_merged['C_EXP'], c='tab:red', label='EXP', alpha=0.5)
# ax.set_xlabel('S')
# ax.set_ylabel('C')
# plt.title('S vs C')
# plt.tight_layout()
# plt.savefig('../results/plots/SvsC.png')
# plt.show()