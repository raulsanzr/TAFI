import pandas as pd
import matplotlib.pyplot as plt
from simulations import *
from plot_solution import *

# Load data
out_file = '/home/raul/Documents/projects/TAFI/results/individual/CPCT02020916T.csv'
donor_id = 'CPCT02020916T'
solution = pd.read_csv(out_file)
solution = solution[solution['donor'] == donor_id]

donor_bed = pd.read_csv("/home/raul/Documents/projects/TAFI/data/processed/HMF/CPCT02020916T.bed.gz", sep=',', compression='gzip')

# Extract Variant Allele Frequency (VAF) and coverage data from the donor's data
real_vaf = donor_bed['VAF']
cov = np.array(donor_bed['coverage'])  # Coverage data
cov_val = np.mean(cov)  # Mean coverage value
min_reads = donor_bed['AD_ALT'].min()  # Minimum number of reads for the alternate allele

# Adjust discretization coverage for Wright-Fisher and Exponential models
discretization_cov = np.array([np.max([int(np.max(cov) * 1.05), 100])])
lowest_frequency_allowed = 1 / np.max(discretization_cov)
nr_of_bins = int(1 / lowest_frequency_allowed)
xdata = np.linspace(lowest_frequency_allowed, 1, nr_of_bins + 1)

# Assuming f_alpha and sim_vafs are defined elsewhere
y_wf = f_alpha(xdata, 1.0, 1.0)
y_exp = f_alpha(xdata, 1.0, 2.0)
y_prob_exp = y_exp / np.sum(y_exp)
y_prob_wf = y_wf / np.sum(y_wf)

pur_wf, s_wf, c_wf, score_wf = solution['pur_WF'][0], solution['S_WF'][0], solution['C_WF'][0], solution['score_WF'][0]
pur_exp, s_exp, c_exp, score_exp = solution['pur_EXP'][0], solution['S_EXP'][0], solution['C_EXP'][0], solution['score_EXP'][0]

wf_vaf = sim_vafs(pur_wf, s_wf, c_wf, cov, min_reads, xdata, y_prob_wf)
exp_vaf = sim_vafs(pur_exp, s_exp, c_exp, cov, min_reads, xdata, y_prob_exp)

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
# Wright-Fisher model
plot_histogram(axs[0, 0], real_vaf, wf_vaf, pred_color='tab:green', model='WF')
plot_cdf(axs[1, 0], real_vaf, wf_vaf, pred_color='tab:green', model='WF')
axs[0, 0].set_title(f"Wright-Fisher model\npur={pur_wf:.2f}, S={s_wf}, C={c_wf}\nscore={score_wf:.5f}")
# Exponential model
plot_histogram(axs[0, 1], real_vaf, exp_vaf, pred_color='tab:red', model='EXP')
plot_cdf(axs[1, 1], real_vaf, exp_vaf, pred_color='tab:red', model='EXP')
axs[0, 1].set_title(f"Exponential model\npur={pur_exp:.2f}, S={s_exp}, C={c_exp}\nscore={score_exp:.5f}")
plt.tight_layout()
plt.savefig(f'../results/plots/{donor_id}.png')
