import os
import sys
import pandas as pd
import numpy as np
from abc import *

wf_vaf = sim_vafs(final_results['pur_WF'][0], final_results['S_WF'][0], final_results['C_WF'][0], cov, min_reads, xdata, y_prob_wf)
wf_score = final_results['score_WF'][0]
exp_vaf = sim_vafs(final_results['pur_EXP'][0], final_results['S_EXP'][0], final_results['C_EXP'][0], cov, min_reads, xdata, y_prob_exp)
exp_score = final_results['score_EXP'][0]

# Creating a 2x2 plot
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

wf_hist=plot_histogram(real_vaf, wf_vaf)
exp_hist=plot_histogram(real_vaf, exp_vaf)

wf_cdf=plot_cdf(real_vaf, wf_vaf)
exp_cdf=plot_cdf(real_vaf, exp_vaf)

plt.tight_layout()
plt.savefig(results_dir + 'plots/' + donor_id + '.png')
plt.show()
