import numpy as np
import matplotlib.pyplot as plt

def plot_cdf(real_vaf, pred_vaf):

    real_sorted = np.sort(real_vaf)
    pred_sorted = np.sort(pred_vaf)

    real_cdf = np.arange(1, len(real_sorted) + 1) / len(real_sorted)
    pred_cdf = np.arange(1, len(pred_sorted) + 1) / len(pred_sorted)

    plot(real_sorted, real_cdf, label='Real', linewidth=2)
    plot(pred_sorted, pred_cdf, label='Prediction', linewidth=2, linestyle='--')
    set_xlabel('Allele Frequency')
    set_ylabel('Cumulative Probability')
    legend()
    set_title('Real vs CDF')
    grid(True)

def plot_histogram(real_vaf, pred_vaf):
    hist(real_vaf, bins=30, alpha=0.5, label='Real', density=True)
    hist(pred_vaf, bins=30, alpha=0.5, label='Predicted', density=True)
    set_xlabel('Allele Frequency')
    set_ylabel('Density')
    legend()
    set_title('Real vs Distribution')