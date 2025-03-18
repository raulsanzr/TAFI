import numpy as np
import matplotlib.pyplot as plt

def plot_cdf(ax, real_vaf, pred_vaf, pred_color, model):
    real_sorted = np.sort(real_vaf)
    pred_sorted = np.sort(pred_vaf)

    real_cdf = np.arange(1, len(real_sorted) + 1) / len(real_sorted)
    pred_cdf = np.arange(1, len(pred_sorted) + 1) / len(pred_sorted)

    ax.plot(real_sorted, real_cdf, label='Observed', linewidth=2,  color='tab:blue')
    ax.plot(pred_sorted, pred_cdf, label='Prediction', linewidth=2, linestyle='--', color=pred_color)
    ax.set_xlabel('Allele Frequency')
    if model == 'WF':
        ax.set_ylabel('Cumulative Probability')

def plot_histogram(ax, real_vaf, pred_vaf, pred_color, model):
    ax.hist(real_vaf, bins=50, alpha=0.5, range=(0,1), label='Observed', color='tab:blue')
    ax.hist(pred_vaf, bins=50, alpha=0.5, range=(0,1), label=f'{model} solution', color=pred_color)
    if model == 'WF':
        ax.set_ylabel('Frequency')
    ax.legend()

def sos_distance(real_vaf, pred_vaf, nbins=100):
    '''
    Calculates the sum of squared differences between the real and predicted VAF histograms
    '''
    real_hist = np.histogram(real_vaf, bins=nbins, range=(0,1))[0]
    pred_hist = np.histogram(pred_vaf, bins=nbins, range=(0,1))[0]
    return np.sum((real_hist - pred_hist)**2)