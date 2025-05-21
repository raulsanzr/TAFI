# Import libraries
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
import seaborn as sns
import random
import scipy
from scipy.stats import chi2

# Read the data
df_clinical = pd.read_csv('data/pcawg_donor_clinical_August2016_v9.csv', sep='\t')

df_TGA = pd.read_excel('data/TCGA-CDR-SupplementalTableS1.xlsx')
df_TGA = df_TGA.rename(columns={'bcr_patient_barcode': 'submitted_donor_id'})

df_design = pd.read_csv('data/PCAWG_experiment-design.tsv', sep='\t')
df_design = df_design.rename(columns={'Sample Characteristic[individual]': 'icgc_donor_id'})

# Merge the three data frames
df = pd.merge(df_TGA, df_clinical, on="submitted_donor_id")
df = pd.merge(df, df_design, on='icgc_donor_id')

# Filter by cancer type
# df=df[df['type']=='BRCA']

# Formatting the gender: FEMALE -> 0, MALE -> 1
df['gender'] = df['gender'].replace(to_replace='FEMALE', value=0)
df['gender'] = df['gender'].replace(to_replace='MALE', value=1)

# Removing NA's
df['OS.time'] = df['OS.time'].fillna(0) # 3 NA's
df['age_at_initial_pathologic_diagnosis'] = df['age_at_initial_pathologic_diagnosis'].fillna(0) # 4 NA's

res = pd.read_csv('/home/raul/Documents/projects/TAFI/results/all_PCAWG.csv')

# Predict model: 2 if WF wins, else 1
res['predicted_model'] = np.where(res['score_WF'] > res['score_EXP'], 2, 1)

df_wf = res[res['predicted_model'] == 1].copy()
df_exp = res[res['predicted_model'] == 2].copy()
df_wf['predicted_C'] = df_wf['C_WF']
df_exp['predicted_C'] = df_exp['C_EXP']
df_wf['predicted_S'] = df_wf['S_WF']
df_exp['predicted_S'] = df_exp['S_EXP']
df_wf['predicted_pur'] = df_wf['pur_WF']
df_exp['predicted_pur'] = df_exp['pur_EXP']

res = pd.concat([df_wf, df_exp], axis=0)

res['donor_short'] = res['donor'].str.split('_').str[1]
df = pd.merge(df, res, left_on='icgc_donor_id', right_on='donor_short', how='inner')

df['ajcc_pathologic_tumor_stage'] = df['ajcc_pathologic_tumor_stage'].replace(to_replace='Stage I', value=1)
df['ajcc_pathologic_tumor_stage'] = df['ajcc_pathologic_tumor_stage'].replace(to_replace='Stage IA', value=1)
df['ajcc_pathologic_tumor_stage'] = df['ajcc_pathologic_tumor_stage'].replace(to_replace='Stage IB', value=1)
df['ajcc_pathologic_tumor_stage'] = df['ajcc_pathologic_tumor_stage'].replace(to_replace='Stage II', value=2)
df['ajcc_pathologic_tumor_stage'] = df['ajcc_pathologic_tumor_stage'].replace(to_replace='Stage IIA', value=2)
df['ajcc_pathologic_tumor_stage'] = df['ajcc_pathologic_tumor_stage'].replace(to_replace='Stage IIB', value=2)
df['ajcc_pathologic_tumor_stage'] = df['ajcc_pathologic_tumor_stage'].replace(to_replace='Stage III', value=3)
df['ajcc_pathologic_tumor_stage'] = df['ajcc_pathologic_tumor_stage'].replace(to_replace='Stage IIIA', value=3)
df['ajcc_pathologic_tumor_stage'] = df['ajcc_pathologic_tumor_stage'].replace(to_replace='Stage IIIB', value=3)
df['ajcc_pathologic_tumor_stage'] = df['ajcc_pathologic_tumor_stage'].replace(to_replace='Stage IIIC', value=3)
df['ajcc_pathologic_tumor_stage'] = df['ajcc_pathologic_tumor_stage'].replace(to_replace='Stage IV', value=4)
df['ajcc_pathologic_tumor_stage'] = df['ajcc_pathologic_tumor_stage'].replace(to_replace='Stage IVA', value=4)
df['ajcc_pathologic_tumor_stage'] = df['ajcc_pathologic_tumor_stage'].replace(to_replace='Stage IVB', value=4)
df['ajcc_pathologic_tumor_stage'] = df['ajcc_pathologic_tumor_stage'].replace(to_replace='Stage IVC', value=4)
df['ajcc_pathologic_tumor_stage'] = df['ajcc_pathologic_tumor_stage'].replace(to_replace='[Not Available]', value=0)
df['ajcc_pathologic_tumor_stage'] = df['ajcc_pathologic_tumor_stage'].replace(to_replace='[Not Applicable]', value=0)
df['ajcc_pathologic_tumor_stage'] = df['ajcc_pathologic_tumor_stage'].replace(to_replace='[Discrepancy]', value=0)

# Reduced data frame with the desired parameters
cox_df=df[['age_at_initial_pathologic_diagnosis', 'ajcc_pathologic_tumor_stage' ,'OS.time','gender','OS', 'predicted_model', 'type']]

cox_df = pd.get_dummies(cox_df, columns=['type'], drop_first=True)


# Fitting the model and printing the results
cph = CoxPHFitter()
cph.fit(cox_df, duration_col = 'OS.time', event_col='OS')
cph.print_summary()

import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import median_survival_times
from lifelines import KaplanMeierFitter

class_wf = cph.predict_survival_function(cox_df[cox_df['predicted_model'] == 1]).mean(axis=1)
class_exp = cph.predict_survival_function(cox_df[cox_df['predicted_model'] == 2]).mean(axis=1)

# Plot survival probability
plt.figure(figsize=(8, 4))
plt.plot(class_wf.index, class_wf.values, label='WF', color='tab:green')
plt.plot(class_exp.index, class_exp.values, label='EXP', color='tab:red')
plt.axvline(median_survival_times(class_wf), linestyle='--', color='tab:green', label='Median survival (WF)')
plt.axvline(median_survival_times(class_exp), linestyle='--', color='tab:red', label='Median survival (EXP)')
plt.xlabel('Time since diagnosis (days)')
plt.ylabel('Survival probability')
plt.title('Kaplan-Meier survival curves per evolutionary model')
plt.legend()
plt.show()