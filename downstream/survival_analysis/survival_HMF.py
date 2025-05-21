# Import libraries
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import median_survival_times
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Read metadata and merge pre_biopsy data
metadata=pd.read_csv('data/metadata_update.tsv',sep="\t", header=0)
pre_biopsy=pd.read_csv('data/pre_biopsy_drugs_update.tsv',sep="\t", header=0)
metadata['patientIdentifier'] = metadata['sampleId'].str[:12]
df_survival=pd.merge(pre_biopsy, metadata, on='patientIdentifier')
df_survival=df_survival[df_survival['biopsyDate'].isnull()==False] #delete all that dont have a given biopsyDate
df_survival=df_survival[['sampleId','hasSystemicPreTreatment','hasRadiotherapyPreTreatment','biopsySite', 'startDate','biopsyDate','birthYear','consolidatedTreatmentType','primaryTumorLocation','gender','type','deathDate', 'primaryTumorSubType','tumorPurity', 'treatmentEndDate']]
df_survival['status'] = 1 # set status as dead
df_survival['deathDate'] = df_survival['deathDate'].fillna(0) # If donor is still alive
df_survival['status'][df_survival['deathDate']==0] = 0 # set alive donors as alive

# Calculating survival times
# Assumption: If there is not death reported, we will asume that the donor is still alive at the date of the last update of the metadata.

df_survival['biopsyDate'] = pd.to_datetime(df_survival['biopsyDate'])
## OS (Overall Survival): Death date - Biopsy date
### Dead donors
df_survival['deathDate'] = pd.to_datetime(df_survival['deathDate'], errors='coerce')
mask = df_survival['status']==1
df_survival.loc[mask, 'OS_time'] = (df_survival['deathDate'] - df_survival['biopsyDate']).dt.days
### Alive donors
last_update = pd.to_datetime('2023-04-04') # If there is not deathDate, I substract it by the last documented update
mask = pd.notnull(df_survival['startDate']) & (df_survival['status'] == 0) 
df_survival.loc[mask, 'OS_time'] = (last_update - df_survival['biopsyDate']).dt.days
df_survival = df_survival[df_survival['OS_time']>0.0]

## PFI (Progression-free interval): treatmentEndDate - Biopsy date
## If treatment ended
df_survival['treatmentEndDate'] = pd.to_datetime(df_survival['treatmentEndDate'])
mask = pd.notnull(df_survival['treatmentEndDate'])
df_survival.loc[mask, 'PFI_time'] = (df_survival['treatmentEndDate'] - df_survival['biopsyDate']).dt.days
### If treatment is still ongoing
mask = pd.isnull(df_survival['treatmentEndDate'])
df_survival.loc[mask, 'PFI_time'] = (last_update - df_survival['biopsyDate']).dt.days
df_survival = df_survival[df_survival['PFI_time']>0]
df_survival = df_survival.drop_duplicates(subset='sampleId', keep="first")

df_out = pd.read_csv('../../results/all_HMF.csv', sep=",", header=0)
# if score_WF < score_EXP, mode is WF otherwise EXP
df_out['class'] = np.where(df_out['score_WF'] < df_out['score_EXP'], 'WF', 'EXP')
df_out['sampleId'] = df_out['donor']
df_class = df_out[['sampleId', 'class']]

# Format the data
# Merge the prediction
df = pd.merge(df_survival, df_class, on='sampleId')
# Remove NA's
df = df.dropna(subset=['birthYear', 'gender', 'OS_time', 'PFI_time'])
# class: WF -> 0, EXP -> 1
df = df[df['class']!='CC']
df['class'] = df['class'].replace(to_replace='WF', value=0)
df['class'] = df['class'].replace(to_replace='EXP', value=1)
# Calculate age at diagnosis based on birthYear
df['year_of_diagnosis']=df['biopsyDate'].astype(str)
df['year_of_diagnosis'] = df['year_of_diagnosis'].str[:4]
df['year_of_diagnosis']=df['year_of_diagnosis'].astype(int)
df['birthYear']=df['birthYear'].astype(int)
df['age'] = (df['year_of_diagnosis']-df['birthYear']) # change to biopsy date. 
df['age']=df['age'].astype(int)
# # gender: female -> 0, male -> 1
df['gender'] = df['gender'].replace(to_replace='female', value=0)
df['gender'] = df['gender'].replace(to_replace='male', value=1)
# # Add vital state
# df['type'] = df['primaryTumorLocation'] + df['primaryTumorSubType']
# Filter by cancer type
# df = df[df['primaryTumorLocation']=='Stomach']
# print(pd.crosstab(index=df['class'], columns=df['type']))
# df = df[df['consolidatedTreatmentType']=='Immunotherapy']

# Survival analysis: all cancer types
# Reduced data frame with the desired parameters
cox_df = df[['age','gender','OS_time','status', 'class', 'primaryTumorLocation']] 

cox_df = pd.get_dummies(cox_df, columns=['primaryTumorLocation'], drop_first=True)


# Fitting the model and printing the results
cph = CoxPHFitter()
cph.fit(cox_df, duration_col = 'OS_time', event_col='status')
cph.print_summary()

# Plot survival probability
class_wf = cph.predict_survival_function(df[df['class'] == 0]).mean(axis=1)
class_exp = cph.predict_survival_function(df[df['class'] == 1]).mean(axis=1)

# Plot survival probability
plt.plot(class_wf.index, class_wf.values, label='WF')
plt.plot(class_exp.index, class_exp.values, label='EXP')
plt.axvline(median_survival_times(class_wf), linestyle='--', color='tab:green', label='Median survival (WF)')
plt.axvline(median_survival_times(class_exp), linestyle='--', color='tab:red', label='Median survival (EXP)')
plt.xlabel('Time (days)')
plt.ylabel('Survival probability')
plt.legend()
plt.show()