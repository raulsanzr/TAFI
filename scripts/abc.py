import sys
import numpy as np
import pandas as pd
from abc_functions import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

collected_data_size=100
folder_path = "smc_real_2609_"+str(collected_data_size)+"/"

folderpath=folder_path
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
print(sys.argv)

sim_index = int(sys.argv[1]) - 1

home_dir = '..'
in_dir= home_dir + '/data_filtered'
out_dir = home_dir + '/results/partial_results/'

os.makedirs(out_dir, exist_ok=True)

donors=pd.read_csv(in_dir + "/ids.txt", sep = '\t',header=None)[0]
donor=donors[sim_index]
donor_id = donor.split('/')[1].split('.')[0]

file_path = in_dir + donor
whole_df= pd.read_csv(file_path, sep = ',', compression='gzip')
vaf=whole_df['VAF']
real_hist=np.histogram(vaf, bins = 100,range=(0,1))[0]
cov=whole_df['coverage']
cov=np.array(cov)
cov_val=np.mean(cov)
cov_str=str(np.mean(cov))
min_reads=whole_df["AD_ALT"].min() 
discr_cov_mean=1000
orig_discr_cov_mean=discr_cov_mean
discretization_cov=np.array([discr_cov_mean])
lowest_frequency_allowed=1/np.max(discretization_cov)
nr_of_bins=int(1/lowest_frequency_allowed)
xdata=np.linspace(lowest_frequency_allowed,1,nr_of_bins+1)

y_wf = f_alpha(xdata,1.0,1.0)
y_exp = f_alpha(xdata,1.0,2.0)

y_prob_exp=y_exp/np.sum(y_exp)
y_prob_wf=y_wf/np.sum(y_wf)

binss = np.linspace(0, 1, 101)

observed_hist=real_hist

wf_discr_cov_mean=1000
exp_discr_cov_mean=np.max([int(np.max(cov)*1.05),100])
discretization_cov=np.array([exp_discr_cov_mean])
lowest_frequency_allowed=1/np.max(discretization_cov)
nr_of_bins=int(1/lowest_frequency_allowed)
exp_xdata=np.linspace(lowest_frequency_allowed,1,nr_of_bins+1)
y_exp = f_alpha(exp_xdata, 1.0,2.0)
y_prob_exp=y_exp/np.sum(y_exp)

################################################
wf_results=pd.DataFrame()
test_model="WF"
y_prob=y_prob_wf
for i in range(0,4):
    pur_pred,S_pred,C_pred,scores_pred=run_abc(test_model,cov,min_reads,observed_hist,collected_data_size,xdata,y_prob)
    wf_ensemble_results = pd.DataFrame({   'purity_pred': pur_pred,  'S_pred': S_pred,  'C_pred': C_pred,   'scores_final': scores_pred } )
    wf_final=wf_ensemble_results#[wf_ensemble_results ['mcmc_steps']==max_steps]
    wf_final['chain']=i
    wf_results=pd.concat([wf_results,wf_final])
    
wf_results['discr_cov_mean']=discr_cov_mean
#wf_results.to_csv(folder_path+"sim_"+str(sim_index)+"_"+test_model+"_pred_first_fit.csv")
df=wf_results

df=df[['purity_pred', 'S_pred', 'C_pred', 'scores_final']]
df['purity_pred_scaled']=df['purity_pred']
df['S_pred_scaled']=df['S_pred']
df['C_pred_scaled']=df['C_pred']
df['scores_final_scaled']=df['scores_final']

df_copy=df

# Define colors for each cluster
colors_2 = ['red', 'blue']
colors_3 = ['red', 'blue', 'green']
colors_4 = ['red', 'blue', 'green', 'purple']

features = ['purity_pred_scaled', 'S_pred_scaled', 'C_pred_scaled', 'scores_final_scaled']
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
# Perform KMeans clustering with 2 clusters
kmeans_2 = KMeans(n_clusters=2, random_state=42,n_init=10).fit(df[features])
df['cluster2'] = kmeans_2.labels_
# Perform KMeans clustering with 3 clusters
kmeans_3 = KMeans(n_clusters=3, random_state=42,n_init=10).fit(df[features])
df['cluster3'] = kmeans_3.labels_
# Perform KMeans clustering with 4 clusters
kmeans_4 = KMeans(n_clusters=4, random_state=42,n_init=10).fit(df[features])
df['cluster4'] = kmeans_4.labels_


lowest_cluster_means=[]
lowest_cluster_nr=[]
highest_cluster_nr=[]
for i in range(2,4):
    group_means = df.groupby('cluster'+str(i))['scores_final'].mean()
    lowest_cluster_nr.append(group_means.idxmin())
    lowest_cluster_means.append(group_means.min())
    highest_cluster_nr.append(group_means.idxmax()) #what is the cluster with highest average score

max_k=np.argmax(lowest_cluster_means)
K_max=max_k+2
cluster_max_index=highest_cluster_nr[max_k]
df_copy=df_copy[df_copy["cluster"+str(K_max)]==cluster_max_index]
#plt.scatter(df.S_pred, df.C_pred,label="outliers")

min_k= np.argmin(lowest_cluster_means)
K=min_k+2
final_scores=[]
Ks=df["cluster"+str(K)].unique()
for cluster_index in Ks:
    cdf=df[df["cluster"+str(K)]==cluster_index]

    pred_purity=cdf.purity_pred.mean()
    pred_S=int(cdf.S_pred.mean())
    pred_C=int(cdf.C_pred.mean())
#do some repetats    
    nr_of_repeats=5
    current_score=compute_distance(observed_hist,nr_of_repeats,pred_purity,pred_S,pred_C,cov,binss,min_reads,xdata,y_prob)
    #print(current_score)
    final_scores.append(current_score)
lowest_score_ind=np.argmin(final_scores)
#final_models_score.append(final_scores[lowest_score_ind])
cluster_index=Ks[lowest_score_ind]
cdf=df[df["cluster"+str(K)]==cluster_index]
#print("best cluster result, uncorrected:")       
pred_purity=cdf.purity_pred.mean()
pred_S=int(cdf.S_pred.mean())
pred_C=int(cdf.C_pred.mean())

df=cdf
cluster_size=len(df)
#remove x samples with z score correction on the final score
mean_score = df['scores_final'].mean()
std_score = df['scores_final'].std()
df['z_score'] = (df['scores_final'] - mean_score) / std_score
extra_correction_df = df[(df['z_score'] >= -1.67) & (df['z_score'] <= 1.67)]
new_cluster_size=len(extra_correction_df)
print("correction would be from ",cluster_size," to ",new_cluster_size)
if new_cluster_size>=3:
    df=extra_correction_df
    cluster_size=new_cluster_size
    
pred_purity=df.purity_pred.mean()
pred_S=int(df.S_pred.mean())
pred_C=int(df.C_pred.mean())
score_wf0=df.scores_final.mean()

################################
final_results=pd.DataFrame()


final_results['cov']=[cov_val]
final_results['min_reads']=min_reads
final_results['discr_cov_mean']=discr_cov_mean
final_results['exp_discr_cov_mean']=exp_discr_cov_mean
final_results['orig_discr_cov_mean']=orig_discr_cov_mean
final_results['donor']=donor_id
final_results['sim_index']=sim_index

##############################################
#pred values
final_results['pred_purity_wf']=pred_purity
final_results['pred_C_wf']=pred_C
final_results['pred_S_wf0']=pred_S
final_results['score_wf0']=score_wf0
final_results['observed_tmb']=sum(observed_hist)

######################################################

test_model="EXP"
xdata=exp_xdata

exp_results=pd.DataFrame()
y_prob=y_prob_exp

pred_purity_wf=pred_purity
pred_C_wf=pred_C
#takes the WF values as initial values to save some runtime finding the approximate values. 
for i in range(0,4):
    pur_pred_exp,S_pred,C_pred_exp,scores_pred=run_fit_smc_with_WF_initial(test_model,cov,min_reads,observed_hist,collected_data_size,xdata,y_prob,pred_purity_wf,pred_C_wf)
    exp_ensemble_results = pd.DataFrame({   'purity_pred': pur_pred_exp,  'S_pred': S_pred,  'C_pred': C_pred_exp,   'scores_final': scores_pred } )
    exp_final=exp_ensemble_results#[wf_ensemble_results ['mcmc_steps']==max_steps]
    exp_final['chain']=i
    exp_results=pd.concat([exp_results,exp_final])


exp_results['discr_cov_mean']=discr_cov_mean
#exp_results.to_csv(folder_path+"sim_"+str(sim_index)+"_"+test_model+"_pred_first_fit.csv")
df=exp_results

df=df[['purity_pred', 'S_pred', 'C_pred', 'scores_final']]
df['purity_pred_scaled']=df['purity_pred']
df['S_pred_scaled']=df['S_pred']
df['C_pred_scaled']=df['C_pred']
df['scores_final_scaled']=df['scores_final']

df_copy=df
# Define colors for each cluster
colors_2 = ['red', 'blue']
colors_3 = ['red', 'blue', 'green']
colors_4 = ['red', 'blue', 'green', 'purple']

features = ['purity_pred_scaled', 'S_pred_scaled', 'C_pred_scaled', 'scores_final_scaled']
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
# Perform KMeans clustering with 2 clusters
kmeans_2 = KMeans(n_clusters=2, random_state=42,n_init=10).fit(df[features])
df['cluster2'] = kmeans_2.labels_
# Perform KMeans clustering with 3 clusters
kmeans_3 = KMeans(n_clusters=3, random_state=42,n_init=10).fit(df[features])
df['cluster3'] = kmeans_3.labels_
# Perform KMeans clustering with 4 clusters
kmeans_4 = KMeans(n_clusters=4, random_state=42,n_init=10).fit(df[features])
df['cluster4'] = kmeans_4.labels_

lowest_cluster_means=[]
lowest_cluster_nr=[]
highest_cluster_nr=[]
for i in range(2,4):
    group_means = df.groupby('cluster'+str(i))['scores_final'].mean()
    lowest_cluster_nr.append(group_means.idxmin())
    lowest_cluster_means.append(group_means.min())
    highest_cluster_nr.append(group_means.idxmax()) #what is the cluster with highest average score

max_k=np.argmax(lowest_cluster_means)
K_max=max_k+2
cluster_max_index=highest_cluster_nr[max_k]
df_copy=df_copy[df_copy["cluster"+str(K_max)]==cluster_max_index]
#plt.scatter(df.S_pred, df.C_pred,label="outliers")
min_k= np.argmin(lowest_cluster_means)
K=min_k+2
final_scores=[]
Ks=df["cluster"+str(K)].unique()
for cluster_index in Ks:
    cdf=df[df["cluster"+str(K)]==cluster_index]
    #change name from true purity,S,C to predicted. otherwise its confusing
    pred_purity=cdf.purity_pred.mean()
    pred_S=int(cdf.S_pred.mean())
    pred_C=int(cdf.C_pred.mean())
#do some repetats    
    nr_of_repeats=5
    current_score=compute_distance(observed_hist,nr_of_repeats,pred_purity,pred_S,pred_C,cov,binss,min_reads,xdata,y_prob)
    #print(current_score)
    final_scores.append(current_score)
lowest_score_ind=np.argmin(final_scores)
#final_models_score.append(final_scores[lowest_score_ind])
cluster_index=Ks[lowest_score_ind]
cdf=df[df["cluster"+str(K)]==cluster_index]
#print("best cluster result, uncorrected:")       
pred_purity=cdf.purity_pred.mean()
pred_S=int(cdf.S_pred.mean())
pred_C=int(cdf.C_pred.mean())
df=cdf
cluster_size=len(df)
#remove x samples with z score correction on the final score
mean_score = df['scores_final'].mean()
std_score = df['scores_final'].std()
df['z_score'] = (df['scores_final'] - mean_score) / std_score
extra_correction_df = df[(df['z_score'] >= -1.67) & (df['z_score'] <= 1.67)]
new_cluster_size=len(extra_correction_df)
#print("correction would be from ",cluster_size," to ",new_cluster_size)
if new_cluster_size>=3:
    df=extra_correction_df
    cluster_size=new_cluster_size
    
pred_purity=df.purity_pred.mean()
pred_S=int(df.S_pred.mean())
pred_C=int(df.C_pred.mean())
score_exp0=df.scores_final.mean()

##############################################
#pred values
final_results['pred_purity_exp']=pred_purity
final_results['pred_C_exp']=pred_C
final_results['pred_S_exp0']=pred_S
final_results['score_exp0']=score_exp0
final_results['observed_tmb']=sum(observed_hist)
final_results.to_csv(out_dir+donor_id+".csv")