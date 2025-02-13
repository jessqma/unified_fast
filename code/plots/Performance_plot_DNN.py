import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data_dir = '../../data'
processed_data_dir = os.path.join(data_dir, 'processed_data_training')
saved_dir = os.path.join(data_dir, 'model_data_saved')
result_dir = '../../results'


def fpr_interp(fp, tp):
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    for ii in range(len(fp)):
        interp_tpr = np.interp(mean_fpr, fp[ii], tp[ii])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
    mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr = np.max(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    return mean_fpr, mean_tpr, std_tpr, tprs_upper, tprs_lower


# In[16]:


def loading(file_type, d_type):

    data= []
    for ii in range(10):
        file = np.load(os.path.join(saved_dir, file_type+"_"+d_type+'%d.npy'%ii))
        data.append(file)

    return(data)



fp_all = loading('fp', 'all')
tp_all = loading('tp', 'all')

fp_CH = loading('fp', 'CH')
tp_CH = loading('tp', 'CH')

fp_GE = loading('fp', 'GE')
tp_GE = loading('tp', 'GE')

fp_GT = loading('fp', 'GT')
tp_GT = loading('tp', 'GT')

fp_GE_GT = loading('fp', 'GE_GT')
tp_GE_GT = loading('tp', 'GE_GT')

fp_CH_GE = loading('fp', 'CH_GE')
tp_CH_GE = loading('tp', 'CH_GE')

fp_CH_GT = loading('fp', 'CH_GT')
tp_CH_GT = loading('tp', 'CH_GT')


auc_all = loading('auc', 'all')
auc_CH = loading('auc', 'CH')
auc_GE = loading('auc', 'GE')
auc_GT = loading('auc', 'GT')
auc_GE_GT = loading('auc', 'GE_GT')
auc_CH_GE = loading('auc', 'CH_GE')
auc_CH_GT = loading('auc', 'CH_GT')


mean_fpr_all, mean_tpr_all, std_tpr_all, tprs_upper_all, tprs_lower_all = fpr_interp(fp_all, tp_all)
mean_fpr_CH, mean_tpr_CH, std_tpr_CH, tprs_upper_CH, tprs_lower_CH = fpr_interp(fp_CH, tp_CH)
mean_fpr_GE, mean_tpr_GE, std_tpr_GE, tprs_upper_GE, tprs_lower_GE = fpr_interp(fp_GE, tp_GE)
mean_fpr_GT, mean_tpr_GT, std_tpr_GT, tprs_upper_GT, tprs_lower_GT = fpr_interp(fp_GT, tp_GT)
mean_fpr_GE_GT, mean_tpr_GE_GT, std_tpr_GE_GT, tprs_upper_GE_GT, tprs_lower_GE_GT = fpr_interp(fp_GE_GT, tp_GE_GT)
mean_fpr_CH_GE, mean_tpr_CH_GE, std_tpr_CH_GE, tprs_upper_CH_GE, tprs_lower_CH_GE = fpr_interp(fp_CH_GE, tp_CH_GE)
mean_fpr_CH_GT, mean_tpr_CH_GT, std_tpr_CH_GT, tprs_upper_CH_GT, tprs_lower_CH_GT = fpr_interp(fp_CH_GT, tp_CH_GT)



sns.set_style('whitegrid')
plt.subplots(1, figsize=(6,6))
plt.title('ROC - DNN model', fontsize=20)

plt.plot([0, 1], ls="--", color='black')
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=16)

plt.plot(mean_fpr_all, mean_tpr_all,lw=2, alpha=.8, label='All (AUC = %0.3f $\pm$%0.3f)' % (np.mean(auc_all), np.std(auc_all)))
plt.fill_between(mean_fpr_all, tprs_lower_all, tprs_upper_all, alpha=.2)

plt.plot(mean_fpr_CH, mean_tpr_CH, lw=2, alpha=.8, label='CH (AUC = %0.3f $\pm$%0.3f)' % (np.mean(auc_CH), np.std(auc_CH)))
plt.fill_between(mean_fpr_CH, tprs_lower_CH, tprs_upper_CH, alpha=.2)

plt.plot(mean_fpr_GE, mean_tpr_GE, lw=2, alpha=.8, label='GE (AUC = %0.3f $\pm$%0.3f)' % (np.mean(auc_GE), np.std(auc_GE)))
plt.fill_between(mean_fpr_GE, tprs_lower_GE, tprs_upper_GE, alpha=.2)

plt.plot(mean_fpr_GT, mean_tpr_GT, lw=2, alpha=.8, label='GT (AUC = %0.3f $\pm$%0.3f)' % (np.mean(auc_GT), np.std(auc_GT)))
plt.fill_between(mean_fpr_GT, tprs_lower_GT, tprs_upper_GT, alpha=.2)

plt.plot(mean_fpr_GE_GT, mean_tpr_GE_GT, lw=2, alpha=.8, label='GE+GT (AUC = %0.3f $\pm$%0.3f)' % (np.mean(auc_GE_GT), np.std(auc_GE_GT)))
plt.fill_between(mean_fpr_GE_GT, tprs_lower_GE_GT, tprs_upper_GE_GT, alpha=.2)

plt.plot(mean_fpr_CH_GE, mean_tpr_CH_GE, lw=2, alpha=.8, label='CH+GE (AUC = %0.3f $\pm$%0.3f)' % (np.mean(auc_CH_GE), np.std(auc_CH_GE)))
plt.fill_between(mean_fpr_CH_GE, tprs_lower_CH_GE, tprs_upper_CH_GE, alpha=.2)

plt.plot(mean_fpr_CH_GT, mean_tpr_CH_GT, lw=2, alpha=.8, label='CH+GT (AUC = %0.3f $\pm$%0.3f)' % (np.mean(auc_CH_GT), np.std(auc_CH_GT)))
plt.fill_between(mean_fpr_CH_GT, tprs_lower_CH_GT, tprs_upper_CH_GT, alpha=.2)

plt.legend(fontsize=12, bbox_to_anchor=(0.5,0.5))

plt.savefig(os.path.join(result_dir, 'ROC_DNN.png'))


x = ['All', 'CH', 'GE', 'GT', 'GE+GT', 'GE+CH', 'GT+CH']

auc = [np.mean(auc_all), np.mean(auc_CH), np.mean(auc_GE), np.mean(auc_GT),
       np.mean(auc_GE_GT), np.mean(auc_CH_GE), np.mean(auc_CH_GT)]
auc_std = [np.std(auc_all), np.std(auc_CH), np.std(auc_GE), np.std(auc_GT),
       np.std(auc_GE_GT), np.std(auc_CH_GE), np.std(auc_CH_GT)]


auc = pd.DataFrame(np.zeros((10*7, 3)), columns=['data_type', 'rate', 'type'])
auc['data_type'].iloc[:10] = 'All'
auc['rate'].iloc[:10] = auc_all
auc['data_type'].iloc[10:20] = 'CH'
auc['rate'].iloc[10:20] = auc_CH
auc['data_type'].iloc[20:30] = 'GE'
auc['rate'].iloc[20:30] = auc_GE
auc['data_type'].iloc[30:40] = 'GT'
auc['rate'].iloc[30:40] = auc_GT
auc['data_type'].iloc[40:50] = 'GE+GT'
auc['rate'].iloc[40:50] = auc_GE_GT
auc['data_type'].iloc[50:60] = 'GE+CH'
auc['rate'].iloc[50:60] = auc_CH_GE
auc['data_type'].iloc[60:70] = 'GT+CH'
auc['rate'].iloc[60:70] = auc_CH_GT


acc_all = loading('acc', 'all')
acc_CH = loading('acc', 'CH')
acc_GE = loading('acc', 'GE')
acc_GT = loading('acc', 'GT')
acc_GE_GT = loading('acc', 'GE_GT')
acc_CH_GE = loading('acc', 'CH_GE')
acc_CH_GT = loading('acc', 'CH_GT')

acc = pd.DataFrame(np.zeros((10*7, 3)), columns=['data_type', 'rate', 'type'])
acc['data_type'].iloc[:10] = 'All'
acc['rate'].iloc[:10] = acc_all
acc['data_type'].iloc[10:20] = 'CH'
acc['rate'].iloc[10:20] = acc_CH
acc['data_type'].iloc[20:30] = 'GE'
acc['rate'].iloc[20:30] = acc_GE
acc['data_type'].iloc[30:40] = 'GT'
acc['rate'].iloc[30:40] = acc_GT
acc['data_type'].iloc[40:50] = 'GE+GT'
acc['rate'].iloc[40:50] = acc_GE_GT
acc['data_type'].iloc[50:60] = 'GE+CH'
acc['rate'].iloc[50:60] = acc_CH_GE
acc['data_type'].iloc[60:70] = 'GT+CH'
acc['rate'].iloc[60:70] = acc_CH_GT


f1_all = loading('f1', 'all')
f1_CH = loading('f1', 'CH')
f1_GE = loading('f1', 'GE')
f1_GT = loading('f1', 'GT')
f1_GE_GT = loading('f1', 'GE_GT')
f1_CH_GE = loading('f1', 'CH_GE')
f1_CH_GT = loading('f1', 'CH_GT')

f1 = pd.DataFrame(np.zeros((10*7, 3)), columns=['data_type', 'rate', 'type'])
f1['data_type'].iloc[:10] = 'All'
f1['rate'].iloc[:10] = f1_all
f1['data_type'].iloc[10:20] = 'CH'
f1['rate'].iloc[10:20] = f1_CH
f1['data_type'].iloc[20:30] = 'GE'
f1['rate'].iloc[20:30] = f1_GE
f1['data_type'].iloc[30:40] = 'GT'
f1['rate'].iloc[30:40] = f1_GT
f1['data_type'].iloc[40:50] = 'GE+GT'
f1['rate'].iloc[40:50] = f1_GE_GT
f1['data_type'].iloc[50:60] = 'GE+CH'
f1['rate'].iloc[50:60] = f1_CH_GE
f1['data_type'].iloc[60:70] = 'GT+CH'
f1['rate'].iloc[60:70] = f1_CH_GT


acc['type'] = 'Accuracy'
auc['type'] = 'AUC'
f1['type'] = 'F1 Score'


con = pd.concat([acc, auc, f1])

col = sns.color_palette('Set2')

g = sns.catplot(
    data=con,kind='bar',
    x="data_type", y="rate", hue="type",
    ci="sd", palette=col, height=5, aspect=2)
g.set_xticklabels(rotation=30, horizontalalignment='center', fontsize=16)
plt.yticks(fontsize=15)
# g.legend.set_title("")
g.set(ylim=(0, 1))
plt.title(' Performance', fontsize=20)
plt.savefig(os.path.join(result_dir,'Performance_DNN.png'))
