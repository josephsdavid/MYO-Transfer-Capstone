# General Stuff
import numpy as np
import pandas as pd
from scipy import stats

# Plot Stuff
import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
plt.style.use('bmh')
import seaborn as sns

## sk-learn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, average_precision_score, accuracy_score
from sklearn.metrics import roc_auc_score, classification_report, matthews_corrcoef, precision_recall_fscore_support


def identify_bad_spots(truth, pred, n_std = 0):
    cm = confusion_matrix(truth, pred, normalize='true')
    vals = cm.diagonal()
    vals = vals[~np.isnan(vals)]
    uniq = np.unique(truth)
    c_dict = dict(zip(vals, uniq))
    return {v: k for k, v in c_dict.items() if k <= vals.mean() - n_std* vals.std()}
def outperformers(truth, pred, n_std = 0):
    cm = confusion_matrix(truth, pred, normalize='true')
    vals = cm.diagonal()
    vals = vals[~np.isnan(vals)]
    uniq = np.unique(truth)
    c_dict = dict(zip(vals, uniq))
    return {v: k for k, v in c_dict.items() if k >= vals.mean() + n_std* vals.std()}

def baselineMetrics(t,b):
    _acc = accuracy_score(t,b)
    _accBal = balanced_accuracy_score(t,b)
    _matt = matthews_corrcoef(t,b)
    _prfs = precision_recall_fscore_support(t,b,average='weighted')
    line = {
        'Acc': np.round(_acc,3),
        'Balanced Acc': np.round(_accBal,3),
        'MCC': np.round(_matt,4),
        'Precision': np.round(_prfs[0],3),
        'Recall': np.round(_prfs[1],3),
        'f1-Score': np.round(_prfs[2],3)
    }
    return line

y = np.load("data/y.npy")
classDist = freqs = np.array(np.unique(y, return_counts=True)).T
options, freqs = classDist[:,0],classDist[:,1].astype('float')/y.shape[0]

y_randomW = np.random.choice(options,y.shape,p=freqs)
y_random = np.random.choice(options,y.shape)
y_preds = np.load("data/preds_raw.npy")
y_zeros = np.zeros_like(y)
y_ones = np.ones_like(y)
baselines = {
    'Weighted Random': y_randomW,
    'Unweighted Random': y_random,
    'All Zeros': y_zeros,
    'All Ones': y_ones
}

full_mask = np.where(y==y)
finger_mask = np.where((y > 0) & (y < 13))
wrist_mask = np.where((y >= 13) & (y < 30))
functional_mask = np.where(y >= 30)
not_resting = y[y!=0]
not_resting_preds = y_preds[y!=0]
finger, finger_preds = y[finger_mask], y_preds[finger_mask]
wrist, wrist_preds = y[wrist_mask], y_preds[wrist_mask]
functional, func_preds = y[functional_mask], y_preds[functional_mask]

balanced_accuracy_score(not_resting, not_resting_preds)
# 70%
balanced_accuracy_score(finger, finger_preds)
# 73%
balanced_accuracy_score(wrist, wrist_preds)
# 70%
balanced_accuracy_score(functional, func_preds)
# 68%
balanced_accuracy_score(y[np.where(y == 0)], y_preds[np.where(y == 0)])
# 97%

full_report = pd.DataFrame(classification_report(y, y_preds, output_dict=True, digits=2)).T
full_report[['recall','f1-score','precision']]= np.round(full_report[['recall','f1-score','precision']],3)
full_report[['support']] = full_report[['support']].astype(np.int)
report_tex = full_report.to_latex()

finger_dict = identify_bad_spots(finger, finger_preds)
wrist_dict = identify_bad_spots(wrist, wrist_preds)
fun_dict = identify_bad_spots(functional, func_preds)
len(fun_dict) + len(finger_dict) + len(wrist_dict)
[np.array(list(d.values())).mean() for d in [fun_dict, wrist_dict, finger_dict]]
# [0.7310328726331228, 0.7837323480018306, 0.7706192134013865]
total_dict = {**fun_dict, **finger_dict, **wrist_dict}
ttl = list(total_dict.values())
ttl.append(.97)
np.array(ttl).mean()



counts = sns.countplot(y, palette="pastel")
counts.set(yscale="log")
plt.title("Number of Occurences of Each Gesture (log Scale)")
for item in counts.get_xticklabels():
    item.set_rotation(45)
plt.tight_layout()
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(12, 5.5)
plt.savefig("outputs/count_log.png", dpi=500, size=(12, 5))

lines=[]
for k,b in baselines.items():
    line = [v for _,v in baselineMetrics(y,b).items()]
    lines.append(line)
    # pprint(line)
np.vstack(lines)
baseline_df = pd.DataFrame(lines,index=baselines.keys(), columns=['Acc', 'Balanced Acc', 'MCC', 'Precision', 'Recall', 'f1-Score'])
print(baseline_df.T.to_latex())

lines=[]
for b in [full_mask ,finger_mask, wrist_mask, functional_mask]:
    line = [v for _,v in baselineMetrics(y[b],y_preds[b]).items()]
    lines.append(line)
    # pprint(line)
np.vstack(lines)
baseline_df = pd.DataFrame(lines,index=['Full', 'Finger Gest', 'Wrist Gest', 'Functional'], columns=['Acc', 'Balanced Acc', 'MCC', 'Precision', 'Recall', 'f1-Score'])
print(baseline_df.T.to_latex())

