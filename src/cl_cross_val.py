"""
single classifier with cross validation
"""
# core python
import io, os
from itertools import cycle

# numerical/scientific computing
import numpy as np
from scipy import interp# Returns the one-dimensional piecewise linear interpolant to a function with given values at discrete data-points.

# data management
import pandas as pd

# machine learning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

# plotting
import matplotlib.pyplot as plt

# set envrionment
root = "/home/kln/Documents/edu/ling_evidence"
os.chdir(root)

# DATA GENERATION
DATA = pd.read_csv("DATA/CLASS_DATA_NONBIAS.csv", index_col = 0)
vectorizer = CountVectorizer(ngram_range = (1,2), stop_words = 'english',
    lowercase = True, max_df = .95, min_df = .01, max_features = 500)

X = vectorizer.fit_transform(DATA.text)# fit vector space
y = DATA['class']

## properties
n_sample, n_feat = X.shape

# CLASSIFICATION
nbclassifier = MultinomialNB(alpha = 1.0, fit_prior = True, class_prior = None)

# Run classifier with cross-validation and plot ROC curves
k = 10
cv = StratifiedKFold(n_splits = k)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    # compute class probabilities
    cPRs = nbclassifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], cPRs[:, 1], pos_label = 'old_testament')
    # interp: one-dimensional piecewise linear interpolant to a function with given values at discrete data-points.
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3)#, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1

# build summary part of the plot
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

# error band
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.5,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('K-fold Cross-Validation (K = {})'.format(k))
plt.legend(loc="lower right")
#plt.show()
plt.savefig('figures/ROC_crossval.png', dpi = 600)
plt.close()
