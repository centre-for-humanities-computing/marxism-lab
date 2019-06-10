#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Pipeline with NB, LogitBoost and SVM
"""
# core python
import io, os

# numerical/scientific computing
import numpy as np

# data management
import pandas as pd

# machine learning
from sklearn.feature_extraction.text import CountVectorizer
## classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
## validation
from sklearn.model_selection import train_test_split
from sklearn import metrics

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1234)

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)
nb_pred = naive_bayes.predict(X_test)
print "Naive Bayes Accurracy: {}".format(round(metrics.accuracy_score(y_test, nb_pred),2))

logit_boost = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, random_state=1234)
logit_boost.fit(X_train, y_train)
lb_pred = logit_boost.predict(X_test.todense())
print "Logit Boost Accurracy: {}".format(round(metrics.accuracy_score(y_test, lb_pred),2))

support_vecm = svm.SVC(probability = True)
support_vecm.fit(X_train, y_train)
svm_pred = support_vecm.predict(X_test)
print "Support Vector Machine Accurracy: {}".format(round(metrics.accuracy_score(y_test, svm_pred),2))

algo = ['Naive Bayes','Logit Boost','SVM']
CLFs = [naive_bayes, logit_boost, support_vecm]

fig, ax = plt.subplots(1,3, figsize=(15, 4), sharey='row')
i = 0
for clf in CLFs:
    y_scores = clf.fit(X_train, y_train).predict_proba(X_test.todense())
    FPR, TPR, thresholds = metrics.roc_curve(y_test, y_scores[:,1], pos_label = 'old_testament')
    AUC = round(metrics.auc(FPR, TPR),2)# alternative Accurracy measure
    ax[i].plot(FPR, TPR, c='b', label=('AUC = {}'.format(AUC)))
    ax[i].set_title(algo[i])
    ax[i].set_ylabel('True Positive Rate')
    ax[i].set_xlabel('False Positive Rate')
    ax[i].plot([0, 1], [0, 1], linestyle='--', lw=2, color='lightgrey', label='Chance', alpha=.8)
    ax[i].legend(loc='lower right', prop={'size':8})
    i += 1
plt.savefig('figures/ROC_multiclass.png', dpi = 300)
plt.close()








#############

















def splitdata(DF, ratio = .8):
    """ split data set in training (tr) and testing (te) data set
    """
    mask = np.random.rand(len(DATB)) <= ratio
    tr = DF[mask]
    te = DF[~mask]
    Xtr = tr['text'].values
    ytr = tr['class'].values
    Xte = te['text'].values
    yte = te[]'class'].values
    return Xtr, ytr, Xte, yte


from sklearn.model_selection import StratifiedKFold
