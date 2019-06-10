#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

"""

# core python
import io, os

# numerical/scientific computing
import random
import numpy as np

# data management
import pandas as pd

# machine learning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# plotting
import matplotlib.pyplot as plt

# set envrionment
root = "/home/kln/Documents/edu/ling_evidence"
os.chdir(root)

# MANAGING CLASSIFICATION DATA (from cl_build_data.py)
DATA = pd.read_csv("DATA/CLASS_DATA.csv", index_col = 0)

## CLASS DIST AND BIAS
def printdist(DF):
    for label in set(DF['class']):
        print "number of " + label + ": {}".format(sum(DF['class'] == label))

printdist(DATA)# BIASED
print "Accuracy for free {}".format(round(9461/float((9461+856)),2))


### UNBIAS DATA
def balance(df, n, classcol = 'class'):
    random.seed(1234)
    res = pd.DataFrame(columns = DATA.columns)
    C = list(set(df[classcol]))
    for c in C:
        idx = df[df[classcol] == c].index.tolist()
        df_c = df.loc[random.sample(idx, n),]# label based indexing
        res = res.append(df_c)
    return res.reindex(np.random.permutation(res.index))#shuffle order for classifirer

DATB = balance(DATA, 800)
printdist(DATB)
DATB.to_csv("DATA/CLASS_DATA_NONBIAS.csv")


# SPLIT DATA SET
ratio = .8
mask = np.random.rand(len(DATB)) <= ratio
TRAIN = DATB[mask]
TEST = DATB[~mask]

## training set
X_train = TRAIN['text'].values
y_train = TRAIN['class'].values
## test set
X_test = TEST['text'].values
y_test = TEST['class'].values


### INTERMEZZO: DOCUMENT REPRESENTATIONS AND UNDERSTANDING VECTORIZERS ###
vectorizer = CountVectorizer()# INSTANTIATE VECTORIZER
print vectorizer

TEXTS = ['This is the first document.',
        'This is the second second document.',
        'And the third one.',
        'Is this the first document? This is right.']

DTM = vectorizer.fit_transform(TEXTS)

print 'DTM: {}'.format(DTM.todense())
print 'vocabulary (index): {}'.format(vectorizer.get_feature_names())

# replace textmining with something more efficient
np.savetxt("DATA/DTM.csv", DTM.todense(), delimiter=",")
lexicon = vectorizer.get_feature_names()
with io.open('DATA/LEXICON.txt','w', encoding = 'utf-8') as f:
    for i in lexicon:
        f.write("%s\n" % i)

##########################################################################

# FEATURE EXTRACTION FOR UNSTRUCTURED DATA
vectorizer = CountVectorizer(ngram_range = (1,2), stop_words = 'english',
    lowercase = True, max_df = .95, min_df = .01, max_features = 500)

FEAT_train = vectorizer.fit_transform(X_train)# fit vector space
FEAT_test =  vectorizer.transform(X_test)# !only transform, ignoring features not occurring in training set!
FEAT_names = vectorizer.get_feature_names()

# TRAIN CLASSIFIER
nb_classifier = MultinomialNB()
nb_classifier.fit(FEAT_train, y_train)

# EVALUATION
pred = nb_classifier.predict(FEAT_test)
confmat = metrics.confusion_matrix(y_test, pred)# horizontal: predicted label; vertical: true label
# obeserved accuracy
print "Accurracy: {}".format(round(metrics.accuracy_score(y_test, pred),2))

# cohen's kappa
print "K: {}".format(metrics.cohen_kappa_score(y_test, pred))
# model summary
print metrics.classification_report(y_test, pred)


## ADVANCED VALIDATION
### obtain class possiblities
y_scores = nb_classifier.fit(FEAT_train, y_train).predict_proba(FEAT_test)

### ROC and AUC for ROC
FPR, TPR, thresholds = metrics.roc_curve(y_test, y_scores[:,1], pos_label = 'old_testament')
AUC = round(metrics.auc(FPR, TPR),2)# alternative Accurracy measure

#### PLOT ROC
plt.title('ROC')
plt.plot(FPR, TPR, c='r', label=('AUC = {}'.format(AUC)))
plt.legend(loc='lower right', prop={'size':8})
plt.plot([0,1],[0,1], color='lightgrey', linestyle='--')
plt.xlim([-0.01,1.0])
plt.ylim([0.0,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('figures/ROC.png', dpi = 300)
plt.close()

### precision-recall curve
precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_scores[:,1], pos_label = 'old_testament')
plt.title('PR Curve')
plt.plot(recall, precision, c='r')
plt.xlim([-0.01,1.0])
plt.ylim([0.0,1.01])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('figures/PRC.png', dpi = 300)
plt.close()


# SAVE CLASSIFIER FOR LATER USE


######
