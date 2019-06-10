#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
feature selection and optimization
"""


# core python
import io, os

# numerical/scientific computing
import numpy as np

# data management
import pandas as pd

# machine learning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
# plotting
import matplotlib.pyplot as plt

# set envrionment
root = "/home/kln/Documents/edu/ling_evidence"
os.chdir(root)

# MANAGING CLASSIFICATION DATA (from cl_build_data.py)
DATA = pd.read_csv("DATA/CLASS_DATA_NONBIAS.csv", index_col = 0)

# VECTOR SPACE
vectorizer = vectorizer = CountVectorizer(ngram_range = (1,2))

X = vectorizer.fit_transform(DATA.text)# fit vector space
y = DATA['class']

## properties
n_sample, n_feat = X.shape

# REMOVE FEATURES WITH LOW VARIANCE
from sklearn.feature_selection import VarianceThreshold


# UNIVARIATE FEATURE SELECTION
## Univariate feature selection works by selecting the best features based on univariate statistical tests

### k best based on chi^2 test
X2 = SelectKBest(chi2, k=100).fit_transform(X, y)
X2.shape

### mutual information
X3 = SelectKBest(mutual_info_classif, k=100).fit_transform(X, y)
X3.shape
