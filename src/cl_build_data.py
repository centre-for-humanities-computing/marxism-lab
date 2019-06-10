#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
1: Build data set for binary classification by crawling directory
2: Clean data set
3: Export data set
"""

# core python
import io, os

# string management
import re
from unidecode import unidecode

# data management
from pandas import DataFrame

# set envrionment
root = "/home/kln/Documents/edu/ling_evidence"
os.chdir(root)

## COMPONENTS
### WALK DIRECTORY
datapath = "DATA/KJV"
for (root, dirnames, filenames) in os.walk(datapath):
    print root
    print dirnames
    print filenames
    print '-----'
#### RELATIVE PATH TO SINGLE FILE
fname = os.path.join(root, filenames[0])
print fname
### OPEN FILE w. encoding included
with io.open(fname, 'r', encoding = 'utf-8') as fobj:
    text = fobj.read()
print text
#### split on PARAGRAPHS escape characters
paragraphs  = text.split('\n\n')
#### NORMALIZE
paragraph = paragraphs[10]
print re.sub(r'\W+',' ',paragraph)
char_only = re.sub(r'[^a-zA-Z]',' ', paragraph)
print char_only
print re.sub(r'  +',' ',char_only)

## FUNCTION 1, WALK DIRECTORY
def read_dir(path, SPLITCHAR = '\n\n', NORM = False):
    """ get paragraphs from documents in subdirectories of root data directory on path
    - normalization optional (remove anythong but alphabetic characters and unicode to ascii)
    """
    paragraphs_ls, filenames_ls = [], []
    for (root, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(root,filename)
            with io.open(filepath, 'r', encoding = 'utf-8') as f:# read text
                text = f.read()
                paragraphs = text.split(SPLITCHAR)# parse paragraphs
                del paragraphs[0]# remove title
                i = 0
                for paragraph in paragraphs:# clean markup
                    paragraph = paragraph.rstrip()
                    if paragraph:
                        if NORM:
                            paragraph = re.sub(r'\W+',' ', paragraph)
                            paragraph = re.sub(r'\d','',paragraph)
                            paragraph = re.sub(r'  +',' ', paragraph)
                            paragraph = unidecode(paragraph.lower())
                        paragraphs_ls.append(paragraph)# append paragraph to list of paragraphs from filepath
                        filenames_ls.append(filename+'_'+ str(i))# filename with running index
                        i += 1
    return filenames_ls, paragraphs_ls

### test
fnames, texts = read_dir("DATA/KJV/", NORM = True)
len(fnames)
len(texts)
texts[10]

# 9461
# 856

## FUNCTION 2, EXPORT TO DATAFRAME/CSV
def make_df(path, classification):
    """ export directory walk to dataframe with CLASS INFORMATION filename as index
    """
    filenames, paragraphs = read_dir(path, NORM = True)
    rows = []
    idx = []
    i = 0
    for paragraph in paragraphs:
        rows.append({'text': paragraph, 'class': classification})
        idx.append(filenames[i])
        i += 1
    df = DataFrame(rows, index = idx)
    return df

# EXECUTE + EXPORT

## CLASS LABELS
NT = 'new_testament'
OT = 'old_testament'
### map CLASS to PATH
SRCS = [("DATA/KJV/OT", OT),("DATA/KJV/NT", NT)]

## Build dataframe
DATA = DataFrame({'text': [], 'class': []})
for path, classification in SRCS:
    DATA = DATA.append(make_df(path, classification))

### inspect
DATA.shape
DATA.head()
DATA.tail()
print DATA.text.iloc[0]

## export
DATA.to_csv("DATA/CLASS_DATA.csv")
