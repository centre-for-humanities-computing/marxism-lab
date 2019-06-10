#!/usr/bin/env python2
# -*- coding: utf-8 -*-pre
"""
"""
import re

s = "Dante passes through the gate of Hell, which bears an inscription ending with the famous phrase 'Abandon all hope, ye who enter here.'"
# case-folding, removal of punctuation, and tokenization
print s
regex = re.compile(r'\W+')
s1 = regex.split(s)
tokens = [unigram for unigram in regex.split(s) if unigram]
print tokens
# stopword filetering
with open('/home/kln/Documents/edu/ling_evidence/res/stopword_us.txt', 'r') as f:
    stopword = f.read().split()

unigramNostop  = [token for token in tokens if token not in stopword]
print unigramNostop

# lemmatize with multiple parts of speech
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer

def treebank2wordnet(treebank_tag):
    """
    map treebank pos tags to wordnets four categories:
    - n: noun (default), v: verb, a: adjective, and r: adverbs
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN# noun is baseline

def pos_sensitive_lemmatizer(tokens):
    """
    lemmatizer with treebank pos tags
    """
    tokens_tag = pos_tag(tokens, tagset = 'universal', lang = 'eng')
    lemmatizer = WordNetLemmatizer()
    output = []
    for i in range(len(tokens_tag)):
        output.append(lemmatizer.lemmatize(tokens_tag[i][0],
        treebank2wordnet(tokens_tag[i][1])))
    return output

unigramPOS = [token.lower() for token in pos_sensitive_lemmatizer(unigramNostop)]

print unigramPOS
