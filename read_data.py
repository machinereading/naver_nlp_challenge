
# coding: utf-8

# In[89]:


import json
from collections import Counter


# In[90]:


def load_trn_data():
    with open('./data/trn.txt','r') as f:
        lines = f.readlines()
    result = []
    sent = []
    for line in lines:
        line = line.rstrip('\n')
        if line != '':
            token = line.split('\t')
            sent.append(token)
        else:
            result.append(sent)
            sent = []
    return result

print('loading TRAINING data...')
trn = load_trn_data()


# In[99]:


def load_trn_nlp():
    with open('./data/trn_nlp.conll','r') as f:
        lines = f.readlines()
    result = []
    sent = []
    for line in lines:
        line = line.rstrip('\n')
        if line != '':
            token = line.split('\t')
            sent.append(token)
        else:
            result.append(sent)
            sent = []
    return result

def get_nlp_for_trn(idx):
    return trn_nlp[idx]

