
# coding: utf-8

# In[17]:


import json
import read_data
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from optparse import OptionParser
import torch.autograd as autograd
from copy import deepcopy
import os
import sys
import pprint
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
torch.manual_seed(1)

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import eval_srl
from src import etri


# # options

# In[2]:


MASKING = False
ARGINFO = True
DPINFO = True

model_dir = './result/model-morp-sum'
model_path = model_dir+'/model.pt'
config = model_dir+'/config.json'


# In[20]:





# In[3]:


with open(config,'r') as f:
    configuration = json.load(f)

TOKDIM = configuration['token_dim']
if DPINFO:
    DPDIM = configuration['dp_dim']
else:
    DPDIM = 0
ARGDIM = configuration['arg_dim']
LSTMINPDIM = configuration['lstm_input_dim']
if ARGINFO:
    FEATDIM = configuration['feat_dim']
else:
    FEATDIM = 1
HIDDENDIM = configuration['hidden_dim']
LSTMDEPTH = configuration['lstm_depth']
DROPOUT_RATE = configuration['dropout_rate']
learning_rate = configuration['learning_rate']
NUM_EPOCHS = configuration['num_epochs']


# # Model

# In[4]:


class LSTMTagger(nn.Module):
    
    def __init__(self, tagset_size):
        super(LSTMTagger, self).__init__()
        
        if DPINFO:
            self.dp_embeddings = nn.Embedding(DP_VOCAB_SIZE, DPDIM)
        
        self.lstm_tok = nn.LSTM(LSTMINPDIM+TOKDIM+DPDIM+FEATDIM, HIDDENDIM//2, bidirectional=True, num_layers=LSTMDEPTH, dropout=DROPOUT_RATE)
        self.hidden_lstm_tok = self.init_hidden_lstm_tok()
        
        # Linear
        self.hidden2tag = nn.Linear(HIDDENDIM, tagset_size)
    
    def init_hidden_lstm_tok(self):
        return (torch.zeros(4, 1, HIDDENDIM//2).cuda(),
            torch.zeros(4, 1, HIDDENDIM//2).cuda())
    
    def forward(self, input_sent, pred_idx, dp_in, feat_vector, mask):
        
        if DPINFO:
            dp_embs = self.dp_embeddings(dp_in)
        
#         LSTM layer 1 (subunit to token)
        tok_vectors = []
    
        pred_vec = torch.zeros(100).cuda()
        for morp in input_sent[pred_idx]:
            pred_vec += get_word2vec(morp)
    
        for morps in input_sent:
            we = torch.zeros(100).cuda()
            for morp in morps:
                we += get_word2vec(morp)
            we = torch.cat( (we, pred_vec) )
            tok_vectors.append(we)

        tok_vec = torch.stack(tok_vectors)
        tok_vec = tok_vec.view(len(tok_vec), -1)
        
#         LSTM layer
        if DPINFO:
            input_embs = torch.cat( (tok_vec, dp_embs, feat_vector), 1)
        else:
            input_embs = torch.cat( (tok_vec, feat_vector), 1)
        input_embs_2 = input_embs.view(len(input_embs), 1, -1)

        lstm_out_tok, self.hidden_lstm_tok = self.lstm_tok(
            input_embs_2, self.hidden_lstm_tok)
        
#         lstm_out_tok = F.relu(lstm_out_tok)
        
        # Linear
        tag_space = self.hidden2tag(lstm_out_tok.view(len(input_embs_2),-1))  
   
        return tag_space


# In[ ]:


my_model = torch.load(model_path)


# In[5]:


data = read_data.load_trn_data()
trn_conll = read_data.load_trn_nlp()

def prepare_idx():
    dp_to_ix, arg_to_ix, morp_to_ix = {},{},{}
    dp_to_ix['null'] = 0
    morp_to_ix['null'] = 0
    
    for sent in trn_conll:
        for token in sent:
            dp = token[11]
            if dp not in dp_to_ix:
                dp_to_ix[dp] = len(dp_to_ix)
                
            morphs = token[2].split('+')
            for morp in morphs:
                if morp not in morp_to_ix:
                    morp_to_ix[morp] = len(morp_to_ix)
    args = ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARGM-CAU', 'ARGM-CND', 'ARGM-DIR', 'ARM-DIS', 'ARGM-INS', 'ARGM-LOC', 'ARCM-MNR', 'ARCM-NEG', 'ARCM-PRD', 'ARCM-PRP', 'ARCM-TMP', 'ARCM-ADV', 'ARCM-EXT', '-']
    for i in args:
        if i not in arg_to_ix:
            arg_to_ix[i] = len(arg_to_ix)
    return dp_to_ix, arg_to_ix, morp_to_ix
dp_to_ix, arg_to_ix, morp_to_ix = prepare_idx()
DP_VOCAB_SIZE = len(dp_to_ix)
ARG_VOCAB_SIZE = len(arg_to_ix)
MORP_VOCAB_SIZE = len(morp_to_ix)
print('DP_VOCAB_SIZE:',DP_VOCAB_SIZE)
print('ARG_VOCAB_SIZE:',ARG_VOCAB_SIZE)
print('MORP_VOCAB_SIZE:', MORP_VOCAB_SIZE)


# # Modules

# In[6]:


from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
print('### loading word2vec model...')
wv_model = KeyedVectors.load_word2vec_format("./wordembedding/100_dim_3_window_5mincount_word2vec.model")
print('... is done')


# In[23]:


def get_word2vec(morp):
    
    vec = torch.rand(100)
    emb = vec.cuda()
    
    try:
        vec = wv_model[morp]
        emb = torch.from_numpy(vec).cuda()
    except KeyboardInterrupt:
        raise
    except:
        pass                

    return emb

def get_pred_idxs(conll):
    result = []
    preds = [0 for i in range(len(conll))]
    for i in range(len(conll)):
        tok = conll[i]
        if tok[10] == 'VP' or tok[10] == 'VP_MOD' :
            preds = [0 for item in range(len(conll))]
            preds[i] = 1
            result.append(preds)
            
    return result

def get_arg_idxs(pred_idx, conll):
    arg_idxs = [0 for i in range(len(conll))]
    for i in range(len(conll)):
        tok = conll[i]
        if int(tok[8]) == pred_idx:
#             arg_idxs[i] = 1
            arg_pos = tok[-1]
            if arg_pos[:2] == 'NP':
                arg_idxs[i] = 1
                
    return arg_idxs

def get_feature(pred_idx, conll):
    feat_vec = []
    arg_idxs = get_arg_idxs(pred_idx, conll)
    for i in range(len(conll)):
#         tok = conll[i]
        feature = []
        if pred_idx == i:
            feature.append(1)
        else:
            feature.append(0)
        if FEATDIM >= 2:
            feature.append(arg_idxs[i])
        if FEATDIM == 3:
            if i >= pred_idx:
                position = 0
            else:
                position = 1
            feature.append(position)
        feat_vec.append(feature)                
        
    return feat_vec

def prepare_sequence(seq, to_ix):
    vocab = list(to_ix.keys())
    idxs = []
    for w in seq:
        if w in vocab:
            idxs.append(to_ix[w])
        else:
            idxs.append(0)  

    return torch.tensor(idxs).cuda()

def get_dps(conll):
    dps = []
    for tok in conll:
        dp = tok[10]
        dps.append(dp)
    return dps

def get_sentence_vec(tokens, conll):
    result = []
    for i in range(len(tokens)):
        token = tokens[i]
        morps = conll[i][2].split('+')
#         morp_ix = prepare_sequence(morps, morp_to_ix)
        result.append(morps)
    return result

def get_labels_by_tensor(t):
    value, indices = t.max(1)
    score = pow(1, value)
    labels = []
    for i in indices:
        for label, idx in arg_to_ix.items():
            if idx == i:
                pred = label
                labels.append(pred)
                break
    return labels, score            
        
    pred = None
    for label, idx in arg_to_ix.items():
        if idx == indices:
            pred = label
            break
    return pred, score


# # Infer

# In[25]:


def infer(data, my_model):
    
    result = []
    for sent in data:
        text = ' '.join(sent[1])
        nlp = etri.getETRI_rest(text)
        conll = etri.getETRI_CoNLL2009(nlp)
        
        with torch.no_grad():
            
            tokens, args = sent[1], sent[2]            
            args_in_all = prepare_sequence(args, arg_to_ix)
            
            dps = get_dps(conll)
            dp_in = prepare_sequence(dps, dp_to_ix)

            pred_idxs = get_pred_idxs(conll)

            pred = ['-' for i in range(len(args))]

            for i in range(len(pred_idxs)):
                pred_seq = pred_idxs[i]
                for j in range(len(pred_seq)):
                    p = pred_seq[j]
                    if p == 1:
                        pred_idx = j
                        
                feature = get_feature(pred_idx, conll)
                feat_vector = torch.tensor(feature).type(torch.cuda.FloatTensor)           
                input_sent = get_sentence_vec(tokens, conll)
                arg_idxs = get_arg_idxs(pred_idx, conll)            
                args_in = torch.zeros(len(arg_idxs))

                for idx in range(len(arg_idxs)):
                    a = arg_idxs[idx]
                    if a == 1:
                        args_in[idx] = args_in_all[idx]
                    else:
                        args_in[idx] = 17

                args_in = args_in.type(torch.cuda.LongTensor)
                
                mask = torch.tensor(arg_idxs).cuda()
                mask = mask.float()

                tag_scores = my_model(input_sent, pred_idx, dp_in, feat_vector, mask)
                
#                 print(tag_scores)
                labels, score = get_labels_by_tensor(tag_scores)
                
                for idx in range(len(labels)):
                    if arg_idxs[idx] == 1:
                        label = labels[idx]
                    else:
                        label = '-'
                        
                    if label == '-':
                        pass
                    else:
                        if pred[idx] == '-':
                            pred[idx] = label
                            
            result.append(pred)
    return result


# In[26]:


def test():
    data = read_data.load_trn_data()
    dummy_data = []
    for sent in data:
        sent_list = []
        
        tok_idx = []
        tok_str = []
        tok_arg = []
        for token in sent:
            tok_idx.append(token[0])
            tok_str.append(token[1])
            tok_arg.append(token[2])
            
        sent_list.append(tok_idx)
        sent_list.append(tok_str)
        sent_list.append(tok_arg)
        dummy_data.append(sent_list)
    dummy_data = dummy_data[:5]
    
    print('#input')
    print(dummy_data)
    
    result = infer(dummy_data, my_model)
    print('#output')
    print(result)
    
    return result
        
# result = test()

