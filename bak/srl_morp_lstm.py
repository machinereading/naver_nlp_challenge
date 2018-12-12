
# coding: utf-8

# In[1]:


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


# # Option

# In[8]:


model_dir = './result/model-morp-lstm'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = model_dir+'/model.pt'

dev_sent = 500


# In[4]:


from datetime import datetime
start_time = datetime.now()
today = start_time.strftime('%Y-%m-%d')


# In[5]:


# load data
data = read_data.load_trn_data()
trn_conll = read_data.load_trn_nlp()


# In[6]:


# input data
# [
#     ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'], 
#     ['인사동에', '들어서면', '다종다양의', '창호지,', '도자기', '등', '고미술품들이', '진열장에', '즐비하게', '널려져', '있는', '것을', '볼', '수', '있다.'], 
#     ['ARGM-LOC', '-', '-', '-', '-', '-', 'ARG1', 'ARG1', '-', '-', '-', 'ARG1', '-', '-', '-']
# ]

def get_input_data(data):
    result = []
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
        result.append(sent_list)
    return result
        
input_data = get_input_data(data)


# In[9]:


div = len(input_data) - dev_sent

dev = input_data[div:]
trn = input_data[:div]
gold_file = './morp_lstm_dev.data'
print('')
print('### dev data:', len(dev), 'sents')

with open(gold_file,'w') as f:
    dev_list = []
    for i in dev:
        dev_list += i[2]
        
    json.dump(dev_list, f)


# In[53]:


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


# # Load Word2Vec

# In[25]:


from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
print('### loading word2vec model...')
wv_model = KeyedVectors.load_word2vec_format("./wordembedding/100_dim_3_window_5mincount_word2vec.model")
print('... is done')


# In[77]:


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


# # Configuration

# In[89]:


configuration = {'token_dim': 60,
                 'hidden_dim': 64,
                 'feat_dim': 1,
                 'dp_dim': 4,
                 'arg_dim': 4,
                 'lu_pos_dim': 5,
                 'dp_label_dim': 10,
                 'lstm_input_dim': 100,
                 'lstm_dim': 64,
                 'lstm_depth': 2,
                 'hidden_dim': 64,
                 'position_feature_dim': 5,
                 'num_epochs': 13,
                 'learning_rate': 0.001,
                 'dropout_rate': 0.01,
                 'pretrained_embedding_dim': 300,
                 'model_dir': model_dir,
                 'model_path': model_path,
                 }
print('\n### CONFIGURATION ###\n')
pprint.pprint(configuration)
print('')

DPDIM = configuration['dp_dim']
ARGDIM = configuration['arg_dim']
LSTMINPDIM = configuration['lstm_input_dim']
FEATDIM = configuration['feat_dim']
HIDDENDIM = configuration['hidden_dim']
LSTMDEPTH = configuration['lstm_depth']
DROPOUT_RATE = configuration['dropout_rate']
learning_rate = configuration['learning_rate']
NUM_EPOCHS = configuration['num_epochs']

print('\n### YOUR MODEL WILL BE SAVED TO', model_path, '###\n')

with open(model_dir+'/config.json', 'w') as f:
    json.dump(configuration, f, ensure_ascii=False, indent=4)


# In[14]:


def get_pred_idxs(conll):
    result = []
    preds = [0 for i in range(len(conll))]
    for i in range(len(conll)):
        tok = conll[i]
        if tok[10].startswith('V'):
            preds = [0 for item in range(len(conll))]
            preds[i] = 1
            result.append(preds)
            
    return result


# In[15]:


def get_arg_idxs(pred_idx, conll):
    arg_idxs = [0 for i in range(len(conll))]
    for i in range(len(conll)):
        tok = conll[i]
        if int(tok[8]) == pred_idx:
            arg_pos = tok[-1]
            if arg_pos[:2] == 'NP':
                arg_idxs[i] = 1
                
    return arg_idxs


# In[16]:


def get_feature(pred_idxs, conll):
    result = []
    for i in pred_idxs:
#         print(i)
        features = []
        for j in range(len(i)):
            pred_idx = i[j]
#             if pred_idx == 1:
#                 arg_idxs = get_arg_idxs(j, conll)
        for j in range(len(i)):
            feature = []                
            feature.append(i[j])
#             feature.append(arg_idxs[j])
            features.append(feature)                
        result.append(features)
        
    return result


# In[17]:


def prepare_sequence(seq, to_ix):
    vocab = list(to_ix.keys())
    idxs = []
    for w in seq:
        if w in vocab:
            idxs.append(to_ix[w])
        else:
            idxs.append(0)  

    return torch.tensor(idxs).cuda()


# In[18]:


def get_dps(conll):
    dps = []
    for tok in conll:
        dp = tok[10]
        dps.append(dp)
    return dps


# In[81]:


def get_sentence_vec(tokens, conll):
    result = []
    for i in range(len(tokens)):
        token = tokens[i]
        morps = conll[i][2].split('+')
#         morp_ix = prepare_sequence(morps, morp_to_ix)
        result.append(morps)
    return result


# # dev eval

# In[142]:


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

def eval_dev(my_model):
    
    with torch.no_grad():
        dev_idx = len(input_data)-len(dev)
        result = []
        n = 0    

        pred_file = model_dir+'/dev.result'
        pred_result = open(pred_file,'w')    

        for s_idx in range(len(dev)):      
            tokens, args = dev[s_idx][1], dev[s_idx][2]
            args_in = prepare_sequence(args, arg_to_ix)  

            conll = read_data.get_nlp_for_trn(s_idx+dev_idx)
            dps = get_dps(conll)
            dp_in = prepare_sequence(dps, dp_to_ix)

            pred_idxs = get_pred_idxs(conll)
            features = get_feature(pred_idxs, conll)
            features = torch.tensor(features).type(torch.cuda.FloatTensor)

            pred = ['-' for i in range(len(args))]

            for i in range(len(pred_idxs)):
                feat_vectors = features[i]            
                input_sent = get_sentence_vec(tokens, conll)       

                pred_seq = pred_idxs[i]
                for j in range(len(pred_seq)):
                    p = pred_seq[j]
                    if p == 1:
                        pred_idx = j
                arg_idxs = get_arg_idxs(pred_idx, conll)
                mask = torch.tensor(arg_idxs).cuda()
                mask = mask.float()

                tag_scores = srl_model(input_sent, dp_in, feat_vectors, mask)
                labels, score = get_labels_by_tensor(tag_scores)

                for idx in range(len(labels)):
                    label = labels[idx]
                    if label == '-':
                        pass
                    else:
                        pred[idx] = label

            pred_result.write(str(pred)+'\n')
            
            annotation = []
            annotation.append(tokens)
            annotation.append(gold_anno[s_idx])
            annotation.append(pred)

            result.append(annotation)
            
    with open(model_dir+'/dev-result.tosee','w') as f:
        for i in result:
            for j in i:
                f.write(str(j)+'\n')
            f.write('\n')
 
    pred_result.close()
    with open(gold_file,'r') as f:
        gold = json.load(f)
    pred = eval_srl.read_prediction(pred_file)
    f1 = eval_srl.evaluate_from_list(pred, gold)
      
    return f1


# # Model

# In[140]:


class LSTMTagger(nn.Module):
    
    def __init__(self, tagset_size):
        super(LSTMTagger, self).__init__()
        
        self.dp_embeddings = nn.Embedding(DP_VOCAB_SIZE, DPDIM)
        
        #LSTM layer        
        self.lstm_morp = nn.LSTM(LSTMINPDIM, HIDDENDIM//2, bidirectional=True, num_layers=LSTMDEPTH, dropout=DROPOUT_RATE)
        self.hidden_lstm_morp = self.init_hidden_lstm_morp()
        
        self.lstm_tok = nn.LSTM(HIDDENDIM+DPDIM+FEATDIM, HIDDENDIM//2, bidirectional=True, num_layers=LSTMDEPTH, dropout=DROPOUT_RATE)
        self.hidden_lstm_tok = self.init_hidden_lstm_tok()
        
        # Linear
        self.hidden2tag = nn.Linear(HIDDENDIM, tagset_size)
              

    def init_hidden_lstm_morp(self):
        return (torch.zeros(4, 1, HIDDENDIM//2).cuda(),
            torch.zeros(4, 1, HIDDENDIM//2).cuda())
    
    def init_hidden_lstm_tok(self):
        return (torch.zeros(4, 1, HIDDENDIM//2).cuda(),
            torch.zeros(4, 1, HIDDENDIM//2).cuda())
    
    def forward(self, input_sent, dp_in, feat_vectors, mask):
        
        dp_embs = self.dp_embeddings(dp_in)
        
#         LSTM layer 1 (subunit to token)
        tok_vectors = []
        for morps in input_sent:
            wes = []
            for morp in morps:
                we = get_word2vec(morp)
                wes.append(we)

            input_vec = torch.stack(wes)
            input_embs_1 = input_vec.view(len(input_vec), 1, -1)
            
            lstm_out_tok, self.hidden_lstm_morp = self.lstm_morp(
                input_embs_1, self.hidden_lstm_morp)

            tok_vectors.append(lstm_out_tok[-1])
        tok_vec = torch.stack(tok_vectors)
        tok_vec = tok_vec.view(len(tok_vec), -1)    
        
#         LSTM layer 2 (token to token)
        input_embs = torch.cat( (tok_vec, dp_embs, feat_vectors), 1)
        input_embs_2 = input_embs.view(len(input_embs), 1, -1)

        lstm_out_tok, self.hidden_lstm_tok = self.lstm_tok(
            input_embs_2, self.hidden_lstm_tok)
        
        # Linear
        tag_space = self.hidden2tag(lstm_out_tok.view(len(input_embs_2),-1))  

        for t_idx in range(len(tag_space)):
            t = tag_space[t_idx]
            m = mask[t_idx]
            if m > 0:
                pass
            else:
                t[-1] = 1
        
        tag_space = F.relu(tag_space)        
        softmax = nn.Softmax(dim=1)
        tag_scores = softmax(tag_space)
        return tag_scores       


# In[143]:


srl_model = LSTMTagger(ARG_VOCAB_SIZE)
srl_model.cuda()
loss_function = nn.CrossEntropyLoss()
# loss_function = nn.NLLLoss()
optimizer = optim.Adam(srl_model.parameters(), lr=learning_rate)
# optimizer = optim.SGD(srl_model.parameters(), lr=learning_rate)

epoch_dev_file = open(model_dir+'/epoch.dev', 'w')
total_step = len(trn)

print('training...')

for epoch in range(NUM_EPOCHS):
    n_of_sent = 0
    total_sent = len(trn)
    for s_idx in range(len(trn)):
        
        tokens, args = trn[s_idx][1], trn[s_idx][2]

        args_in = prepare_sequence(args, arg_to_ix)  

        conll = read_data.get_nlp_for_trn(s_idx)
        dps = get_dps(conll)
        dp_in = prepare_sequence(dps, dp_to_ix)

        pred_idxs = get_pred_idxs(conll)
        features = get_feature(pred_idxs, conll)
        features = torch.tensor(features).type(torch.cuda.FloatTensor)  

        for i in range(len(pred_idxs)):
            feat_vectors = features[i]            
            input_sent = get_sentence_vec(tokens, conll)       
            
            pred_seq = pred_idxs[i]
            for j in range(len(pred_seq)):
                p = pred_seq[j]
                if p == 1:
                    pred_idx = j
            arg_idxs = get_arg_idxs(pred_idx, conll)
            mask = torch.tensor(arg_idxs).cuda()
            mask = mask.float()
            
            srl_model.zero_grad()
            srl_model.hidden_lstm_morp = srl_model.init_hidden_lstm_morp()            
            srl_model.hidden_lstm_tok = srl_model.init_hidden_lstm_tok()

            tag_scores = srl_model(input_sent, dp_in, feat_vectors, mask)
            loss = loss_function(tag_scores, args_in)
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(srl_model.parameters(), 0.25)
            optimizer.step()
            
        n_of_sent +=1
        
        if n_of_sent % 100 == 0:
            print('Epoch [{}/{}], Sent [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, NUM_EPOCHS, n_of_sent, total_sent, loss.item()))
        
#         break
        
    # EPOCH 마다 dev에 대한 성능 평가

        
    f1 = eval_dev(srl_model)
    print('Epoch [{}/{}], F1: {:4f}' 
                   .format(epoch+1, NUM_EPOCHS, f1))
    
    line = 'epoch '+str(epoch+1)+': '+str(f1)+'\n'
    epoch_dev_file.write(line)
    
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    print('')
    
    torch.save(srl_model, model_path)
    
#     break
       
torch.save(srl_model, model_path)
print('')
print('### YOUR MODEL IS SAVED TO', model_path, '###')


# In[ ]:


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

