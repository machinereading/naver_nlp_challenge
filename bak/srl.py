
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

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import eval_srl


# # option 

# In[37]:


# lambda_score = 0.1
lambda_score = 1
dev_sent = 100

model_dir = './result/model-'+str(lambda_score)+''
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = model_dir+'/model.pt'

if lambda_score == 1:
    penalty = False
else:
    penalty = True


# In[3]:


from datetime import datetime
start_time = datetime.now()
today = start_time.strftime('%Y-%m-%d')


# In[4]:


# load data
data = read_data.load_trn_data()
trn_conll = read_data.load_trn_nlp()


# In[5]:


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


# # gen TRN and DEV data

# In[40]:


div = len(input_data) - dev_sent

dev = input_data[div:]
trn = input_data[:div]
gold_file = './dev.data'
print('dev data:', len(dev), 'sents')

with open(gold_file,'w') as f:
    dev_list = []
    for i in dev:
        dev_list += i[2]
        
    json.dump(dev_list, f)


# In[7]:


def prepare_idx():
    dp_to_ix, arg_to_ix = {},{}
    dp_to_ix['null'] = 0
    for sent in trn_conll:
        for token in sent:
            dp = token[11]
            if dp not in dp_to_ix:
                dp_to_ix[dp] = len(dp_to_ix)    
    args = ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARGM-CAU', 'ARGM-CND', 'ARGM-DIR', 'ARM-DIS', 'ARGM-INS', 'ARGM-LOC', 'ARCM-MNR', 'ARCM-NEG', 'ARCM-PRD', 'ARCM-PRP', 'ARCM-TMP', 'ARCM-ADV', 'ARCM-EXT', '-']
    for i in args:
        if i not in arg_to_ix:
            arg_to_ix[i] = len(arg_to_ix)
    return dp_to_ix, arg_to_ix
dp_to_ix, arg_to_ix = prepare_idx()
DP_VOCAB_SIZE = len(dp_to_ix)
ARG_VOCAB_SIZE = len(arg_to_ix)
print('DP_VOCAB_SIZE:',DP_VOCAB_SIZE)
print('ARG_VOCAB_SIZE:',ARG_VOCAB_SIZE)


# # Configuration

# In[8]:


configuration = {'token_dim': 60,
                 'hidden_dim': 64,
                 'feat_dim': 2,
                 'dp_dim': 4,
                 'arg_dim': 4,
                 'lu_pos_dim': 5,
                 'dp_label_dim': 10,
                 'lstm_input_dim': 768,
                 'lstm_dim': 64,
                 'lstm_depth': 2,
                 'hidden_dim': 64,
                 'position_feature_dim': 5,
                 'num_epochs': 25,
                 'learning_rate': 0.001,
                 'dropout_rate': 0.01,
                 'pretrained_embedding_dim': 300,
                 'model_dir': model_dir,
                 'model_path': model_path,
                 'lambda_score': lambda_score
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


# In[9]:


# load BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual')
bert_model = BertModel.from_pretrained('bert-base-multilingual')


# In[10]:


def tokenization(tokens):
    bert_tokens = []
    orig_to_token_map = []
    bert_tokens.append("[CLS]")
    for i in range(len(tokens)):
        origin_token = tokens[i]
        orig_to_token_map.append(len(bert_tokens))
        bert_tokens.extend(tokenizer.tokenize(origin_token))
    
    return bert_tokens, orig_to_token_map  


# # functions

# In[11]:


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


# In[12]:


def get_arg_idxs(pred_idx, conll):
    arg_idxs = [0 for i in range(len(conll))]
    for i in range(len(conll)):
        tok = conll[i]
        if int(tok[8]) == pred_idx:
            arg_pos = tok[-1]
            if arg_pos[:2] == 'NP':
                arg_idxs[i] = 1
                
    return arg_idxs


# In[13]:


def get_feature(pred_idxs, conll):
    result = []
    for i in pred_idxs:
#         print(i)
        features = []
        for j in range(len(i)):
            pred_idx = i[j]
            if pred_idx == 1:
                arg_idxs = get_arg_idxs(j, conll)
#                 print(arg_idxs)
        for j in range(len(i)):
            feature = []                
            feature.append(i[j])
            feature.append(arg_idxs[j])
            features.append(feature)                
        result.append(features)
        
    return result


# In[14]:


def prepare_sequence(seq, to_ix):
    vocab = list(to_ix.keys())
    idxs = []
    for w in seq:
        if w in vocab:
            idxs.append(to_ix[w])
        else:
            idxs.append(0)  

    return torch.tensor(idxs).cuda()


# In[15]:


def get_bert_vecs(tokens, args_in):      
    text = ' '.join(tokens) 
    bert_tokens, orig_to_token_map = tokenization(tokens)
    segments_ids = [0 for i in range(len(bert_tokens))]
    indexed_tokens = tokenizer.convert_tokens_to_ids(bert_tokens)
    
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    
    encoded_layers, pooled_output = bert_model(tokens_tensor, segments_tensors, output_all_encoded_layers=False)
    vecs = encoded_layers[0].cuda()
    
    return vecs, orig_to_token_map


# In[16]:


def get_dps(conll):
    dps = []
    for tok in conll:
        dp = tok[10]
        dps.append(dp)
    return dps


# In[17]:


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
            input_vec = []
            feat_vectors = features[i]
            bert_vecs, orig_to_token_map = get_bert_vecs(tokens,args_in)

            for tok_idx in range(len(feat_vectors)):                
                add_feat = torch.cat((bert_vecs[orig_to_token_map[tok_idx]], feat_vectors[tok_idx]))
                input_vec.append(add_feat)
            input_vec = torch.stack(input_vec)                   
            
            pred_seq = pred_idxs[i]
            for j in range(len(pred_seq)):
                p = pred_seq[j]
                if p == 1:
                    pred_idx = j
            arg_idxs = get_arg_idxs(pred_idx, conll)
            mask = torch.tensor(arg_idxs).cuda()
            mask = mask.float()

            tag_scores = my_model(input_vec, dp_in, mask)
            labels, score = get_labels_by_tensor(tag_scores)

            for idx in range(len(labels)):
                label = labels[idx]
                if label == '-':
                    pass
                else:
                    pred[idx] = label

        pred_result.write(str(pred)+'\n')       
 
    pred_result.close()
    with open(gold_file,'r') as f:
        gold = json.load(f)
    pred = eval_srl.read_prediction(pred_file)
    f1 = eval_srl.evaluate_from_list(pred, gold)
      
    return f1


# # model

# In[18]:


class LSTMTagger(nn.Module):
    
    def __init__(self, tagset_size):
        super(LSTMTagger, self).__init__()
        
        self.dp_embeddings = nn.Embedding(DP_VOCAB_SIZE, DPDIM)
        
        #LSTM layer        
        self.lstm_tok = nn.LSTM(LSTMINPDIM+DPDIM+FEATDIM, HIDDENDIM//2, bidirectional=True, num_layers=LSTMDEPTH, dropout=DROPOUT_RATE)
        self.hidden = self.init_hidden()
        
        # Linear
        self.hidden2tag = nn.Linear(HIDDENDIM, tagset_size)
              

    def init_hidden(self):
        return (torch.zeros(4, 1, HIDDENDIM//2).cuda(),
            torch.zeros(4, 1, HIDDENDIM//2).cuda())
    
    def forward(self, input_vecs, dp_in, mask):
        
        dp_embs = self.dp_embeddings(dp_in)
        
        input_embs = torch.cat( (input_vecs, dp_embs), 1)        
        input_embs = input_embs.view(len(input_vecs), 1, -1)
        
        # LSTM layer
        lstm_out_tok, self.hidden = self.lstm_tok(
            input_embs, self.hidden)
        
        # Linear
        tag_space = self.hidden2tag(lstm_out_tok.view(len(input_vecs),-1))        

        for t_idx in range(len(tag_space)):
            t = tag_space[t_idx]
            m = mask[t_idx]
            if m > 0:
                if penalty:
                    if t[-1] >= lambda_score:
                        t[-1] = lambda_score
            else:
                t[-1] = 1
        
        tag_space = F.relu(tag_space)
        
        softmax = nn.Softmax(dim=1)
        tag_scores = softmax(tag_space)
        return tag_scores       


# In[19]:


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
    n_of_step = 0
    n_of_sent = 1
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
            input_vec = []
            feat_vectors = features[i]
            bert_vecs, orig_to_token_map = get_bert_vecs(tokens,args_in)
            
            for tok_idx in range(len(feat_vectors)):
                add_feat = torch.cat((bert_vecs[orig_to_token_map[tok_idx]], feat_vectors[tok_idx]))
                input_vec.append(add_feat)
            input_vec = torch.stack(input_vec)         
            
            pred_seq = pred_idxs[i]
            for j in range(len(pred_seq)):
                p = pred_seq[j]
                if p == 1:
                    pred_idx = j
            arg_idxs = get_arg_idxs(pred_idx, conll)
            mask = torch.tensor(arg_idxs).cuda()
            mask = mask.float()

            srl_model.zero_grad()
            srl_model.hidden = srl_model.init_hidden()            

            tag_scores = srl_model(input_vec, dp_in, mask)
            loss = loss_function(tag_scores, args_in)
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(srl_model.parameters(), 0.25)
            optimizer.step()
            
            n_of_step +=1

            if n_of_step % 100 == 0:
                print('Epoch [{}/{}], Sent/Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, NUM_EPOCHS, n_of_sent, n_of_step, loss.item()))
        n_of_sent +=1
        
        break
        
    # EPOCH 마다 dev에 대한 성능 평가
            
    f1 = eval_dev(srl_model)
    print('Epoch [{}/{}], F1: {:4f}' 
                   .format(epoch+1, NUM_EPOCHS, f1))
    
    line = 'epoch '+str(epoch+1)+': '+str(f1)+'\n'
    epoch_dev_file.write(line)
    
    break
       
torch.save(srl_model, model_path)
print('')
print('### YOUR MODEL IS SAVED TO', model_path, '###')


# In[20]:


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

