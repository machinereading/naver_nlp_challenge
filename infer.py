
# coding: utf-8

# In[1]:


import torch
import json
import numpy as np

from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertModel
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from tqdm import tqdm, trange

import read_data
import eval_srl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0) 


# # Options

# In[2]:


MAX_LEN = 256
batch_size = 6


# # gen data

# In[3]:


# load data
data = read_data.load_trn_data()

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


# In[4]:


tags_vals = ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARGM-EXT', 'ARGM-LOC', 'ARGM-DIR', 'ARGM-MNR', 'ARGM-TMP', 'ARGM-CAU', 'ARGM-INS', 'ARGM-PRP', '-']
tag2idx = {}
for i in tags_vals:
    tag2idx[i] = len(tag2idx)


# # Load BERT Model

# In[5]:


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
def bert_tokenizer(text):
    orig_tokens = text.split(' ')
    bert_tokens = []
    orig_to_tok_map = []
    bert_tokens.append("[CLS]")
    for orig_token in orig_tokens:
        orig_to_tok_map.append(len(bert_tokens))
        bert_tokens.extend(tokenizer.tokenize(orig_token))
    bert_tokens.append("[SEP]")
    
    return orig_tokens, bert_tokens, orig_to_tok_map


# # I/O

# In[6]:


# Input data example

def example():
    dummy = input_data[:500]
    print(dummy[:2])
    answer = [ d[2] for d in dummy ]


# # Processing

# In[7]:


def gen_input(data):
    tokenized_texts = []
    
    orig_to_tok_maps = []

    for i in range(len(data)):    
        d = data[i]
        text = ' '.join(d[1])
        orig_tokens, bert_tokens, orig_to_tok_map = bert_tokenizer(text)
        orig_to_tok_maps.append(orig_to_tok_map)
        tokenized_texts.append(bert_tokens)

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
#     tag2idx = {t: i for i, t in enumerate(tags_vals)}
    attention_masks = [[float(i>0) for i in ii] for ii in input_ids]
    
    inputs = torch.tensor(input_ids)
    input_masks = torch.tensor(attention_masks)
    
    return inputs, input_masks, orig_to_tok_maps


# In[8]:


def infer(data, model):
    model.eval()
    predictions = []
   
    inputs, input_masks, orig_to_tok_maps = gen_input(data)
    
    n = 0
    for i in range(len(inputs)):
        input_idxs = inputs[i].view(1, len(inputs[i])).to(device)
        input_mask = input_masks[i].view(1, len(input_masks[i])).to(device)
        orig_to_tok_map = orig_to_tok_maps[i]       

        with torch.no_grad():
            logits = model(input_idxs, token_type_ids=None,
                           attention_mask=input_mask)

        logits = logits.detach().cpu().numpy()
        
        bert_pred = [list(p) for p in np.argmax(logits, axis=2)]
        token_pred = []
        for idx in orig_to_tok_map:
            token_pred.append(bert_pred[0][idx])
        
        predictions.extend([token_pred])

    pred_tags = [[tags_vals[p_i] for p_i in p] for p in predictions]
    
    return pred_tags
    
#     print(pred_tags)
#     print("Validation loss: {}".format(eval_loss/nb_eval_steps))
#     print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
#     print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))


# # IO example

# In[9]:


def example():
    # 1) load MODEL
    model = torch.load('./result/model-bert/basic-model.pt')
    
    # 2) Input example
    print('\ninput example')
    dummy = input_data[:50]
    print(dummy[:2])
    
    # 3) infer example
    pred = infer(dummy, model)
    print('\noutput example')
    print(pred[:2])
    
    # 4) evaluation example
    answer = [ d[2] for d in dummy ]
    gold = []
    for i in dummy:
        gold += i[2]
    predict = []
    for i in pred:
        predict += i
        
    f1 = eval_srl.evaluate_from_list(predict, gold)
    print('\nf1:', f1)
    
# example()

