
# coding: utf-8

# In[89]:


import json
from collections import Counter
from src import etri


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


# In[91]:


def stats():
    print('# of sentences:', len(trn))
    
    arg_types = []
    for sent in trn:
        for token in sent:
            arg_type = token[2]
            if arg_type != '-':
                arg_types.append(arg_type)
            
    print('# of arg_types')
    print('\ttotal:', len(arg_types), '('+str(round(len(arg_types)/len(trn), 4))+' arg-per-sent)')
    print('\tunique:', len(list(set(arg_types))))
    print('\tfor each:', Counter(arg_types))
                
stats()


# In[53]:


# for sent in trn:
#     tokens = []
#     for t in sent:
#         tokens.append(t[1])
#     text = ' '.join(tokens)
#     for t in range(len(tokens)):   
#         token = tokens[t]
#         if t != len(tokens)-1:
#             if t > 1:
#                 token = token.replace('.\'', '\'')
#                 if '.' in token:
#                     print(text)
        


# In[92]:


def gen_nlp_file():
    result = []
    f = open('./data/trn_nlp.conll','w')
    
    c = 0
    n = 1
    sent_n = 0
    for sent in trn:
#         if c > 27050:
        tokens = []
        for t in sent:
            tokens.append(t[1])




        new_tokens = []
        if tokens[0] == '호사다마(好事多魔)라더니':
            tokens[0] = '호사다마(好事多魔)라'
#             if tokens[1] == '서녘인지':
#                 tokens[2] = '뭔지를'
        old_text = ' '.join(tokens)   

        for t in range(len(tokens)):
            token = tokens[t]

            token = token.replace('“', '\'')
            token = token.replace('”', '\'')
            token = token.replace('\"', '\'')
            token = token.replace(']', '\'')
            token = token.replace('[', '\'')
            token = token.replace('뭔지', '무언지')


            if t != len(tokens) -1:

                token = token.replace('….', '…')

                if token == '.':
                    token = ','

                if token[-1] == '.':
                    token = token[:-1]+','

                if token[-1] == '?':
                    token = token[-1]+'?,'

                token = token.replace('.\'', '\'')
                token = token.replace('?', '??')



            new_tokens.append(token)

        text = ' '.join(new_tokens)
        nlp = etri.getETRI(text)
        conll = etri.getETRI_CoNLL2009(nlp)

        for token in conll:
            token = [str(i) for i in token]
            line = '\t'.join(token)
            line = line+'\n'
            f.write(line)

        f.write('\n')       
        print('parsing',sent_n,'is done')
        sent_n += 1

        if len(sent) != len(conll):
            n +=1
            print(len(sent))
            print(tokens)
            print(new_tokens)
            print(len(conll))
            break
        c +=1
#         break

    f.close()
    print(sent_n, 'sents nlp is generated')
# gen_nlp_file()


# In[96]:


def get_nlp(sent):

    tokens = []
    for t in sent:
        tokens.append(t[1])
    new_tokens = []
    if tokens[0] == '호사다마(好事多魔)라더니':
        tokens[0] = '호사다마(好事多魔)라'
#             if tokens[1] == '서녘인지':
#                 tokens[2] = '뭔지를'
    old_text = ' '.join(tokens)   

    for t in range(len(tokens)):
        token = tokens[t]

        token = token.replace('“', '\'')
        token = token.replace('”', '\'')
        token = token.replace('\"', '\'')
        token = token.replace(']', '\'')
        token = token.replace('[', '\'')
        token = token.replace('뭔지', '무언지')


        if t != len(tokens) -1:
            token = token.replace('….', '…')
            if token == '.':
                token = ','
            if token[-1] == '.':
                token = token[:-1]+','
            if token[-1] == '?':
                token = token[-1]+'?,'
            token = token.replace('.\'', '\'')
            token = token.replace('?', '??')
        new_tokens.append(token)
    text = ' '.join(new_tokens)
    nlp = etri.getETRI(text)
    conll = etri.getETRI_CoNLL2009(nlp)

    return conll


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

trn_nlp = load_trn_nlp()

def get_nlp_for_trn(idx):
    return trn_nlp[idx]

