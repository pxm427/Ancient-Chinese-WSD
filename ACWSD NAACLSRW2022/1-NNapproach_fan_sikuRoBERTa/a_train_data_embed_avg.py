###E(avg) → embeddings(sentences of same target character with its specified meaning from train_data) → avg embeddings
###dim: 768
import os
import torch
import opencc
import argparse
import sklearn.metrics as M
import torch.nn.functional as F
from collections import defaultdict
#from transformers import AutoTokenizer, AutoModel


#tokenizer = AutoTokenizer.from_pretrained("ethanyt/guwenbert-base")
#model = AutoModel.from_pretrained("ethanyt/guwenbert-base")

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("SIKU-BERT/sikuroberta")
model = AutoModel.from_pretrained("SIKU-BERT/sikuroberta")

model.eval()
converter = opencc.OpenCC('t2s')

def get_embeddings(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs, return_dict=True)
    last_hidden = outputs.last_hidden_state
    return last_hidden

def load_train_data(file):
    train_data_dict = {}
    train_data_file = open(file, "r", encoding="utf-8").readlines()
    for line in train_data_file:
        line = line.strip().split(" ")
        character = line[0]
        meaning = line[-1]
        new_keys = character + meaning
        if new_keys not in train_data_dict.keys():
            train_data_dict[new_keys] =[]
        train_data_dict[new_keys].append(line[1:])
    return train_data_dict

def get_train_data_embed(sentence):
    input_sentence = sentence
    input_last_hidden = get_embeddings(input_sentence)
    char_avg_embed = input_last_hidden.mean(1)
    return char_avg_embed

train_dict = load_train_data('/clwork/xiaomeng/Test/ZuoZhuan/train_list_f.txt')


senses_vecs = {}
embed_vecs = {}
##save every embedding of sentence in new_keys
for key in train_dict.keys():
    for sent in train_dict[key]:
        sent_embed = get_train_data_embed(sent[0])
        if key not in embed_vecs.keys():
            embed_vecs[key] = []
        embed_vecs[key].append(sent_embed)

for key in embed_vecs.keys():
    all_embed = embed_vecs[key][0]
    ##embed_vec[key] => {'key1': [E1,E2,E3]}
    ##start: all_embed => E1
    for embed in embed_vecs[key][1:]:
    ##rest of embedding in embed_vecs[key]
    ##start with E2
        all_embed = all_embed + embed
    avg_embed = all_embed/len(embed_vecs[key])
    if key not in senses_vecs.keys():
        senses_vecs[key] = []
    senses_vecs[key].append(avg_embed)

#print(key, avg_embed.size())
#print(train_dict)
#print(embed_vecs)
#print(embed_vecs[key][1:])
#print(senses_vecs)

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, help="pxm", default='0')
    args = parser.parse_args()
    return args

if __name__ =="__main__":
    args = parse_arg()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'