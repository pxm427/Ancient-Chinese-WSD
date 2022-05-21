###E(test) → embedding of a sentence from test_data
###E(test) = concat(test,test)
###dim = 1536
import os
import torch
import opencc
import argparse
import numpy as np
import sklearn.metrics as M
import torch.nn.functional as F
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel

model = AutoModel.from_pretrained("ethanyt/guwenbert-base")
tokenizer = AutoTokenizer.from_pretrained("ethanyt/guwenbert-base")
model.eval()
converter = opencc.OpenCC('t2s')

def get_embeddings(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs, return_dict=True)
    last_hidden = outputs.last_hidden_state
    return last_hidden

def load_test_data(file):
    test_data_dict = {}
    test_data_file = open(file, 'r', encoding='utf-8').readlines()
    for line in test_data_file:
        line = line.strip().split(" ")
        character = line[0]
        if character not in test_data_dict.keys():
            test_data_dict[character] = []
        test_data_dict[character].append(line[1:])
    return test_data_dict

def get_test_data_embed(sentence):
    input_sentence = sentence
    input_last_hidden = get_embeddings(input_sentence)
    avg_embed = input_last_hidden.mean(1)
    return avg_embed

test_dict = load_test_data('/users/kcnco/GitHub2021ACWSD/Zuo Zhuan/1nn/test_list.txt')

test_vecs = {}
test_vecs_2 = {}
for key in test_dict.keys():
    for sent in test_dict[key]:
        test_embed = get_test_data_embed(sent[0])
        if key not in test_vecs.keys():
            test_vecs[key] = []
        test_vecs[key].append(test_embed)
        x = test_vecs[key][0]
        test_vecs_1536 = torch.cat([x, x], dim = 1)
        if key not in test_vecs_2.keys():
            test_vecs_2[key] = []
        test_vecs_2[key].append([test_vecs_1536,sent[1],sent[0]])

#format: {'key1':[[tesor,'0'],[tesor,'9']...]}


#print(test_dict)
#print(test_vecs)
#print(test_vecs[key])
#print(test_vecs_2.size())
#print(test_vecs_2[key][0].size())
#print(test_vecs_2)

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, help="pxm", default='0')
    args = parser.parse_args()
    return args

if __name__ =="__main__":
    args = parse_arg()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'