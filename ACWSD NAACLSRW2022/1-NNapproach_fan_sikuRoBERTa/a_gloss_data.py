###E(g) → Embeddings of target characters' each description from dictionary
###dim: 768
import os
import torch
import opencc
import argparse
import numpy as np
import sklearn.metrics as M
import torch.nn.functional as F
from torch._C import device
from collections import defaultdict
#from transformers import AutoTokenizer, AutoModel

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("SIKU-BERT/sikuroberta")
model = AutoModel.from_pretrained("SIKU-BERT/sikuroberta")


#model = AutoModel.from_pretrained("ethanyt/guwenbert-base")
#tokenizer = AutoTokenizer.from_pretrained("ethanyt/guwenbert-base")



model.eval()
converter = opencc.OpenCC('t2s')

def get_embeddings(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs, return_dict=True)
    last_hidden = outputs.last_hidden_state
    return last_hidden

##make a dict to save all the glosses
def load_gloss(file):
    gloss_dict = {}
    gloss_file = open(file, 'r', encoding='utf-8').readlines()
    for line in gloss_file:
        line = line.strip().split(" ")
        ##for example, the format of a sentence in gloss_file is like: [从 从之。 3]
        ##line[0] => 从
        ##line[1] => 从之。
        ##line[-1] => 3
        character = line[0]
        meaning = line[-1]
        #old_keys => 从
        #ne_ keys => 从3
        new_keys = character + meaning
        if new_keys not in gloss_dict.keys():
            gloss_dict[new_keys] = []
        ##make a dict to save glosses with new_keys
        ##format: {'从3': [从之。 3]
        ##          ...            }   
        gloss_dict[new_keys].append(line[1:])
    return gloss_dict

def get_gloss_embed(sentence):
    gloss = converter.convert(sentence)
    gloss_embed = get_embeddings(gloss)
    ##get the average embedding of a sentence(sense embedding of target character) => gloss embedding
    gloss_avg_embed = gloss_embed.mean(1)
    return gloss_avg_embed

gloss_dict = load_gloss('/clwork/xiaomeng/Test/KangxiDict/gloss_with_key.txt')

##append all the gloss embedding with key(new_keys) in gloss_vecs(a dict)
gloss_vecs = {}
for key in gloss_dict.keys():
    for gloss in gloss_dict[key]:
        ##calculate the embedding of a sentence
        ##gloss_dict[key] => '从3': [从之。 3]
        ##gloss[0] => gloss_dict[key][0] => 从之
        gloss_embed = get_gloss_embed(gloss[0])
        if key not in gloss_vecs.keys():
            gloss_vecs[key] = []
        ##save embeddding    
        gloss_vecs[key].append(gloss_embed)

##format of gloss_vecs is be like:
##{'从3': [tensor]
##  ...           }

#print(gloss_dict)       
#print(gloss_dict[key])
#print(gloss_embed) 
#print(gloss_vecs)
#print(key,gloss_embed.size())
#print(gloss_vecs[key][0].size())

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, help="pxm", default='0')
    args = parser.parse_args()
    return args

if __name__ =="__main__":
    args = parse_arg()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'