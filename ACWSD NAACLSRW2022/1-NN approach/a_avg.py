###E → E(avg)&E(g)
###dim: 768
import torch
import argparse
from collections import Counter
from a_gloss_data import gloss_vecs
from a_train_data_embed_avg import senses_vecs

avg_vecs = {}
for key in gloss_vecs.keys():
    if key in gloss_vecs.keys()&senses_vecs.keys():
        x = senses_vecs[key][0]
        y = gloss_vecs[key][0]
    else:
        x = gloss_vecs[key][0]
        y = gloss_vecs[key][0]
    vecs = 0.5*(x+y)##dim = 768
    if key not in avg_vecs.keys():
        avg_vecs[key] = []
    avg_vecs[key].append(vecs)

#for example
#gloss: 0 1 2 3
#train: 0 -1
#0: avg(gloss,train)
#1,2,3: avg(gloss,gloss)

avg_vecs_new = {}
for key in avg_vecs.keys():
    key_char = key[0]
    if key_char not in avg_vecs_new.keys():
        avg_vecs_new[key_char] = []
    avg_vecs_new[key_char].append([key, avg_vecs[key]])

##format:
##{'A':[['A0',[E1]],
##      ['A1',[E1]]]
##                   }
       

#print(concat_vecs.size())
#print(concat_vecs)
#print(concat_vecs_new)
#print(concat_vecs_new['君'][0][1])

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, help="pxm", default='0')
    args = parser.parse_args()
    return args

if __name__ =="__main__":
    args = parse_arg()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'