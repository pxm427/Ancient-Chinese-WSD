###E&E(test)
###dim = 1536
import numpy
import torch
import argparse
import numpy as np
import sklearn.metrics as M
import torch.nn.functional as F
from a_concat import concat_vecs_new
from a_test_data_concat import test_vecs_2, test_dict



for key in test_vecs_2.keys():
    gold_label_list = []
    output_label_list = []
    for test_embed in test_vecs_2[key]:
        gold_label_list.append(int(test_embed[1]))
        cos_max = 0.0
        output_label = 0
        for meaning_embeds in concat_vecs_new[key]:
            cos_sim = float(F.cosine_similarity(test_embed[0], meaning_embeds[1][0]))
            if cos_sim > cos_max:
                cos_max = cos_sim
                output_label = meaning_embeds[0][1:]
        output_label_list.append(int(output_label))
        
        with open('/users/kcnco/GitHub2021ACWSD/1NN/output_concat.txt', 'a+', encoding='utf-8')as f:
            f.write(f'char:{key}, sent:{test_embed[2]}, predict:{output_label}, gold:{int(test_embed[1])}\r\n')
        

    accuracy = M.accuracy_score(gold_label_list, output_label_list)    
    with open('/users/kcnco/GitHub2021ACWSD/1NN/accuracy_concat.txt', 'a+', encoding='utf-8')as f1:
        f1.write(f'char:{key}, acc:{accuracy}\r\n')


#print(output_label_list)
#print(gold_label_list)
#print(accuracy)
#print(test_number_dict)
#print(gold_label_list)

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, help="pxm", default='0')
    args = parser.parse_args()
    return args

if __name__ =="__main__":
    args = parse_arg()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'