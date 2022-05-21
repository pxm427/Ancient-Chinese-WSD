import numpy as np
from sklearn.model_selection import train_test_split


def load_2490(file):
    data = {}
    data_file = open(file, 'r', encoding='utf-8').readlines()
    for line in data_file:
        line = line.strip().split(' ')
        char = line[0]
        sense = line[-1]
        new_keys = char + sense
        if new_keys not in data.keys():
            data[new_keys] = []
        data[new_keys].append(line[0:])
    return data

all_data = load_2490('/users/kcnco/2021ACWSD/DataFile/Data/2490sents_first.txt')


with open('/users/kcnco/2021ACWSD/DataFile/Data/train_list.txt', 'a+', encoding='utf-8')as fa:
    for key in all_data.keys():
        if len(all_data[key]) == 1:
            str_sent_list_a = ' '.join(all_data[key][0])
            fa.write(f'{str_sent_list_a}\n')
            #print(key, all_data[key])            

with open('/users/kcnco/2021ACWSD/DataFile/Data/train_list.txt', 'a+', encoding='utf-8')as fb:
    for key in all_data.keys():
        if len(all_data[key]) > 1:
            y_train, y_test = train_test_split(all_data[key], test_size=0.2, random_state=42)
            for sent_list_train in y_train:
                str_sent_list_b = ' '.join(sent_list_train)
                fb.write(f'{str_sent_list_b}\n')
                #f.write(f'{str(sent_list)}\r\n')

with open('/users/kcnco/2021ACWSD/DataFile/Data/test_list.txt', 'w+', encoding='utf-8')as f:
    for key in all_data.keys():
        if len(all_data[key]) > 1:
            y_train, y_test = train_test_split(all_data[key], test_size=0.2, random_state=42)
            for sent_list_test in y_test:
                str_sent_list = ' '.join(sent_list_test)
                f.write(f'{str_sent_list}\r\n')
                #f.write(f'{str(sent_list)}\r\n')