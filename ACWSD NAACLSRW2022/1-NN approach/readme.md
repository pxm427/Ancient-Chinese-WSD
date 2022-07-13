# Instruction

a_avg.py: E = Combination(Eavg, Eg)  
a_concat.py: E = Combination(Eavg, Eg)  
ps: Eavg from Etrain: a_train_data_embed_avg.py, Eg from gloss: a_gloss_data.py

a_test_data_avg.py: Etest  
a_test_data_concat.py: Etest

a_similarity_avg: a_avg.py & a_test_data_avg.py  
a_similarity_concat: a_concat.py & a_test_data_concat.py
