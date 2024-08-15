import argparse
import pickle

import numpy as np
from tqdm import tqdm

'''
Function used to combine S-TR and T-TR into ST-TR
Function adapted from: https://github.com/kenziyuliu/Unofficial-DGNN-PyTorch
'''

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', default='kinetics', choices={'kinetics', 'ntu/xsub', 'ntu/xview'},
                    help='the work folder for storing results')
parser.add_argument('--alpha', default=1, help='weighted summation')
arg = parser.parse_args()

dataset = arg.datasets
label = open('new_data_processed/xset/val_label_120.pkl', 'rb')
label = np.array(pickle.load(label))
#code/results_FR_Head/NTU-120/x-sub/S-TR/test
r1 = open('./results_FR_Head/NTU-120/W_bones/xset/S-TR/epoch1_test_score.pkl', 'rb')
r1 = list(pickle.load(r1).items())
r2 = open('./results_FR_Head/NTU-120/W_bones/xset/T-TR/epoch1_test_score.pkl', 'rb')
r2 = list(pickle.load(r2).items())
right_num = total_num = right_num_5 = 0
print(label.size)
for i in tqdm(range(len(label[0]))):
    _, l = label[:, i]
    _, r11 = r1[i]
    _, r22 = r2[i]
    r = r11 + r22 * arg.alpha
    rank_5 = r.argsort()[-5:]
    right_num_5 += int(int(l) in rank_5)
    r = np.argmax(r)
    right_num += int(r == int(l))
    total_num += 1
acc = right_num / total_num
acc5 = right_num_5 / total_num
print(acc, acc5)