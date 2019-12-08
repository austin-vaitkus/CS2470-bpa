# -*- coding: utf-8 -*-
"""
For additional RQ analysis

"""
import os
import sys
import numpy as np
import time
import subprocess
import tensorflow as tf
import matplotlib.pyplot as plt
import aLib
from aLib import dp
from preprocess import get_data

#%%

RQ_list_switch = 2
dataset_list = ['lux10_20130425T1047']
if RQ_list_switch == 1:
    # Below: RQs used by the standard LUX Pulse Classifier
    fields = ['pulse_area_phe',  # OG LPC
              'luxstamp_samples',  # OG LPC
              's2filter_max_area_diff',  # OG LPC
              'prompt_fraction_tlx',  # OG LPC
              'top_bottom_asymmetry',  # OG LPC
              'aft_t0_samples',  # OG LPC
              'aft_t1_samples',  # OG LPC
              'aft_t2_samples',  # OG LPC
              'peak_height_phe_per_sample',  # OG LPC
              'skinny_peak_area_phe',  # OG LPC
              'prompt_fraction',  # OG LPC
              'pulse_height_phe_per_sample',  # OG LPC
              'file_number',  # OG LPC
              'pulse_classification']
elif RQ_list_switch == 2:
    # RQs used by the standard LUX Pulse Classifier + Additional ones for better performance
    # Currently up-to-date with google sheets list as of 112719T1206
    fields = ['pulse_area_phe',  # OG LPC
              'luxstamp_samples',  # OG LPC
              's2filter_max_area_diff',  # OG LPC
              'prompt_fraction_tlx',  # OG LPC
              'top_bottom_asymmetry',  # OG LPC
              'aft_t0_samples',  # OG LPC
              'aft_t05_samples',
              'aft_t25_samples',
              'aft_t1_samples',  # OG LPC
              'aft_t75_samples',
              'aft_t95_samples',
              'aft_t2_samples',  # OG LPC
              'peak_height_phe_per_sample',  # OG LPC
              'skinny_peak_area_phe',  # OG LPC
              'prompt_fraction',  # OG LPC
              'pulse_height_phe_per_sample',  # OG LPC
              'cor_x_cm',
              'cor_y_cm',
              'file_number',  # OG LPC
              # 's1s2pairing', # may help later for finding Kr events?
              'pulse_classification',
              ]
use_these_classifiers = (1, 2, 3, 4)

train_rqs, train_labels, train_event_index, test_rqs, test_labels, test_event_index, rq_list = get_data(dataset_list, fields, use_these_classifiers)

#%%

class1 = [True if x == 1 else False for x in train_labels]
class2 = [True if x == 2 else False for x in train_labels]
class3 = [True if x == 3 else False for x in train_labels]
class4 = [True if x == 4 else False for x in train_labels]

print(rq_list)

for i in range(np.shape(train_rqs)[1]):
    print(max(train_rqs[:,i][class1]), np.mean(train_rqs[:,i][class1]), np.median(train_rqs[:,i][class1]))
    plt.hist(train_rqs[:,i][class1], bins = 30, histtype = 'step',label = 'S1')
    print(max(train_rqs[:,i][class2]), np.mean(train_rqs[:,i][class2]), np.median(train_rqs[:,i][class2]))
    plt.hist(train_rqs[:,i][class2], bins = 30, histtype = 'step',label = 'S2')
    print(max(train_rqs[:,i][class3]), np.mean(train_rqs[:,i][class3]), np.median(train_rqs[:,i][class3]))
    plt.hist(train_rqs[:,i][class3], bins = 30, histtype = 'step',label = 'SPE')
    print(max(train_rqs[:,i][class4]), np.mean(train_rqs[:,i][class4]), np.median(train_rqs[:,i][class4]))
    plt.hist(train_rqs[:,i][class4], bins = 30, histtype = 'step',label = 'SE')
    plt.xlabel('%s'%rq_list[i])
    plt.yscale('linear')
    plt.legend()
    plt.show()



