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

RQ_list_switch = 3
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
elif RQ_list_switch == 3:
    # RQs used by the standard LUX Pulse Classifier + Additional ones for better performance
    # Currently up-to-date with google sheets list as of 112719T1206
    fields = [
              'luxstamp_samples',  # OG LPC
              'pulse_start_samples',
              # 'pulse_length_samples',
              'pulse_end_samples',
              'aft_t1_samples',  # OG LPC
              'file_number',  # OG LPC
              'pulse_classification',
              ]
use_these_classifiers = (1, 2, 3, 4)

train_rqs, train_labels, train_event_index, test_rqs, test_labels, test_event_index, test_label_5, test_rqs_5, test_event_index_5,pulse_index_unique_5 = get_data(dataset_list, fields, use_these_classifiers)

#%%

# class1 = [True if x == 1 else False for x in train_labels]
# class2 = [True if x == 2 else False for x in train_labels]
# class3 = [True if x == 3 else False for x in train_labels]
# class4 = [True if x == 4 else False for x in train_labels]

# print(rq_list)

# for i in range(np.shape(train_rqs)[1]):
#     print('Max: %2.1f Mean:%2.1f Median:%2.1f'%(max(train_rqs[:,i][class1]), np.mean(train_rqs[:,i][class1]), np.median(train_rqs[:,i][class1])))
#     plt.hist(train_rqs[:,i][class1], bins = 30, histtype = 'step',label = 'S1')
#     print('Max: %2.1f Mean:%2.1f Median:%2.1f'%(max(train_rqs[:,i][class2]), np.mean(train_rqs[:,i][class2]), np.median(train_rqs[:,i][class2])))
#     plt.hist(train_rqs[:,i][class2], bins = 30, histtype = 'step',label = 'S2')
#     print('Max: %2.1f Mean:%2.1f Median:%2.1f'%(max(train_rqs[:,i][class3]), np.mean(train_rqs[:,i][class3]), np.median(train_rqs[:,i][class3])))
#     plt.hist(train_rqs[:,i][class3], bins = 30, histtype = 'step',label = 'SPE')
#     print('Max: %2.1f Mean:%2.1f Median:%2.1f'%(max(train_rqs[:,i][class4]), np.mean(train_rqs[:,i][class4]), np.median(train_rqs[:,i][class4])))
#     plt.hist(train_rqs[:,i][class4], bins = 30, histtype = 'step',label = 'SE')
#     plt.xlabel('%s'%rq_list[i])
#     plt.yscale('linear')
#     plt.legend()
#     plt.show()

#%%
sumpod_data_small = './data/sumpods_small.npy'
sumpod_data_big = './data/sumpods_big.npy'
sumpods = np.load(sumpod_data_small, allow_pickle = True)

sumpodIndex_small = list(np.load('./data/events_forsumpods_small.npy', allow_pickle = True))
sumpodIndex_big = list(np.load('./data/events_forsumpods_big.npy', allow_pickle = True))

all_event_index = np.concatenate((train_event_index,test_event_index))
all_rqs = np.concatenate((train_rqs,test_rqs))
all_labels = np.concatenate((train_labels,test_labels))

#%%

#TODO: pulse viewing... currently does not work
def pulseViewer(pulse_index, sumpods, use_5s = True):
    
    rq_start = 0
    rq_end = 1
    rq_aft_t1_samples = 2
    

    #First, find the event:
    event_index = list(pulse_IDs).index(pulse_index)
    
    # Now select the corresponding sumpod for that event
    # Here, I'm choosing from a list of events ('test_events') 
    # that I have previously declared as having 5s in them.
    # The order is important, that's why it's done this weird 
    sumpod_viz = sumpods[sumpod_event_index.index(event_index)]
    
    # Now we want to find the corresponding rq values for that event
        
    pulse_pick = list(rq_event_index).index(event_index)
    
    pulse_start = int(rqs[pulse_pick,rq_start]) 
    pulse_end = int(rqs[pulse_pick,rq_end]) 
    pulse_50 = int(rqs[pulse_pick,rq_aft_t1_samples]) 

    # Plot the results
    plt.plot(sumpod_viz, label = event_index)
    plt.xlim([pulse_start,pulse_end])
    plt.legend()
    plt.xlabel(pulse_50)
    plt.show()
    
    return None



# S2listindex = list(set(all_event_index[all_labels[:,0]==2]).intersection(eventIndex5_small))
# #S2listindex[0:100]
# #eventIndex5_small[0:10]

# for i in S2listindex[0:10]:
#     pulseViewer(i,
#                 eventIndex5_small, sumpods,
#                 all_event_index, all_rqs)
    
pulseViewer(pulse_index_unique_5[1006],sumpods)
