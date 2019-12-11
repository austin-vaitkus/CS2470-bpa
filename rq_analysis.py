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
              'time_since_livetime_start_samples',
              'time_until_livetime_end_samples',
              'aft_t1_samples',  # OG LPC
              'file_number',  # OG LPC
              'pulse_classification',
              ]
use_these_classifiers = (1, 2, 3, 4)

train_rqs, train_labels, train_event_index, test_rqs, test_labels, test_event_index, test_order_index, test_label_5, test_rqs_5, test_event_index_5, test_order_index_5, pulse_rq_list,time_since_5, time_until_5 = get_data(dataset_list, fields, use_these_classifiers)


#%%


def sumpodLoader(event_index, all_event_indices):
    
    sumpods = []
    
    filename = './data/sumpods/sumpods5_0_1000.npy'
    print('Loading sumpods from: '+filename)
    all_sumpods = np.load(filename, allow_pickle = True)
    
    if type(event_index)==int:
        sumpods = all_sumpods[list(all_event_indices).index(event_index)]
    else: 
        for e_i in np.array(event_index):
            sumpods += [all_sumpods[list(all_event_indices).index(e_i)]]
    
    
    if len(sumpods) == 0:
        print('No Sumpod Exists for that event_index. Try again!')
    
    return sumpods
        
# Note, use RQ list 3 for visualization

def pulseViewer(event_index, pulse_index, start_rqs, end_rqs, all_event_indices, all_pulse_indices):
    
    sumpod_viz = sumpodLoader(event_index, all_event_indices)
    
    for i in range(len(all_event_indices)):
        if all_event_indices[i] == event_index and all_pulse_indices[i] == pulse_index:
            rq_index = i
            break

    print(len(sumpod_viz))
    
    event_start = -len(sumpod_viz)/2
    event_end = len(sumpod_viz)/2
    samples = np.linspace(event_start,event_end,len(sumpod_viz))
    
    pulse_start = start_rqs[rq_index]
    pulse_end = end_rqs[rq_index]
    total_time = pulse_end - pulse_start
    buffer = 20 * total_time
    
    
    plt.plot(samples,sumpod_viz)
    plt.xlim((pulse_start-buffer, pulse_end+buffer));
    plt.ylim((0,100));


    print(pulse_start, pulse_end)   

    return None

# # Choose an event and a pulse index [0,9] that you'd like to view
# viable_event_index = list(set(sumpodIndex_small).intersection(test_event_index_5))
# event_index = viable_event_index[2]
# pulse_index = int(test_order_index_5[0])


# Run the visualization

pulseViewer(4,6,test_rqs_5[:,1], test_rqs_5[:,0],test_event_index_5[0:1000], test_order_index_5[0:1000])

#%%

# DEPRECATED
# def sumpodLoader(event_index):
#     sumpod_viz = [0]
    
#     sumpodIndex_small = list(np.load('./data/sumpods/events_forsumpods_small.npy', allow_pickle = True))
#     sumpodIndex_big = list(np.load('./data/sumpods/events_forsumpods_big_sliced.npy', allow_pickle = True))
    
#     if len(set(sumpodIndex_small).intersection(set([event_index]))) > 0: 
#         filename = './data/sumpods/sumpods_small.npy'
#         print('Loading sumpods from: '+filename)
#         sumpod = np.load(filename, allow_pickle = True)
#         sumpod_viz = sumpod[sumpodIndex_small.index(event_index)]
#     else:   
#         for i in range(len(sumpodIndex_big)):     
#             if len(set(sumpodIndex_big[i]).intersection(set([event_index]))) > 0:
#                 print('Loading sumpod from: ./data/sumpods/sumpods%s.npy'%(i+1))
#                 sumpod = np.load('./data/sumpods/sumpods%s.npy'%(i+1), allow_pickle = True)
#                 sumpod_viz = sumpod[sumpodIndex_big[i].index(event_index)]
#                 break
                
#     if len(sumpod_viz) == 1:
#         print('No Sumpod Exists for that event_index. Try again!')
    
#     return sumpod_viz
