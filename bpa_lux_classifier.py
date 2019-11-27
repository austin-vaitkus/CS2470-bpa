import os
import sys
import numpy as np
import time
import subprocess
import tensorflow as tf
import matplotlib.pyplot as plt
import aLib
from aLib import dp
from preprocess import *

def train(model, data):
    """
    This function should train your model for one episode.
    Each call to this function should generate a complete trajectory for one episode (lists of states, action_probs,
    and rewards seen/taken in the episode), and then train on that data to minimize your model loss.
    Make sure to return the total reward for the episode.

    :param model: The model
    :param data: LUX RQ data, preprocessed
    :return: The total reward for the episode
    """

    return None


def main():

    # dataset_list = ["lux10_20160627T0824_cp24454"] # Short pulse DD data
    #dataset_list = ['lux10_20160802T1425']  # Small piece of Kr + DD data
    dataset_list = ['lux10_20130425T1047'] # Run03 Kr83 dataset. Target ~10 Hz of Krypton.

    # Generic pulse finding RQs
    # fields = ["pulse_area_phe", "luxstamp_samples", "pulse_classification",
    #           "s1s2_pairing", "z_drift_samples", "cor_x_cm", "cor_y_cm",
    #           "top_bottom_ratio", "rms_width_samples", "xyz_corrected_pulse_area_all_phe",
    #           "event_timestamp_samples", 'file_number']

    # Decide which RQs to use. 1 for original LPC (1-to-1 comparison) vs 2 for larger list of RQs.
    RQ_list_switch = 2
    if RQ_list_switch == 1:
        #Below: RQs used by the standard LUX Pulse Classifier
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
                  'pulse_classification',
                  #  ADD MORE RQs HERE 
                  ]
                  

    rq = get_data(dataset_list, fields)
    print('All RQs loaded!')

    # Create some variables for ease of broad event classification and population checking.
    pulse_classification = rq[0].pulse_classification
    num_pulses = np.sum(pulse_classification > 0, axis=0)
    num_S1s = np.sum(pulse_classification == 1,axis=0)
    num_S2s = np.sum(pulse_classification == 2,axis=0)
    num_se = np.sum(pulse_classification == 3, axis=0)
    num_sphe = np.sum(pulse_classification == 4, axis=0)





    print('[end file]')





if __name__ == '__main__':
    main()

