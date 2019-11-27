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


###################################

class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that 
		classifies images. Do not modify the constructor, as doing so 
		will break the autograder. We have left in variables in the constructor
		for you to fill out, but you are welcome to change them if you'd like.
		"""
        super(Model, self).__init__()
        
        # Model Hyperparameters
        self.batch_size = 10
        self.num_classes = 5
        self.learning_rate = 1e-3

        # Model Layers
        self.dense1 = tf.keras.layers.Dense(self.num_classes*10, activation = 'relu', dtype=tf.float32, name='dense1')
        self.dense2 = tf.keras.layers.Dense(self.num_classes, dtype=tf.float32, name='dense2')

        # Initialize Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        
        

    def call(self, inputs):
        """
        Performs the forward pass on a batch of RQs to generate pulse classification probabilities. 

        :param inputs: a batch of RQ pulses of size [batch_size x num_RQs] 
        :return: A [batch_size x num_classes] tensor representing the probability distribution of pulse classifications
        """
        
        # Forward pass on inputs
        dense1_output = self.dense1(inputs)
        dense2_output = self.dense1(dense1_output)
        
        # Probabilities of each classification
        probabilities = tf.nn.softmax(dense2_output)

        return(probabilities)


    def loss_function(self, probabilities, labels):
        """
        Calculate model's cross-entropy loss after one forward pass.
        
		:param probabilities: tensor containing probabilities of RQ classification prediction     [batch_size x num_classes]
		:param labels: tensor containing RQ classification labels                                 [batch_size x num_classes]

        :return: model loss as a tensor
        """
        return(tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probabilities)))


    def accuracy_function(self, probabilities, labels):
        """
		Calculate model's accuracy by comparing logits and labels.
        
		:param probabilities: tensor containing probabilities of RQ classification prediction     [batch_size x num_classes]
		:param labels: tensor containing RQ classification labels                                 [batch_size x num_classes]
        
		:return: model accuracy as scalar tensor
		"""
        correct_predictions = tf.equal(tf.argmax(probabilities, 1), tf.argmax(labels, 1))
        return(tf.reduce_mean(tf.cast(correct_predictions, dtype = tf.float32)))




################################

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
                    #  ADD MORE RQs HERE

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

