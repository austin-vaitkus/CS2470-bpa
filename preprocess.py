import numpy as np
import tensorflow as tf
import os
import aLib
from aLib import dp

from scipy import io

def get_data(dataset_list, fields,use_these_classifiers):
    """
	Given a file path and a list of RQs returns the data for the specified dataset and RQs.
	:param file_path: file path for inputs and labels. Trial set is '/data/lux10_20160627T0824_cp24454'
	:param fields: list of strings containing the desired rq names to be loaded
	:param use_these_classifiers: list of classifier labels (0-5) that should be included in the training and test sets.
	:return: NumPy array of training rqs of size [num_pulses x num_pulse_RQs], classifier labels of size [num_pulses x 1] and pulse event index of size [num_pulses x 1] for the
	training and test sets.
	"""

    rqBasePath_list = []
    for ii in range(0, len(dataset_list)):
        rqBasePath_list.append("./data/{:s}/matfiles/".format(dataset_list[ii]))

    # Recover the requested rqs for the datasets provided
    rq = dp.concatRQsWcuts([rqBasePath_list[0]], fields)

    # Break apart events into pulses, then compile these into a list while assigning them an event index for later association when we look at event structures.

    test_set_fraction = 0.1
    rq_norm_mode = 0 #  Set to 0 for no normalization of RQs, 1 for normalization by the 80th percentile largest value, and 2 for normalization via the abs(maximum value)
    num_pulses_per_event = 10

    num_events = rq['pulse_classification'].shape[1]
    num_pulses_with_blanks = num_events * num_pulses_per_event

    # Pull all N pulse labels out into one long N x 1 tensor
    labels = np.reshape(a=rq['pulse_classification'].T, newshape=[-1,1])

    # Locate pulse-based RQs. These will be of size 10 x N for N pulses
    pulse_rq_list = []
    for key, value in rq.items():
        # Check for rqs of shape 10 x N and EXCLUDE pulse_classification (already grabbed above as the labels
        if np.array(rq[key]).shape == (10,num_events) and key != 'pulse_classification':
            pulse_rq_list.append(key)

    num_pulse_rqs = len(pulse_rq_list)

    # Compile all pulse-level RQs into a block of shape (10xnum_events) x num_pulse_rqs
    pulse_rqs = np.zeros(shape=[num_pulses_with_blanks, num_pulse_rqs])
    for i in range(num_pulse_rqs):
        pulse_rq_name = pulse_rq_list[i]
        if rq_norm_mode==0:
            rq_norm_factor = 1
        elif rq_norm_mode==1:
            rq_norm_factor = np.percentile(abs(rq[pulse_rq_name]), 80)
        elif rq_norm_mode==2:
            rq_norm_factor = np.max(abs(rq[pulse_rq_name]))
        pulse_rqs[:, i] = np.reshape(a=rq[pulse_rq_name].T, newshape=[1, -1])/rq_norm_factor
        # TODO: Add functionality to truncate rq outlier values?

    # Remove pulses from the RQ block that are not in the use_these_classifiers input
    empty_pulse_index = []
    for i in range(num_pulses_with_blanks):
        if labels[i]==0 or labels[i]==5:# or labels[i] == 4 or labels[i] == 3:
        # if not set(list(use_these_classifiers)).intersection(list(labels[i],)):
        #     empty_pulse_index.append(i)

    # Initialize array of length 10xnum_events (i.e. num_pulses_with_blanks) with an index repeating 10x.
    # This will be cut down in the same way as the empty pulses to give a unique index per pulse to
    # each event. This can be used later to re-associate pulses from an event together.
    pulse_event_index = np.repeat(range(num_events), 10)

    # Delete those pulse_rq colums, labels, and event indexes where the pulse was empty
    pulse_rqs = np.delete(arr=pulse_rqs, obj=empty_pulse_index, axis=0)
    labels = np.delete(arr=labels, obj=empty_pulse_index, axis=0)
    pulse_event_index = np.delete(arr=pulse_event_index, obj=empty_pulse_index, axis=0)

    num_real_pulses = len(labels)

    # Shuffle the three arrays on the same indexing
    shuffle_index = np.arange(num_real_pulses)
    np.random.shuffle(shuffle_index)
    pulse_rqs = pulse_rqs[shuffle_index,:]
    labels = labels[shuffle_index]
    pulse_event_index = pulse_event_index[shuffle_index]

    # Separate out training and test data. TODO: Makes sure to split cleanly on an event so we don't have events split between the two sets. Currently doesn't keep events together
    split_index = int(test_set_fraction*num_real_pulses)
    train_rqs = pulse_rqs[split_index:]
    train_labels = labels[split_index:]
    train_event_index = pulse_event_index[split_index:]

    test_rqs = pulse_rqs[:split_index]
    test_labels = labels[:split_index]
    test_pulse_event_index = pulse_event_index[:split_index]


    print('Pulse RQ data block pre-processed')
    return train_rqs, train_labels, train_event_index, test_rqs, test_labels, test_pulse_event_index
