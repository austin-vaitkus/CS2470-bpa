import numpy as np
import tensorflow as tf
import os
import aLib
from aLib import dp

from scipy import io

def get_data(dataset_list, fields):
    """
	Given a file path and a list of RQs returns the data for the specified dataset and RQs.
	:param file_path: file path for inputs and labels. Trial set is '/data/lux10_20160627T0824_cp24454'
	:param fields: list of strings containing the desired rq names to be loaded
	:return: NumPy array of training rqs and classifier labels
	"""


    rqBasePath_list = []
    for ii in range(0, len(dataset_list)):
        # rqBasePath_list.append("/data/rq/{:s}/matfiles/".format(dataset[ii]))
        rqBasePath_list.append("./data/{:s}/matfiles/".format(dataset_list[ii]))

    d = dp.concatRQsWcuts([rqBasePath_list[0]], fields)

    rq = None
    labels = None
    return rq, labels
