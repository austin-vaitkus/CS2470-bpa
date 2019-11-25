import numpy as np
import tensorflow as tf
import os
import aLib
from aLib import dp

def get_data(file_path, rq_list):
	"""
	Given a file path and a list of RQs returns the data for the specified dataset and RQs.
	:param file_path: file path for inputs and labels. Trial set is '/data/lux10_20160627T0824_cp24454'
	:param rq_list: list of strings containing the desired rq names to be loaded
	:return: NumPy array of training rqs and classifier labels
	"""


	rq = None
	labels = None
	return rq, labels
