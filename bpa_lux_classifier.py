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

    dataset = ["lux10_20160627T0824_cp24454"]
    rq_list = []
    data_path = '/data/'
    file_path = data_path + dataset
    rqs = get_data(file_path, rq_list)


if __name__ == '__main__':
    main()

