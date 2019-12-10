import numpy as np
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA


def pca_analyze(model, lpc_known_RQs, lpc_unknown_RQs, save_figs=True, disp_figs=False):
    """
    Runs inputted events through most of the trained network, then applies PCA to visualize their embeddings after the penultmate layer.

    :param lpc_known_RQs: a batch of LPC-identified (class 1,2,3 or 4) RQ pulses of size [num_examples x num_RQs]
    :param lpc_unknown_RQs: a batch of LPC-unidentified (class 5) RQ pulses of size [num_examples x num_RQs]
    :param save_figs: Set to True to save the embedding pca plots
    :param disp_figs: Set to True to print the embedding pca plots to print to screen
    :return: None
    """

    # Concat the two pulse populations together temporarily:
    inputs = tf.concat((lpc_known_RQs, lpc_unknown_RQs), axis=0)

    # Pass inputted events through most of the network
    dense1_output = tf.nn.leaky_relu(model.BN1(model.dense1(inputs)))
    dense2_output = tf.nn.leaky_relu(model.BN2(model.dense2(dense1_output)))
    embeddings = model.BN3(model.dense3(dense2_output))

    # Grab the network's final outputs for labels as well. Runs from 0 to 3.
    net_labels = tf.argmax(model.call(inputs), 1)

    makeflat = PCA(n_components=2)
    # Reduce from num_RQs to 2 dimensions
    vectors = makeflat.fit_transform(embeddings)

    # Plot the 2D embeddings
    x_vec = vectors[:, 0]
    y_vec = vectors[:, 1]

    # Find ranges that encompass most of the data (no skewing for massive outliers)
    low_p = 0.5
    high_p = 99.5
    margin_factor = 0.2
    x_min = np.percentile(x_vec, low_p) - margin_factor * (np.percentile(x_vec, high_p) - np.percentile(x_vec, low_p))
    x_max = np.percentile(x_vec, high_p) + margin_factor * (np.percentile(x_vec, high_p) - np.percentile(x_vec, low_p))
    y_min = np.percentile(y_vec, low_p) - margin_factor * (np.percentile(y_vec, high_p) - np.percentile(y_vec, low_p))
    y_max = np.percentile(y_vec, high_p) + margin_factor * (np.percentile(y_vec, high_p) - np.percentile(y_vec, low_p))

    plt_rq_names = ['S1', 'S2', 'Single Photoelectron', 'Single Electron']
    plt_rq_abbrev = ['s1', 's2', 'sphe', 'se']

    plt.ion()

    # fig, axs = plt.subplots(1, 1, figsize=(18, 16))
    for i in range(4):
        fig = plt.figure(i + 1, figsize=(18, 16))

        # Plot the lpc-known examples of this class:
        cutoff = len(lpc_known_RQs)

        x_known = x_vec[:cutoff][net_labels[:cutoff] == i]
        y_known = y_vec[:cutoff][net_labels[:cutoff] == i]
        x_unknown = x_vec[cutoff:][net_labels[cutoff:] == i]
        y_unknown = y_vec[cutoff:][net_labels[cutoff:] == i]

        plt.scatter(x_vec, y_vec, c='gray', s=1)  # Plot all pulses
        plt.scatter(x_known, y_known, c='b', s=1)  # Plot LPC-known pulses
        plt.scatter(x_unknown, y_unknown, c='r', s=1, alpha=1)  # Plot LPC-unknown pulses

        plt.title('2D embedding projection for LPC-classified (blue) and LPC-unclassified (red) ' + plt_rq_names[i] + ' pulses')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.show()
        if save_figs:
            now = time.localtime()
            timestamp = int(1E8 * (now.tm_year - 2000) + 1E6 * now.tm_mon + 1E4 * now.tm_mday + 1E2 * now.tm_hour + now.tm_min)
            if not os.path.exists('png/'):
                os.mkdir('png/')
            fig.savefig('png/' + plt_rq_abbrev[i] + '_pulses_t' + str(timestamp) + '.png')

    if disp_figs:
        plt.show()
        # input('Displaying PCA plots. Press Enter to continue...')

    return