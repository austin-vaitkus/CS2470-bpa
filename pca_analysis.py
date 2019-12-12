import numpy as np
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA

from sklearn import neighbors
from matplotlib.colors import ListedColormap

from matplotlib import path

def pca_analyze(model, lpc_known_RQs, lpc_unknown_RQs, labels_to_plot, lpc_known_event_index, lpc_unknown_event_index, lpc_known_order_index, lpc_unknown_order_index, save_figs=True, disp_figs=False):
    """
    Runs inputted events through most of the trained network, then applies PCA to visualize their embeddings after the penultmate layer.

    :param model: The trained classification model from bpa_lux_classifier.py.
    :param lpc_known_RQs: a batch of LPC-identified (class 1,2,3 or 4) RQ pulses of size [num_examples x num_RQs]
    :param lpc_unknown_RQs: a batch of LPC-unidentified (class 5) RQ pulses of size [num_examples x num_RQs]
    :param save_figs: Set to True to save the embedding pca plots
    :param disp_figs: Set to True to print the embedding pca plots to print to screen
    :return: None
    """

    print('Beginning PCA analysis...')

    # Concat the two pulse populations together temporarily:
    inputs = tf.concat((lpc_known_RQs, lpc_unknown_RQs), axis=0)

    # Pass inputted events through the first 3 layers of trained model to extract functional embeddings
    dense1_output = tf.nn.leaky_relu(model.BN1(model.dense1(inputs)))
    dense2_output = tf.nn.leaky_relu(model.BN2(model.dense2(dense1_output)))
    embeddings = model.BN3(model.dense3(dense2_output))
    # Grab the network's final outputs for labels as well. Runs from 0 to 3.
    net_labels = tf.argmax(model.call(inputs), 1)

    cutoff = len(lpc_known_RQs)
    lpc_known_labels = net_labels[:cutoff]
    lpc_unknown_labels = net_labels[cutoff:]
    lpc_known_embeddings = embeddings[:cutoff, :]
    lpc_unknown_embeddings = embeddings[cutoff:, :]


    pca_known, pca_unknown = pca_plot_subset(lpc_known_embeddings, lpc_unknown_embeddings, lpc_known_labels, lpc_unknown_labels, labels_to_plot=labels_to_plot, save_figs=save_figs, disp_figs=disp_figs)


    # Remove pulses that are not in the event indices for known events
    del_pulse_index = []
    for i in range(len(lpc_known_labels)):
        if np.sum(labels_to_plot == lpc_known_labels[i]) == 0:
            del_pulse_index.append(i)
    lpc_known_embeddings = np.delete(arr=lpc_known_embeddings, obj=del_pulse_index, axis=0)
    lpc_known_labels = np.delete(arr=lpc_known_labels, obj=del_pulse_index, axis=0)
    lpc_known_event_index = np.delete(arr=lpc_known_event_index, obj=del_pulse_index, axis=0)
    lpc_known_order_index = np.delete(arr=lpc_known_order_index, obj=del_pulse_index, axis=0)

    # Remove pulses that are not in the event indices for unknown events
    del_pulse_index = []
    for i in range(len(lpc_unknown_labels)):
        if np.sum(labels_to_plot == lpc_unknown_labels[i]) == 0:
            del_pulse_index.append(i)
    lpc_unknown_embeddings = np.delete(arr=lpc_unknown_embeddings, obj=del_pulse_index, axis=0)
    lpc_unknown_labels = np.delete(arr=lpc_unknown_labels, obj=del_pulse_index, axis=0)
    lpc_unknown_event_index = np.delete(arr=lpc_unknown_event_index, obj=del_pulse_index, axis=0)
    lpc_unknown_order_index = np.delete(arr=lpc_unknown_order_index, obj=del_pulse_index, axis=0)

    return lpc_known_embeddings, lpc_unknown_embeddings, pca_known, pca_unknown, lpc_known_labels, lpc_unknown_labels


def pca_plot_subset(lpc_known_embeddings, lpc_unknown_embeddings, lpc_known_labels, lpc_unknown_labels, labels_to_plot=(0, 1, 2, 3), save_figs=True, disp_figs=False):

    labels_to_plot = np.reshape(np.asarray(labels_to_plot), [-1])

    # Remove pulses that are not in the labels_to_plot input in the lpc_known events
    del_pulse_index = []
    for i in range(len(lpc_known_labels)):
        if np.sum(labels_to_plot == lpc_known_labels[i]) == 0:
            del_pulse_index.append(i)
    lpc_known_embeddings = np.delete(arr=lpc_known_embeddings, obj=del_pulse_index, axis=0)
    lpc_known_labels = np.delete(arr=lpc_known_labels, obj=del_pulse_index, axis=0)

    # Remove pulses that are not in the labels_to_plot input in the lpc_unknown events
    del_pulse_index = []
    for i in range(len(lpc_unknown_labels)):
        if np.sum(labels_to_plot == lpc_unknown_labels[i]) == 0:
            del_pulse_index.append(i)
    lpc_unknown_embeddings = np.delete(arr=lpc_unknown_embeddings, obj=del_pulse_index, axis=0)
    lpc_unknown_labels = np.delete(arr=lpc_unknown_labels, obj=del_pulse_index, axis=0)

    # Reduce from num_RQs to 2 dimensions
    makeflat = PCA(n_components=2)
    vectors = makeflat.fit_transform(tf.concat((lpc_known_embeddings, lpc_unknown_embeddings), axis=0))

    # Plot the 2D embeddings
    x_vec = vectors[:, 0]
    y_vec = vectors[:, 1]

    # Find ranges that encompass most of the data (no skewing for massive outliers)
    low_p = 0.5
    high_p = 99.5
    margin_factor = 0.25
    x_min = np.percentile(x_vec, low_p) - margin_factor * (np.percentile(x_vec, high_p) - np.percentile(x_vec, low_p))
    x_max = np.percentile(x_vec, high_p) + margin_factor * (np.percentile(x_vec, high_p) - np.percentile(x_vec, low_p))
    y_min = np.percentile(y_vec, low_p) - margin_factor * (np.percentile(y_vec, high_p) - np.percentile(y_vec, low_p))
    y_max = np.percentile(y_vec, high_p) + margin_factor * (np.percentile(y_vec, high_p) - np.percentile(y_vec, low_p))

    plt_rq_names = ['S1', 'S2', 'single phe', 'single electron']
    plt_rq_abbrev = ['s1', 's2', 'sphe', 'se']

    plt.ion()

    # fig, axs = plt.subplots(1, 1, figsize=(18, 16))

    for i in labels_to_plot:
        fig = plt.figure(i + 1, figsize=(21.5, 16))

        # Plot the lpc-known examples of this class:
        cutoff = len(lpc_known_labels)

        x_known = x_vec[:cutoff][lpc_known_labels == i]
        y_known = y_vec[:cutoff][lpc_known_labels == i]
        x_unknown = x_vec[cutoff:][lpc_unknown_labels == i]
        y_unknown = y_vec[cutoff:][lpc_unknown_labels == i]



        plt.clf()
        if len(labels_to_plot)>1:
            plt.scatter(x_vec, y_vec, c='gray', s=2.5, label='All pulses')  # Plot all pulses
        plt.scatter(x_known, y_known, c='b', s=6, label='LPC-classified pulses')  # Plot LPC-known pulses
        plt.scatter(x_unknown, y_unknown, c='r', s=6, label='LPC-unclassified pulses', alpha=1)  # Plot LPC-unknown pulses

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.title('2D embedding PCA projection for LPC-classified and unclassified ' + plt_rq_names[i] + ' pulses', fontsize=24)
        plt.xlabel('Principal Axis 1', fontsize=18)
        plt.ylabel('Princial Axis 2', fontsize=18)
        plt.legend(fontsize=18, loc=1)

        plt.show()
        if save_figs:
            now = time.localtime()
            timestamp = int(1E8 * (now.tm_year - 2000) + 1E6 * now.tm_mon + 1E4 * now.tm_mday + 1E2 * now.tm_hour + now.tm_min)
            if not os.path.exists('png/'):
                os.mkdir('png/')

            # Decide on a name and save
            indices_txt = str(labels_to_plot)
            indices_txt = indices_txt.replace(" ","").replace("[","").replace("]","")
            filename = plt_rq_abbrev[i] + '_pulses_over_labels_' + indices_txt + '_t' + str(timestamp) + '.png'
            fig.savefig('png/' + filename)

    if disp_figs:
        plt.show()
        # input('Displaying PCA plots. Press Enter to continue...')

    pca_known = vectors[:cutoff,:]
    pca_unknown = vectors[cutoff:,:]
    return pca_known, pca_unknown


def K_Nearest_Neighbor(known_points, labels, unknown_points, pulse_species, save_figs = True):
    
    # Set the number of neighbors to compare to
    n_neighbors = 5
    
    # Create an instance of Neighbours Classifier and fit the known data
    NNfunc = neighbors.KNeighborsClassifier(n_neighbors, weights = 'uniform')
    NNfunc.fit(known_points, labels)
    
    # Using the "trained" Neighbor classifier, predict the class of the 5's
    NN_class = NNfunc.predict(unknown_points)
  
    NN_0 = unknown_points[NN_class == 0]
    NN_1 = unknown_points[NN_class == 1]

    # Plot also the training points
    fig = plt.figure(figsize=(21.5, 16))
    
    plt.scatter(known_points[:, 0], known_points[:, 1], color='gray', s=2.5, label="deepLPC")
    plt.scatter(NN_0[:, 0], NN_0[:, 1], color='blue', edgecolors='blue', s=6, label="5s (Inliers)")
    plt.scatter(NN_1[:, 0], NN_1[:, 1], color='red', s=6, label="5s (Outliers)")

    # Find plot ranges that encompass most of the data (no skewing for massive outliers)
    low_p = 0.5
    high_p = 99.5
    margin_factor = 0.25
    x = known_points[:,0]
    y = known_points[:,1]
    x_min = np.percentile(x, low_p) - margin_factor * (np.percentile(x, high_p) - np.percentile(x, low_p))
    x_max = np.percentile(x, high_p) + margin_factor * (np.percentile(x, high_p) - np.percentile(x, low_p))
    y_min = np.percentile(y, low_p) - margin_factor * (np.percentile(y, high_p) - np.percentile(y, low_p))
    y_max = np.percentile(y, high_p) + margin_factor * (np.percentile(y, high_p) - np.percentile(y, low_p))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.legend(fontsize=18, loc=1)
    plt.title('Nearest Neighbor Classification for %s pulses'%pulse_species, fontsize=24)
    plt.xlabel('Principal Axis 1', fontsize = 18)
    plt.ylabel('Princial Axis 2', fontsize = 18)
    plt.show()
    
    if save_figs:
        now = time.localtime()
        timestamp = int(1E8 * (now.tm_year - 2000) + 1E6 * now.tm_mon + 1E4 * now.tm_mday + 1E2 * now.tm_hour + now.tm_min)
        if not os.path.exists('png/KNN/'):
            os.mkdir('png/KNN/')

        # Decide on a name and save
        filename = str(pulse_species) + '_KNN_' + str(timestamp) + '.png'
        fig.savefig('png/KNN/' + filename)
  
    return

#%%

def polyPath(polypath, xs, ys):

    polypath = np.concatenate((polypath,polypath[0].reshape(-1,2)))

    p = path.Path(polypath[:-1])
    
    xsys = np.array((xs,ys)).reshape((-1,2))
    contained = p.contains_points(xsys)
    notcontained = np.array([not i for i in contained])
    
    
    # Plot also the points within and outside the polygon path
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(xsys[:,0][notcontained],xsys[:,1][notcontained], color = 'gray', s = 10, label = 'Not Contained')
    plt.scatter(xsys[:,0][contained],xsys[:,1][contained], color = 'blue', s = 10,label = 'Contained')
    plt.plot(polypath[:,0],polypath[:,1], color = 'black', label = 'Path')
    
    # Find plot ranges that encompass most of the data (no skewing for massive outliers)
    low_p = 0.5
    high_p = 99.5
    margin_factor = 0.25
    x = xs
    y = ys
    x_min = np.percentile(x, low_p) - margin_factor * (np.percentile(x, high_p) - np.percentile(x, low_p))
    x_max = np.percentile(x, high_p) + margin_factor * (np.percentile(x, high_p) - np.percentile(x, low_p))
    y_min = np.percentile(y, low_p) - margin_factor * (np.percentile(y, high_p) - np.percentile(y, low_p))
    y_max = np.percentile(y, high_p) + margin_factor * (np.percentile(y, high_p) - np.percentile(y, low_p))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    return contained, notcontained


#%%
def cutSelection(xs, ys):
    # numCoord = integer number of coordinates to create polypath over
    plt.plot(xs,ys,'.')


    keep_going = True
    polypath = []
        
    # Prompt user for coordinates to cut over
    print('When you are done entering coordinates, just press enter for both x and y prompts.')
    i = 0
    while keep_going:
        x_coord = input('Enter x coordinate %1i:\t'%(i+1))
        y_coord = input('Enter y coordinate %1i:\t'%(i+1))
        if np.logical_and(len(x_coord) == 0,len(y_coord) == 0):
            keep_going = False
            polypath = np.array(polypath)
            print('All coordinates ',polypath)
        else:
            polypath.append([x_coord, y_coord])
        i+=1
    
    c,nc = polyPath(polypath,xs,ys)
#%%
#xs = np.random.uniform(0,1,1000)
#ys = np.random.uniform(0,1,1000)
#cutSelection(xs, ys)


 