import numpy as np
import time
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from preprocess import get_data
from pca_analysis import pca_analyze, K_Nearest_Neighbor


class Model(tf.keras.Model):
    def __init__(self, num_classes):
        """
        Model architecture for pulse classification. Contains forward pass, accuracy, and loss.
		"""
        super(Model, self).__init__()

        # Model Hyperparameters
        self.batch_size = 200
        self.num_classes = num_classes
        self.learning_rate = 2e-4
#        self.drop_rate = 0.1
        self.num_filters = 15
        self.kernel_size = 1
        self.strides = 1

        # Model Layers
#        self.conv1 = tf.keras.layers.Conv1D(self.num_filters, self.kernel_size, self.strides, padding='SAME', activation='relu')
        
        self.dense1 = tf.keras.layers.Dense(self.num_classes * 4, dtype=tf.float32, name='dense1')
        self.BN1 = tf.keras.layers.BatchNormalization(trainable = False)
        self.dense2 = tf.keras.layers.Dense(self.num_classes * 3, dtype=tf.float32, name='dense2')
        self.BN2 = tf.keras.layers.BatchNormalization(trainable = False)
        self.dense3 = tf.keras.layers.Dense(self.num_classes * 2, dtype=tf.float32, name='dense3')
        self.BN3 = tf.keras.layers.BatchNormalization(trainable = False)
        self.dense4 = tf.keras.layers.Dense(self.num_classes, dtype=tf.float32, name='dense4')

        # Initialize Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def call(self, inputs):
        """
        Performs the forward pass on a batch of RQs to generate pulse classification probabilities.

        :param inputs: a batch of RQ pulses of size [batch_size x num_RQs]
        :return: A [batch_size x num_classes] tensor representing the probability distribution of pulse classifications
        """

        # Forward pass on inputs
        dense1_output = tf.nn.leaky_relu(self.BN1(self.dense1(inputs)))
        dense2_output = tf.nn.leaky_relu(self.BN2(self.dense2(dense1_output)))
        dense3_output = tf.nn.leaky_relu(self.BN3(self.dense3(dense2_output)))
        dense4_output = self.dense4(dense3_output)

        # Probabilities of each classification
        probabilities = tf.nn.softmax(dense4_output)

        return (probabilities)

    def loss_function(self, probabilities, labels):
        """
        Calculate model's cross-entropy loss after one forward pass.

		:param probabilities: tensor containing probabilities of RQ classification prediction     [batch_size x num_classes]
		:param labels: tensor containing RQ classification labels                                 [batch_size x num_classes]

        :return: model loss as a tensor
        """
        return (tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probabilities)))

    def accuracy_function(self, probabilities, labels):
        """
		Calculate model's accuracy by comparing logits and labels.

		:param probabilities: tensor containing probabilities of RQ classification prediction     [batch_size x num_classes]
		:param labels: tensor containing RQ classification labels                                 [batch_size x num_classes]

		:return: model accuracy as scalar tensor
		"""
        correct_predictions = tf.equal(tf.argmax(probabilities, 1), np.transpose(labels))
        return (tf.reduce_mean(tf.cast(correct_predictions, dtype=tf.float32)))




def train(model, inputs, labels):
    """
    This function should train your model for one epoch
    Each call to this function should generate a complete trajectory for one episode (lists of states, action_probs,
    and rewards seen/taken in the episode), and then train on that data to minimize your model loss.
    Make sure to return the total reward for the episode.

    :param model: The model
    :param data: LUX RQ data, preprocessed
    :return: The total reward for the episode
    """
    # Options:
    shuffle_per_epoch = True
    print_every_x_percent = 20

    # If enabled, shuffle the training inputs and labels at start of epoch
    if shuffle_per_epoch:
        shuffle_index = np.arange(labels.shape[0])
        np.random.shuffle(shuffle_index)
        inputs = inputs[shuffle_index,:] 
        labels = labels[shuffle_index] 

    # Initialize variables
    t = time.time()
    print_counter = print_every_x_percent
    accuracy = 0
    batch_counter = 0

    # Loop through inputs in model.batch_size increments
    for start, end in zip(range(0, inputs.shape[0] - model.batch_size, model.batch_size),
                          range(model.batch_size, inputs.shape[0], model.batch_size)):
        batch_counter += 1

        # Redefine batched inputs and labels
        batch_inputs = inputs[start:end]
        batch_labels = labels[start:end]

        with tf.GradientTape() as tape:
            probabilities = model(batch_inputs)  # probability distribution for pulse classification
            loss = model.loss_function(probabilities, batch_labels)  # loss of model

        accuracy += model.accuracy_function(probabilities, batch_labels)
        # Update
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Print Current Progress
        if 100 * start / inputs.shape[0] >= print_counter:
            print_counter += print_every_x_percent  # Update print counter
            accuracy_mean = accuracy / batch_counter  # Get current model accuracy
            print("{0:.0%} complete, Time = {1:2.1f} min, Accuracy = {2:.0%}".format(end / inputs.shape[0], (time.time() - t) / 60, accuracy_mean))

    return None


def test(model, inputs, labels, is_testing_5 = False, quiet_mode=True):
    

    # Initialize variables
    print_every_x_percent = 20
    if is_testing_5:
        print_test_diagnostics = False
    else:
        print_test_diagnostics = True

    print_counter = print_every_x_percent
    batch_counter = 0
    accuracy = 0
    loss = 0
    predicted_labels = np.zeros(labels.shape)

    for start, end in zip(range(0, inputs.shape[0] - model.batch_size, model.batch_size),
                          range(model.batch_size, inputs.shape[0], model.batch_size)):
        batch_counter += 1

        batch_inputs = inputs[start:end]
        batch_labels = labels[start:end]

        probabilities = model(batch_inputs)
        loss += model.loss_function(probabilities, batch_labels)
        accuracy += model.accuracy_function(probabilities, batch_labels)

        # Store the predicted labels for the batch
        predicted_labels[start:end] = tf.reshape(tf.argmax(probabilities, axis=1), [-1, 1])

        if quiet_mode == False:
            if 100 * start / inputs.shape[0] >= print_counter:
                print_counter += print_every_x_percent  # Update print counter
                print('{0:.0%}'.format(print_counter/100))
                if is_testing_5:
                    prediction = np.argmax(np.array(probabilities),axis=1)
                    for i in range(model.num_classes):
                        number_of_pulses_in_label_predicted = np.sum(prediction==i)
                        print('Number of Label = {0:1d} predicted is : {1:1d}'.format(i,number_of_pulses_in_label_predicted))
                    
                else:
                    for i in range(model.num_classes):
                        number_of_pulses_in_label = np.array(probabilities)[np.where(np.array(tf.reshape(batch_labels,[-1])) == i)].shape[0]
                        number_correctly_identified = (np.argmax(np.array(probabilities)[np.where(np.array(tf.reshape(batch_labels,[-1])) == i)],axis=1)==i).sum()
                        print('Label = {0:1d}. {1:.0%} Correctly Predicted = {2:1d}/{3:1d}'.format(i,number_correctly_identified/number_of_pulses_in_label,number_correctly_identified,number_of_pulses_in_label))
                    
                        if number_of_pulses_in_label != number_correctly_identified:
                            incorrect_predictions = (np.argmax(np.array(probabilities)[np.where(np.array(tf.reshape(batch_labels,[-1])) == i)],axis=1))
                            print('\t Label =',i,'\n \t Labels:',incorrect_predictions[incorrect_predictions!=i])


    loss /= batch_counter
    accuracy /= batch_counter

    # Print diagnostic data on test batch:
    if print_test_diagnostics:
        # Truncate labels at end of last batch
        diag_elements = end
        diag_labels = labels[:diag_elements]
        diag_predictions = predicted_labels[:diag_elements]

        # Print actual distribution of classes:
        print('Per-class accuracies for entire test set:')
        for label in range(model.num_classes):
            actual_label_fraction = np.sum(diag_labels == label)/diag_elements        # What fraction of the data is this class
            pred_label_fraction = np.sum(diag_predictions == label)/diag_elements     # What fraction of the data does the net think is this class.
            label_accuracy = np.sum(np.logical_and(diag_labels == label, diag_predictions == label)) / np.sum(diag_labels == label) # Fraction of this class that were correctly identified
            print('Class: %i \t Actual fraction:%.2f \t Predicted fraction:%.2f \t Class accuracy:%.2f' % (label, actual_label_fraction, pred_label_fraction, label_accuracy))
        print('')

    return accuracy  # , loss


def main():
    # %%

    # Parameters for the run
    dataset_switch = 3 # Use 2 for standard Kr dataset.
    RQ_list_switch = 1 # Use 1 to train on basic RQs, 2 for all available relevant RQs
    use_these_classifiers = (1, 2, 3, 4) # Net will ONLY train on the listed LPC classes.
    epochs = 10 # num of times during training to loop over all data for an independent training trial.
    num_trials = 1  # Number of independent training/testing runs (trials) to perform
    save_figs = True
    disp_figs = False
    label_list = ('S1','S2','SPE','SE')
   
    # Select the dataset to use
    if dataset_switch == 0:
        dataset_list = ["lux10_20160627T0824_cp24454"] # Short pulse DD data
    elif dataset_switch == 1:
        dataset_list = ['lux10_20160802T1425']  # Small piece of Kr + DD data
    elif dataset_switch == 2:
        dataset_list = ['lux10_20130425T1047']  # Run03 Kr83 dataset. Target ~10 Hz of Krypton. 330 MB
    elif dataset_switch == 3:
        dataset_list = ['lux10_20130425T1047_half']  # Run03 Kr83 dataset. Target ~10 Hz of Krypton. 2.4 GB


    # Decide which RQs to use. 1 for original LPC (1-to-1 comparison) vs 2 for larger list of RQs.
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
                  'pulse_classification',  # To be used as labels
                  'pulse_start_samples']  # To be used for pulse ordering (for identifying pulses in traces)

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
                  # 's1s2pairing', # may help later for finding Kr events?
                  'pulse_classification',  # To be used as labels
                  'pulse_start_samples']  # To be used for pulse ordering (for identifying pulses in traces)

    trial_test_acc = []
    trial_test_acc_5 = []
    for trial_index in range(num_trials):

        # Load and preprocess the data
        train_rqs, train_labels, train_event_index, test_rqs, test_labels, test_event_index, test_order_index, test_labels_5, test_rqs_5, test_event_index_5, test_order_index_5, _, _ = get_data(dataset_list, fields, use_these_classifiers)
        # train_labels = train_labels - 1
        # test_labels = test_labels - 1
        
        # Define model
        num_classes = len(use_these_classifiers)
        model = Model(num_classes)

        # Train model
        t = time.time()
        for epoch in range(epochs):
            train(model, train_rqs, train_labels)
            test_acc = test(model, test_rqs, test_labels)
            print("Epoch {0:1d} Complete.\nTotal Time = {1:2.1f} minutes, Testing Accuracy = {2:.0%}\n\n".format(epoch + 1, (time.time() - t) / 60, test_acc))
        trial_test_acc.append(test_acc)
        print('Training complete.')

        # Test distribution of LPC-unclassified pulses
        print('Testing LPC-unclassified pulses:')
        test_acc_5 = test(model, test_rqs_5, test_labels_5, is_testing_5=True, quiet_mode=True)
        trial_test_acc_5.append(test_acc_5)

        for pulse_type_index in range(4):
            # Use PCA to visualize clustering populations for the LPC-unclassified pulses
            labels_to_plot = pulse_type_index
            lpc_known_embeddings, lpc_unknown_embeddings, pca_known, pca_unknown, lpc_known_labels, lpc_unknown_labels = pca_analyze(model, test_rqs, test_rqs_5, labels_to_plot, test_event_index, test_event_index_5, test_order_index, test_order_index_5, save_figs=save_figs, disp_figs=disp_figs)
    
            # Create a "train" set, including labels, for the nearest neighbor algorithm
            # Here, 0's are considered inliers, 1's are considered outliers
            KNN_train = np.concatenate((pca_known, pca_unknown))
            KNN_labels = np.concatenate((np.zeros(len(pca_known)), np.ones(len(pca_unknown))))
        
            K_Nearest_Neighbor(KNN_train, KNN_labels, pca_unknown, label_list[labels_to_plot])

    # Summarize the num_trials independent trials with a list of final testing accuracies
    if num_trials>1:
        print('\n%i independent training trials complete. Final testing accuracies:' % num_trials)
        print(*np.array(trial_test_acc), sep='  ')
        print('Mean testing accuracy over all trials: %0.2f +/- %0.2f' % (np.average(trial_test_acc), np.std(trial_test_acc) ) )
    else:
        print('Training complete.')
    
    
if __name__ == '__main__':
    main()
