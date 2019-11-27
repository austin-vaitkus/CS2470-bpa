import numpy as np
import tensorflow as tf
            
class DPC(tf.keras.Model):
    def __init__(self, rq_num):
        """
        Deep Pulse Classifier
        """
        super(Model, self).__init__()
        
        # Model Hyperparameters
        self.batch_size = 10
        self.num_classes = 4 # 1 = S1, 2 = S2, 3 = SE, 4 = SP 
        self.learning_rate = 1e-3
        
        self.dense1_size = 64
        self.rq_num = rq_num
​
        # Model Layers
        self.Dense1 = tf.keras.layers.Dense(self.dense1_size, 
                                            input_shape(self.rq_num,), 
                                            activation = 'relu',
                                            dtype=tf.float32, 
                                            name='dense1')
        self.dense2 = tf.keras.layers.Dense(self.num_classes, 
                                            dtype=tf.float32, 
                                            name='dense2')
​
        # Initialize Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        
        
​
    def call(self, inputs):
        """
        Performs the forward pass on a batch of RQs to generate pulse classification probabilities. 
​
        :param inputs: a batch of RQ pulses of size [batch_size x num_RQs] 
        :return: A [batch_size x num_classes] tensor representing the probability distribution of pulse classifications
        """
        
        # Forward pass on inputs
        dense1_output = self.dense1(inputs)
        dense2_output = self.dense1(dense1_output)
        
        # Probabilities of each classification
        probabilities = tf.nn.softmax(dense2_output)
​
        return(probabilities)
​
​
    def loss_function(self, probabilities, labels):
        """
        Calculate model's cross-entropy loss after one forward pass.
        
		:param probabilities: tensor containing probabilities of RQ classification prediction     [batch_size x num_classes]
		:param labels: tensor containing RQ classification labels                                 [batch_size x num_classes]
​
        :return: model loss as a tensor
        """
        return(tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probabilities)))
​
​
    def accuracy_function(self, probabilities, labels):
        """
		Calculate model's accuracy by comparing logits and labels.
        
		:param probabilities: tensor containing probabilities of RQ classification prediction     [batch_size x num_classes]
		:param labels: tensor containing RQ classification labels                                 [batch_size x num_classes]
        
		:return: model accuracy as scalar tensor
		"""
        correct_predictions = tf.equal(tf.argmax(probabilities, 1), tf.argmax(labels, 1))
        return(tf.reduce_mean(tf.cast(correct_predictions, dtype = tf.float32)))