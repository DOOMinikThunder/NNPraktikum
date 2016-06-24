# -*- coding: utf-8 -*-
import sys
import logging
import numpy as np
from model.logistic_layer import LogisticLayer
from model.auto_encoder import AutoEncoder
from model.mlp import MultilayerPerceptron
from sklearn.metrics import accuracy_score

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class DenoisingAutoEncoder(AutoEncoder):
    """
    A denoising autoencoder.
    """

    def __init__(self, train, valid, test, learning_rate=0.1, epochs=30):
        """
         Parameters
        ----------
        train : list
        valid : list
        test : list
        learning_rate : float
        epochs : positive int

        Attributes
        ----------
        training_set : list
        validation_set : list
        test_set : list
        learning_rate : float
        epochs : positive int
        performances: array of floats
        """
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.training_set = train
        self.validation_set = valid
        self.test_set = test
        
        
        
        self.autoencLayers = []
        # First hidden layer
        number_of_1st_hidden_layer = 100
        self.autoencLayers.append(LogisticLayer(train.input.shape[1],
                                                number_of_1st_hidden_layer,
                                                None,
                                                activation="sigmoid",
                                                is_classifier_layer=False))
        # Output layer
        self.autoencLayers.append(LogisticLayer(number_of_1st_hidden_layer,
                                                train.input.shape[1],
                                                None,
                                                activation="softmax",
                                                is_classifier_layer=True))

        self.autoencMLP = MultilayerPerceptron(self.training_set,
                                               self.validation_set,
                                               self.test_set,
                                               layers=self.autoencLayers,
                                               learning_rate=0.05,
                                               epochs=30)

        
        
    def train(self, verbose=True):
        """
        Train the denoising autoencoder
        """
        
        
        """
        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        # Run the training "epochs" times, print out the logs
        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))

            self._train_one_epoch()

#             if verbose:
#                 accuracy = accuracy_score(self.validation_set.label,
#                                           self.evaluate(self.validation_set))
#                 # Record the performance of each epoch for later usages
#                 # e.g. plotting, reporting..
#                 self.performances.append(accuracy)
#                 print("Accuracy on validation: {0:.2f}%"
#                       .format(accuracy * 100))
#                 print("-----------------------------")
        

    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """

        for img, label in zip(self.training_set.input,
                              self.training_set.label):

            # without first element which is the bias "1"
            target = img[1:]
            
            self.autoencMLP._feed_forward(img)
            self.autoencMLP._compute_error(target)
            self.autoencMLP._update_weights()
        

    def _get_weights(self):
        """
        Get the weights (after training)
        """

        return self.autoencMLP._get_layer(0).weights
