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

    def __init__(self, train, valid, test, learning_rate=0.1, noiseRatio=0.3, hiddenLayerNeurons=100, epochs=30):
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
        


        self.noiseRatio = noiseRatio
        
        
        self.autoencLayers = []
        # First hidden layer
        number_of_1st_hidden_layer = hiddenLayerNeurons
        self.autoencLayers.append(LogisticLayer(train.input.shape[1],
                                                number_of_1st_hidden_layer,
                                                None,
                                                activation="sigmoid",
                                                is_classifier_layer=False))
        # Output layer
        self.autoencLayers.append(LogisticLayer(number_of_1st_hidden_layer,
                                                train.input.shape[1],
                                                None,
                                                activation="sigmoid",
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

            if verbose:
                validation = self.validiation_set.input
                error = 0
                for img in validation:
                    # without first element which is the bias "1"
                    imgWithoutBias = img[1:]
                    # set randomly entries to zero (depending on noiseRatio)
                    noisedImg = np.concatenate(([1], self._addNoise(imgWithoutBias)), axis=0) 
                    
                    self.autoencMLP._feed_forward(noisedImg)
                    error += np.sum(abs(imgWithoutBias - self.autoencMLP._get_output_layer().outp))
                    
                print("Total epoch error: {0}".format(error))

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

        for img in self.training_set.input:
            # without first element which is the bias "1"
            imgWithoutBias = img[1:]
            # set randomly entries to zero (depending on noiseRatio)
            noisedImg = np.concatenate(([1], self._addNoise(imgWithoutBias)), axis=0) 
            
            self.autoencMLP._feed_forward(noisedImg)
            #print("error: {0}".format(np.sum(imgWithoutBias - self.autoencMLP._get_output_layer().outp)))
            self.autoencMLP._compute_error(imgWithoutBias)  # target is the input img (without bias "1")
            self.autoencMLP._update_weights()
           
            
    def _addNoise(self, instance):
        # get the amount of zeros and ones in the noiseMask
        zerosSize = int(instance.shape[0] * self.noiseRatio)
        onesSize = instance.shape[0] - zerosSize
        # create mask array with the specific amounts of zeros and ones and shuffle it
        noiseMask = np.concatenate((np.zeros(zerosSize), np.ones(onesSize)), axis=0) 
        np.random.shuffle(noiseMask)
        # apply mask to instance
        return instance * noiseMask


    def _get_weights(self):
        """
        Get the weights (after training)
        """

        return self.autoencMLP._get_layer(0).weights
