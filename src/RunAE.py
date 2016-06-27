#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven

from model.denoising_ae import DenoisingAutoEncoder
from model.mlp import MultilayerPerceptron
from model.logistic_layer import LogisticLayer

from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot


def main():
    # data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
    #                  one_hot=True, target_digit='7')

    # NOTE:
    # Comment out the MNISTSeven instantiation above and
    # uncomment the following to work with full MNIST task
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                      one_hot=False)

    # NOTE:
    # Other 1-digit classifiers do not make sense now for comparison purpose
    # So you should comment them out, let alone the MLP training and evaluation

    # Train the classifiers #
    print("=========================")
    print("Training the autoencoder..")



    hiddenLayerNeurons = 100
    myDAE = DenoisingAutoEncoder(data.training_set,
                                 data.validation_set,
                                 data.test_set,
                                 learning_rate=0.05,
                                 noiseRatio=0.3,
                                 hiddenLayerNeurons=hiddenLayerNeurons,
                                 epochs=30)
 
    print("\nAutoencoder  has been training..")
    myDAE.train()
    print("Done..")
 
    # Multi-layer Perceptron
    # NOTES:
    # Now take the trained weights (layer) from the Autoencoder
    # Feed it to be a hidden layer of the MLP, continue training (fine-tuning)
    # And do the classification
 
 
    myMLPLayers = []
    # First hidden layer
    number_of_1st_hidden_layer = hiddenLayerNeurons
    myMLPLayers.append(LogisticLayer(data.training_set.input.shape[1]-1,    # bias "1" already added so remove one
                                     number_of_1st_hidden_layer,
                                     weights=myDAE._get_weights(),
                                     activation="sigmoid",
                                     is_classifier_layer=False))
    # Output layer
    number_of_output_layer = 10
    myMLPLayers.append(LogisticLayer(number_of_1st_hidden_layer,
                                     number_of_output_layer,
                                     None,
                                     activation="softmax",
                                     is_classifier_layer=True))
 
    # Correct the code here
    myMLPClassifier = MultilayerPerceptron(data.training_set,
                                           data.validation_set,
                                           data.test_set,
                                           layers=myMLPLayers,
                                           learning_rate=0.05,
                                           epochs=30)
    
    # remove double added bias "1"
    myMLPClassifier.__del__()



    print("\nMulti-layer Perceptron has been training..")
    myMLPClassifier.train()
    print("Done..")
    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    mlpPred = myMLPClassifier.evaluate()

    # Report the result #
    print("=========================")
    evaluator = Evaluator()

    # print("Result of the stupid recognizer:")
    # evaluator.printComparison(data.testSet, stupidPred)
    # evaluator.printAccuracy(data.test_set, stupidPred)

    # print("\nResult of the Perceptron recognizer (on test set):")
    # evaluator.printComparison(data.testSet, perceptronPred)
    # evaluator.printAccuracy(data.test_set, perceptronPred)

    # print("\nResult of the Logistic Regression recognizer (on test set):")
    # evaluator.printComparison(data.testSet, perceptronPred)
    # evaluator.printAccuracy(data.test_set, lrPred)

    print("\nResult of the DAE + MLP recognizer (on test set):")
    # evaluator.printComparison(data.testSet, perceptronPred)
    evaluator.printAccuracy(data.test_set, mlpPred)

    # Draw
    plot = PerformancePlot("DAE + MLP on MNIST task")
    plot.draw_performance_epoch(myMLPClassifier.performances,
                                myMLPClassifier.epochs)

if __name__ == '__main__':
    main()
