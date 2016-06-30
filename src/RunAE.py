#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
from data.mnist_seven import MNISTSeven

from model.denoising_ae import DenoisingAutoEncoder
from model.mlp import MultilayerPerceptron
from model.logistic_layer import LogisticLayer

from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot

# parameters that identify each run plots
dae_nr = 0.3
dae_lr = 0.05
dae_epochs = 30
mlp_lr = 0.05
mlp_epochs = 30
hiddenLayerNeurons = 100

if len(sys.argv) == 7:
    print "-----> Running new configuration from script <-----"
    dae_nr = sys.argv[1]
    dae_lr = sys.argv[2]
    dae_epochs = sys.argv[3]
    mlp_lr = sys.argv[4]
    mlp_epochs = sys.argv[5]
    hiddenLayerNeurons = sys.argv[6]

filename = "dae_nr" + str(dae_nr)+ "_" + "dae_lr" + str(dae_lr) + "_" \
            + "dae_epochs" + str(dae_epochs) + "_" + "mlp_lr" + str(mlp_lr) + "_" \
            + "mlp_epochs" + str(mlp_epochs) + "_" + "hiddenLayerNeurons" \
            + str(hiddenLayerNeurons)

print filename

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

    myDAE = DenoisingAutoEncoder(data.training_set,
                                 data.validation_set,
                                 data.test_set,
                                 learning_rate=dae_lr,
                                 noiseRatio=dae_nr,
                                 hiddenLayerNeurons=hiddenLayerNeurons,
                                 epochs=dae_epochs)
 
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
                                           learning_rate=mlp_lr,
                                           epochs=mlp_epochs)
    
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

    os.chdir("..")
    # Draw
    plot = PerformancePlot("DAE + MLP on MNIST task on validation set")
    plot.draw_performance_epoch(myMLPClassifier.performances, \
            myMLPClassifier.epochs, "plots", filename)

if __name__ == '__main__':
    main()
