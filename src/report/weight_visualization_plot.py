import matplotlib.image as mpimg


class WeightVisualizationPlot(object):
    '''
    Here to implement the visualization of the weights
    '''

    def __init__(self, mlp):
        self.path = "plots/weight_plots/"
        self.mlp = mlp

    def plot(self):
        layer = self.mlp._get_input_layer()
        for n_idx, column in enumerate(layer.weights.T):
            print("visualizing weights of neuron {}".format(n_idx))
            img = column[1:].reshape(28, 28)  # omit bias...
            mpimg.imsave(self.path + "neuron{0}.png".format(n_idx), img, cmap="gray")
