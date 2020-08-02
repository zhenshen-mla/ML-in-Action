import numpy as np


class AdalineGD(object):

    def __init__(self, lr_base=0.01, iterations=50, seed=7, attributes=2):
        """
        lr_base: learning rate
        iterations:
        seed: initialize random seeds of parameters
        attributes: feature number
        """
        self.lr_base = lr_base
        self.iterations = iterations
        self.seed = seed
        # initialize
        rgen = np.random.RandomState(self.seed)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=1 + attributes)  # mean/standard deviation/shape
        # record the acc of each sample
        self.errors_list = []

    def fit(self, x, y):
        """
        x: [n_samples, n_features]. Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y: [n_samples]. True values.

        return: Object
        """
        # attention: each epoch take update, not each sample(perceptron)
        # Batch gradient descent
        for i in range(self.iterations):
            net_input = self.net_input(x)
            # Please note that the "activation" method has no effect
            # in the code since it is simply an identity function. We
            # could write `output = self.net_input(x)` directly instead.
            # The purpose of the activation is more conceptual, i.e.,
            # in the case of logistic regression (as we will see later),
            # we could change it to
            # a sigmoid function to implement a logistic regression classifier.
            output = self.activation(net_input)
            errors = (y - output)
            # print(x.shape, errors.shape)  (100, 2) (100,)
            self.w[1:] += self.lr_base * x.T.dot(errors)
            self.w[0] += self.lr_base * errors.sum()
            # MSE loss function
            cost = (errors**2).sum() / 2.0
            self.errors_list.append(cost)
        return self

    def net_input(self, x):
        """Calculate net input"""
        return np.dot(x, self.w[1:]) + self.w[0]

    def activation(self, x):
        """Compute linear activation"""
        return x

    def predict(self, x):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(x)) >= 0.0, 1, -1)