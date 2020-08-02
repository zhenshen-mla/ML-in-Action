import numpy as np


class Perceptron(object):

    def __init__(self, lr_base=0.01, iterations=50, seed=7, attributes=2):
        """
        lr_base: learning rate
        iterations:
        seed: initialize random seeds of parameters
        attributes: feature number
        """
        self.lr_base = lr_base
        self.n_iter = iterations
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
        # iter
        for i in range(self.n_iter):
            error = 0
            for xi, target in zip(x, y):
                print(xi.shape, target.shape)
                update = self.lr_base * (target - self.predict(xi))
                self.w[1:] += update * xi  # update weight
                self.w[0] += update  # update bias
                error += int(update != 0.0)
            self.errors_list.append(error)
        return self

    def net_input(self, x):
        y = np.dot(x, self.w[1:]) + self.w[0]  # self.w[0] is bias
        # print(x.shape, x)
        # print(y.shape, y)
        return y

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)