import numpy as np


class AdalineSGD(object):

    def __init__(self, lr_base=0.01, iterations=10, shuffle=True, seed=None, attributes=2):
        self.lr_base = lr_base
        self.iterations = iterations
        self.shuffle = shuffle
        self.seed = seed
        self._initialize_weights(attributes)

    def fit(self, X, y):
        self.errors_list = []
        for i in range(self.iterations):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                # each sample update: SGD
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.errors_list.append(avg_cost)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.seed)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.lr_base * xi.dot(error)
        self.w_[0] += self.lr_base * error
        cost = 0.5 * error ** 2
        return cost

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)