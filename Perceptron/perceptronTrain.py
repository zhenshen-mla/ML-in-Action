import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Perceptron.perceptronAPI import Perceptron
from matplotlib.colors import ListedColormap


def make_dataset():
    """
    In this dataset, there are three kinds of flowers, namely Setosa, Versicolour and Virginica (Attribute 4).
    Attribute 0~3: sepal length, sepal width, petal length, petal width
    """
    # download dataset
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    df.tail()

    # make true label, two-classification
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    # extract sepal length and petal length as features
    x = df.iloc[0:100, [0, 2]].values

    # plot data
    plt.scatter(x[:50, 0], x[:50, 1],
                color='red', marker='o', label='setosa')
    plt.scatter(x[50:100, 0], x[50:100, 1],
                color='blue', marker='x', label='versicolor')

    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')

    plt.show()
    return x, y


def train(model, x, y):

    model.fit(x, y)
    print(model.w)  # [-0.38309474 -0.70465937  1.8403282 ]
    # print(model.errors_list)  [1, 3, 3, 2, 1, 0, 0, 0, 0, 0]

    plt.plot(range(1, len(model.errors_list) + 1), model.errors_list, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Errors')

    plt.show()


def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')



if __name__ == '__main__':

    model = Perceptron(lr_base=0.1, iterations=10)
    x, y = make_dataset()
    np.random.seed(7)
    np.random.shuffle(x)
    np.random.seed(7)
    np.random.shuffle(y)
    train(model, x, y)
    # print(type(x), x.shape, type(y), y.shape)  <class 'numpy.ndarray'> (100, 2) <class 'numpy.ndarray'> (100,)
    plot_decision_regions(x, y, model)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()

