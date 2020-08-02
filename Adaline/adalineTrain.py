import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Adaline.adalineAPI import AdalineGD
from Adaline.adalineSGDAPI import AdalineSGD
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


def make_region(x, y, model, name):
    plot_decision_regions(x, y, classifier=model)
    plt.title('Adaline - '+name)
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.savefig('images/02_14_1.png', dpi=300)
    plt.show()


def lr_variation_GD():
    x, y = make_dataset()
    np.random.seed(7)
    np.random.shuffle(x)
    np.random.seed(7)
    np.random.shuffle(y)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

    ada1 = AdalineGD(iterations=10, lr_base=0.01).fit(x, y)
    ax[0].plot(range(1, len(ada1.errors_list) + 1), ada1.errors_list, marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Sum-squared-error')
    ax[0].set_title('Adaline - Learning rate 0.01')

    ax[1].plot(range(1, len(ada1.errors_list) + 1), np.log10(ada1.errors_list), marker='o')  # log function makes errors-trend more obvious
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('log(Sum-squared-error)')
    ax[1].set_title('Adaline - Learning rate 0.01')

    ada2 = AdalineGD(iterations=10, lr_base=0.0001).fit(x, y)
    ax[2].plot(range(1, len(ada2.errors_list) + 1), ada2.errors_list, marker='o')
    ax[2].set_xlabel('Epochs')
    ax[2].set_ylabel('Sum-squared-error')
    ax[2].set_title('Adaline - Learning rate 0.0001')

    # plt.savefig('images/02_11.png', dpi=300)
    plt.show()
    make_region(x, y, ada1, 'lr=0.01')
    make_region(x, y, ada2, 'lr=0.0001')


def data_norm_GD():
    x, y = make_dataset()
    np.random.seed(7)
    np.random.shuffle(x)
    np.random.seed(7)
    np.random.shuffle(y)
    # normalize
    x_std = np.copy(x)
    x_std[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
    x_std[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()

    ada = AdalineGD(iterations=15, lr_base=0.01)
    ada.fit(x_std, y)

    make_region(x_std, y, ada, 'gd')

    plt.plot(range(1, len(ada.errors_list) + 1), ada.errors_list, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')

    plt.tight_layout()
    # plt.savefig('images/02_14_2.png', dpi=300)
    plt.show()


def data_norm_SGD():
    x, y = make_dataset()
    np.random.seed(7)
    np.random.shuffle(x)
    np.random.seed(7)
    np.random.shuffle(y)
    # normalize
    x_std = np.copy(x)
    x_std[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
    x_std[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()

    ada = AdalineSGD(iterations=15, lr_base=0.01)
    ada.fit(x_std, y)

    make_region(x_std, y, ada, 'sgd')

    plt.plot(range(1, len(ada.errors_list) + 1), ada.errors_list, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')

    plt.tight_layout()
    # plt.savefig('images/02_14_2.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    data_norm_SGD()
    # data_norm()
    # lr_variation()

