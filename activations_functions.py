import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


def leaky_relu(x):
    return np.maximum(0.1 * x, x)


def elu(x):
    return np.where(x > 0, x, np.exp(x) - 1)


def softplus(x):
    return np.log(1 + np.exp(x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


if __name__ == '__main__':
    plots = [sigmoid, relu, tanh, leaky_relu, elu, softplus, softmax]
    x = np.linspace(-5, 5, 100)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for plot in plots:
        ax.plot(x, plot(x), label=plot.__name__)
    ax.legend()
    plt.show()