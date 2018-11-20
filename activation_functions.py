"""Activation Functions."""
from math import exp, tan


def sigmoid(x):
    """Sigmoid activation function."""
    return (1 / (1 + exp(-x)))


def tanh(x):
    """Tanh activation function."""
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))


def arctan(x):
    """Arctan activation function."""
    return tan(-1) * x


def gaussian(x):
    """Gaussian activation function."""
    return exp((-x) ** 2)
