import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.probability import FreqDist


def analysis(name, data):
    """
    Plot Zipfian distribution of words + true Zipfian distribution. Compute MSE.

    :param name: title of the graph
    :param data: list of words
    """
    frequency_distribution = FreqDist(data)  # token counter
    # sort words by frequency:
    dataframe = pd.DataFrame(frequency_distribution.most_common())

    X = np.array(range(len(dataframe.index)))  # use ranks for x values
    X = X + 1  # indexing needs to start from 1 for log scale
    Y = np.array(list(dataframe[1])) / len(data)  # use word frequency for y values

    K = X * Y  # Total count
    k = np.mean(K)  # average relative frequency
    f_k = k / X  # expected frequency by 1/r

    # intercept (b) and slope (m) coefficients for best fit line:
    m, b = np.polyfit(np.log(X), np.log(Y), 1)
    y_fit = np.exp(m * np.log(X) + b)  # predicted y-hat values for best fit line

    # Gettin' Zipfian up in here!
    # Z(r) = N / (C + r)^alpha
    if name == "English":
        C = 3
        alpha = 1
        N = 0.15
    elif name == "German":
        C = 6
        alpha = 1
        N = 0.15
    elif name == "Pirates":
        C = 4
        alpha = 1
        N = 0.15
    else:
        C = 0
        alpha = 1.125
        N = 0.15

    Z = N / ((C + X) ** alpha)

    mean_square_error = round(np.mean((Z - Y) ** 2), 10)  # MSE
    print(f"Mean Square Error for {name}:{mean_square_error:10}")

    plt.scatter(X, Y, s=2, label="Observed frequencies")
    plt.plot(X, y_fit, "r", label="Least squares fit")
    plt.plot(X, Z, "g", label="Mandelbrot distribution: 1.5/(5+k)^1")
    plt.plot(X, f_k, "b", label="Predicted frequencies by f ~ 1/r")
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel("relative frequency")
    plt.xlabel("rank")
    plt.legend()
    plt.title(name)
    plt.show()
