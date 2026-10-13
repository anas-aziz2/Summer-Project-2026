import numpy as np


def flyvbjerg_petersen(data):

    data = np.asarray(data, dtype=float).copy()

    block_sizes = []
    errors = []
    variances = []

    block_size = 1

    while len(data) >= 16:

        n = len(data)

        variance = np.var(data, ddof=1)

        error = np.sqrt(variance / n)

        block_sizes.append(block_size)
        errors.append(error)
        variances.append(variance)

        if n % 2 == 1:
            data = data[:-1]

        data = 0.5 * (data[0::2] + data[1::2])

        block_size *= 2

    return np.array(block_sizes), np.array(errors), np.array(variances)


def autocorrelation(data, max_lag):

    data = np.asarray(data)

    data = data - np.mean(data)

    variance = np.var(data)

    autocorr = []

    for lag in range(max_lag):

        if lag == 0:
            numerator = np.mean(data * data)

        else:
            numerator = np.mean(
                data[:-lag] * data[lag:]
            )

        autocorr.append(numerator / variance)

    return np.array(autocorr)


def sokal_window(acf, c=5.0):

    W = 1

    while True:

        tau = 0.5 + np.sum(acf[1:W+1])

        new_W = int(np.ceil(c * tau))

        if new_W >= len(acf):
            new_W = len(acf) - 1

        if new_W == W:
            break

        W = new_W

    return tau, W