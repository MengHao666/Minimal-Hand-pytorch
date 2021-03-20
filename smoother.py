import numpy as np


class LowPassFilter:
    def __init__(self):
        self.prev_raw_value = None
        self.prev_filtered_value = None

    def process(self, value, alpha):
        if self.prev_raw_value is None:
            s = value
        else:
            s = alpha * value + (1.0 - alpha) * self.prev_filtered_value
        self.prev_raw_value = value
        self.prev_filtered_value = s
        return s


class OneEuroFilter:
    def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0, freq=30):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()

    def compute_alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def process(self, x):
        prev_x = self.x_filter.prev_raw_value
        dx = 0.0 if prev_x is None else (x - prev_x) * self.freq
        edx = self.dx_filter.process(dx, self.compute_alpha(self.dcutoff))
        cutoff = self.mincutoff + self.beta * np.abs(edx)
        return self.x_filter.process(x, self.compute_alpha(cutoff))


if __name__ == '__main__':
    fliter = OneEuroFilter(4.0, 0.0)
    noise = 0.01 * np.random.rand(1000)
    x = np.linspace(0, 1, 1000)
    X = x + noise
    import matplotlib.pyplot as plt

    plt.plot(x)
    plt.plot(X)
    y = np.zeros((1000,))
    for i in range(1000):
        y[i] = fliter.process(x[i])
    plt.plot(y)
    plt.draw()
    plt.show()
