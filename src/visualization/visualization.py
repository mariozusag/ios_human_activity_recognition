# Mario Zusag mariozusag@gmail.com
# 07.03.19
# Purpose:
# Description:
import matplotlib.pyplot as plt


class Visualization:

    def __init__(self):
        pass

    def plot_n_ms(label, df, hertz=50, milliseconds=10000):
        every_n_ms = 1000.0 / hertz
        n_datapoints = int(milliseconds / every_n_ms)

        data = df[df['label'] == label][['acc.x', 'acc.y', 'acc.z']][:n_datapoints]
        axis = data.plot(subplots=True, figsize=(16, 12),
                         title=label)
        for ax in axis:
            ax.legend(loc='lower left')