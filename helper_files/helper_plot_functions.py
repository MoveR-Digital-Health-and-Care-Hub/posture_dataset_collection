import matplotlib.pyplot as plt
import numpy as np

def plotGridSensors(arr, figFile):
    fig, axs = plt.subplots(4, 4)
    axs[0, 0].plot(arr[0], linewidth=1)
    axs[0, 0].set_title('Sensor 1 A')
    axs[0, 1].plot(arr[1], linewidth=1)
    axs[0, 1].set_title('Sensor 1 B')
    axs[0, 2].plot(arr[2], linewidth=1)
    axs[0, 2].set_title('Sensor 1 C')
    axs[0, 3].plot(arr[3], linewidth=1)
    axs[0, 3].set_title('Sensor 1 D')

    axs[1, 0].plot(arr[4], linewidth=1)
    axs[1, 0].set_title('Sensor 2 A')
    axs[1, 1].plot(arr[5], linewidth=1)
    axs[1, 1].set_title('Sensor 2 B')
    axs[1, 2].plot(arr[6], linewidth=1)
    axs[1, 2].set_title('Sensor 2 C')
    axs[1, 3].plot(arr[7], linewidth=1)
    axs[1, 3].set_title('Sensor 2 D')

    axs[2, 0].plot(arr[8], linewidth=1)
    axs[2, 0].set_title('Sensor 3 A')
    axs[2, 1].plot(arr[9], linewidth=1)
    axs[2, 1].set_title('Sensor 3 B')
    axs[2, 2].plot(arr[10], linewidth=1)
    axs[2, 2].set_title('Sensor 3 C')
    axs[2, 3].plot(arr[11], linewidth=1)
    axs[2, 3].set_title('Sensor 3 D')

    axs[3, 0].plot(arr[12], linewidth=1)
    axs[3, 0].set_title('Sensor 4 A')
    axs[3, 1].plot(arr[13], linewidth=1)
    axs[3, 1].set_title('Sensor 4 B')
    axs[3, 2].plot(arr[14], linewidth=1)
    axs[3, 2].set_title('Sensor 4 C')
    axs[3, 3].plot(arr[15], linewidth=1)
    axs[3, 3].set_title('Sensor 4 D')

    for ax in axs.flat:
        ax.set(xlabel=' ', ylabel=' ')

    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(figFile+'.svg')
    plt.savefig(figFile+'.png')
    plt.show()


def plotSingleChannel(prefix, grasp, channel):
    # load npy
    emg = np.load(prefix + '\\' + str(grasp) + '\\emg_raw.npy')
    # plot data
    plt.plot(emg[channel])


def pltDatabyChannel(data, channel):
    plt.plot(data[channel])


def plotByTrial(hf, trial):
    for i in range(trial):
        print('I', i)
        n1 = hf.get(str(i))
        n1 = np.array(n1)
        print(n1[0, 0:n1.shape[1]].shape)
        print(n1.shape)
        arr = n1
        figFile = "power.jpg"
        plotGridSensors(arr, figFile)
