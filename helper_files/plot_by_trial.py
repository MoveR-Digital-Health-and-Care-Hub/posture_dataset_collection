import h5py
import numpy as np
import helper_plot_functions as plotFunctions


def plotByTrial(trial):
    for i in range(trial):
        print('I', i)
        n1 = hf.get(str(i))
        n1 = np.array(n1)
        print(n1[0, 0:n1.shape[1]].shape)
        print(n1.shape)
        arr = n1
        # a = -1
        # b = 1
        # for i in range(3,4):
        #     arr[i] = a + ((n1[i, 0:n1.shape[1]] - np.min(n1[i, 0:n1.shape[1]])) * (b - a) / (np.max(n1[i, 0:n1.shape[1]]) - np.min(n1[i, 0:n1.shape[1]])))
        #
        figFile = "power.jpg"
        plotFunctions.plotGridSensors(arr, figFile)


def concatenateDataFromTrials(hf_rest, rmin, rmax):
    data = hf_rest.get(str(rmin-1))
    for i in range(rmin, rmax):
        n1 = hf_rest.get(str(i))
        data = np.concatenate([data, n1], axis=1)
    return data


def plotByBlock(hf, rmin, rmax):
    data  = concatenateDataFromTrials(hf, rmin, rmax)
    figFile = "power.jpg"
    plotFunctions.plotGridSensors(data, figFile)


if __name__ == '__main__':
    participant = 'participant_7\\participant7_day1_block1_a\\'
    prefix = 'C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\data\\posture_dataset_15_8\\'
    prefixParticipant = prefix+participant
    hf = h5py.File(prefixParticipant+'emg_raw.hdf5', 'r')
    plotByBlock(hf, 25, 30)
    # plotByTrial(1)
