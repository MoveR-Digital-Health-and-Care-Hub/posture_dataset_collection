import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import h5py
import csv
import os
from sklearn import preprocessing
from axopy.features import WaveformLength, LogVar
from scipy.signal import butter
from configparser import ConfigParser

from axopy.pipeline import Pipeline, Windower, Filter, FeatureExtractor, Ensure2D


def sliding_window(array, window_size, step):
    num_windows = (array.shape[0]-window_size+1)/step
    print(int(num_windows))
    indexes = np.arange(window_size)[None, :] + step*np.arange(int(num_windows))[:, None]
    return array[indexes]


def extract_windows(array, clearing_time_index, max_time, sub_window_size):
    examples = []
    start = clearing_time_index + 1 - sub_window_size + 1

    for i in range(max_time + 1):
        example = array[start + i:start + sub_window_size + i]
        examples.append(np.expand_dims(example, 0))

    return np.vstack(examples)


def extract_windows_vectorized(array, clearing_time_index, max_time, sub_window_size):
    start = clearing_time_index + 1 - sub_window_size + 1

    sub_windows = (
            start +
            # expand_dims are used to convert a 1D array to 2D array.
            np.expand_dims(np.arange(sub_window_size), 0) +
            np.expand_dims(np.arange(max_time + 1), 0).T
    )

    return array[sub_windows]


def EisaPlot():
    # C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\data\\Eisa_recording\\Eisa_Squats_Edinburgh_2.csv


    a = np.array([])
    # with open("C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\data\\Eisa_recording\\power_20221012143756\\trials.csv", 'r') as file:
    with open("C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\data\\Eisa_recording\\Eisa_Squats_Edinburgh_2.csv", 'r') as file:
      csvreader = csv.reader(file)
      for row in csvreader:
        # print(row)
        x = row[1].split(",")
        a = np.append(a, int(x[0]))
        # a.append(int(x[0]))


    n1 = np.array(a)
    print(n1.shape)

    n1 = hf.get('0')
    n1 = np.array(n1)


    fig, axs = plt.subplots(4,1)
    axs[0].plot(n1[0],linewidth=1)
    axs[0].set_title('Sensor 1 A')
    axs[1].plot(n1[1],linewidth=1)
    axs[1].set_title('Sensor 1 B')
    axs[2].plot(n1[2],linewidth=1)
    axs[2].set_title('Sensor 1 C')
    axs[3].plot(n1[3],linewidth=1)
    axs[3].set_title('Sensor 1 D')

    for ax in axs.flat:
        ax.set(xlabel=' ', ylabel=' ')

    for ax in axs.flat:
        ax.label_outer()

    # plt.savefig("raw_data_plot.jpg")
    # plt.plot(n1[0, 0:n1.shape[1]],linewidth=1)
    plt.show()

    trials = 41
    n1 = hf.get('0')
    for i in range(2*trials):

        if int(i) % 2 == 0:
            print('Key', i)

            if int(i)>0:
                arr1 = hf.get(str(i))
                n1 = np.concatenate([n1, arr1], axis=1)



    n1 = np.array(n1)
    # print(n1[0, 0:n1.shape[1]].shape)
    print(n1.shape)

    plt.plot(n1[1])


    np.savetxt("C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\data\\Eisa_recording\\EMG_quatro.csv", n1, delimiter=",")

    #
    breakpoint()
    print(len(hf.keys()))




def extractFeatures(data):
    featMav = extractMAV(data)
    featLogVar = extractLogVar(data)
    return featMav


def EisaExtractFeatures(data):
    df = EisaReadData()
    feats = extractFeatures(df.emg)

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(df.emf, linewidth=1)
    axs[0].set_title('emg data Eisa')
    axs[1].plot(feats, linewidth=1)
    axs[1].set_title('features Eisa')


def EisaReadData():
    # Eisa's sensor
    df = pd.read_csv(
        'C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\data\\Eisa_recording\\Eisa_Squats_Edinburgh_2.csv')
    df.columns = ['time', 'emg', '1', '2', '3', '4', '5', '6', '7']
    return df


def make_pipeline():
    """Returns a processing pipeline. """
    b, a = butter(FILTER_ORDER,
                  (LOWCUT / S_RATE / 2., HIGHCUT / S_RATE / 2.),
                  'bandpass')
    pipeline = Pipeline([
        Windower(int(S_RATE * WIN_SIZE)),
        Filter(b, a=a,
               overlap=(int(S_RATE * WIN_SIZE) -
                        int(S_RATE * READ_LENGTH))),
        FeatureExtractor([('wl', WaveformLength()), ('logvar', LogVar())]),
        Ensure2D(orientation='col')
    ])

    return pipeline



def EisaDataPlot():
    # Eisa
    df = EisaReadData()
    plt.plot(df.emg)

    emgEisa = df.emg
    normEmgEisa = preprocessing.normalize([emgEisa])[0]
    plt.plot(normEmgEisa)


    # fig, axs = plt.subplots(2, 1)
    # axs[0].plot(emgEisa, linewidth=1)
    # axs[0].set_title('Raw Eisa')
    # axs[1].plot(normEmgEisa, linewidth=1)
    # axs[1].set_title('Norm Eisa')

    # Delsys
    hf = h5py.File(
        'C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\data\\Eisa_recording\\power_20221012143756\\emg_raw.hdf5',
        'r')
    print(hf.keys())

    # Delsys sensor
    emg1 = hf.get(str('0'))
    for i in range(1,len(hf.keys())):
        # print(i)
        if i%2==1:
            continue
        n1 = hf.get(str(i))
        emg1 = np.concatenate([emg1, n1], axis=1)
    plt.plot(emg1[0])
    normEmgDelsys = preprocessing.normalize([emg1[0]])[0]

    # fig, axs = plt.subplots(2, 1)
    # axs[0].plot(emg1[0], linewidth=1)
    # axs[0].set_title('raw Delsys')
    # axs[1].plot(normEmgDelsys, linewidth=1)
    # axs[1].set_title('Norm Delsys')


    # select 2 squats
    fromE = 25000 + 45500  # digital electrode
    fromD = 16000 + 45500  # Delsys
    offset = 9000
    emgEisaSq = normEmgEisa[fromE:fromE+offset]
    emgDelsysSq = normEmgDelsys[fromD:fromD+offset]

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(emgEisaSq, linewidth=1)
    axs[0].set_title('Sensor Eisa')
    axs[1].plot(emgDelsysSq, linewidth=1)
    axs[1].set_title('Sensor Delsys')

    # select motor units
    fromE = 25000 + 207800
    fromD = 16000 + 207800
    offset = 300
    emgEisaMU = normEmgEisa[fromE:fromE + offset]
    emgDelsysMU = normEmgDelsys[fromD:fromD + offset]
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(emgEisaMU, linewidth=1)
    axs[0].set_title('Sensor Eisa')
    axs[1].plot(emgDelsysMU, linewidth=1)
    axs[1].set_title('Sensor Delsys')

    # show features
    # sliding windows
    Y = extract_windows(emgEisa, 0, len(emgEisa), 128)
    # extract features
    # visualise features


    # show fatigue


def selectGrasps(grasps_sequence, gr_unique, hf):
    ind = 0
    gr_index = np.zeros([len(gr_unique), 5])

    for gr in gr_unique:
        gr_index[gr-1] = [i for i, x in enumerate(grasps_sequence) if x == gr]
        ind = ind+1

    emgShape = hf.get(str('0'))
    emgDict = {}

    for gr in gr_unique:
        print(gr)
        emgtest = np.zeros([emgShape.shape[0], emgShape.shape[1]])
        indt = 0
        for ind in gr_index[gr-1]:
            print(ind)
            n1 = hf.get(str(int(ind)))
            if (indt == 0):
                emgtest = n1
            else:
                emgtest = np.concatenate([emgtest, n1], axis=1)
            indt = indt + 1
        emgtest = np.array(emgtest)
        emgDict[str(gr-1)] = emgtest
    return emgDict


def selectGraspsManually(hf):
    # emg1 = hf.get(str('0'))
    # for i in range(1,125):
    #     n1 = hf.get(str(i))
    #     emg1 = np.concatenate([emg1, n1], axis=1)
    #
    # emg1 = np.array(emg1)
    # print(emg1.shape)

    emg1 = hf.get(str('0'))
    for i in range(1,5):
        n1 = hf.get(str(i))
        emg1 = np.concatenate([emg1, n1], axis=1)

    emg1 = np.array(emg1)
    print(emg1.shape)

    for i in range(49,54):
        n1 = hf.get(str(i))
        emg1 = np.concatenate([emg1, n1], axis=1)

    emg1 = np.array(emg1)
    print(emg1.shape)


    for i in range(13*5-1,13*5-1+5 ):
        n1 = hf.get(str(i))
        emg1 = np.concatenate([emg1, n1], axis=1)

    emg1 = np.array(emg1)
    print(emg1.shape)

    for i in range(17*5-1,17*5-1+5 ):
        n1 = hf.get(str(i))
        emg1 = np.concatenate([emg1, n1], axis=1)

    emg1 = np.array(emg1)
    print(emg1.shape)

    for i in range(24*5-1,24*5-1+5 ):
        n1 = hf.get(str(i))
        emg1 = np.concatenate([emg1, n1], axis=1)

    emg1 = np.array(emg1)
    print(emg1.shape)

    return emg1


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

    plt.savefig(figFile)
    # plt.plot(n1[0, 0:n1.shape[1]],linewidth=1)
    plt.show()


def printByTrial(trial):
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
        plotGridSensors(arr, figFile)


def concatenateDataFronTrials(hf_rest, rmin, rmax):
    data = hf_rest.get(str(rmin-1))
    for i in range(rmin, rmax):
        n1 = hf_rest.get(str(i))
        data = np.concatenate([data, n1], axis=1)
    plt.plot(data[0])
    return data


def fixTheRecording():
    prefix = 'C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\data\\participanttestIris\\'

    prefixRest1 = prefix + 'participanttestIris_day1_block1_20221129142623\\'
    prefixRest2 = prefix + 'participanttestIris_day1_block1_20221129143115\\'
    saveRestFolder = prefix + 'participanttestIris_day1_rest_20221129142623\\'
    if not os.path.exists(saveRestFolder):
        os.mkdir(saveRestFolder)

    fileEnds = ['emg_raw.hdf5', 'emg_proc.hdf5', 'glove_raw.hdf5', 'glove_proc.hdf5']

    for f in fileEnds:
        rest1 = prefixRest1 + f
        rest2 = prefixRest2 + f
        print(rest1)
        print(rest2)

        hf_rest1 = h5py.File(rest1, 'r')
        hf_rest2 = h5py.File(rest2, 'r')

        data1 = concatenateDataFronTrials(hf_rest1, 1, 15)
        data2 = concatenateDataFronTrials(hf_rest2, 1, len(hf_rest2.keys()) - 1)

        # emg2 = hf_rest2.get(str('0'))
        # for i in range(1, len(hf_rest2.keys()) - 1):
        #     print(i)
        #     n1 = hf_rest2.get(str(i))
        #     emg2 = np.concatenate([emg2, n1], axis=1)
        # plt.plot(emg2[0])

        data = np.concatenate([data1, data2], axis=1)
        plt.plot(data[0])

        restFile = saveRestFolder + f
        hf = h5py.File(restFile, 'w')
        hf.create_dataset('rest', data=data)
        hf.close()


def concatenateRecording(participant):
    prefix = 'C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\data\\' + participant + '\\'
    rest1 = prefix + 'fullRecording\\emg_raw.hdf5'

    substring = 'day1'
    folder='C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\data\\participantXinyu\\'
    for dirpath, dnames, fnames in os.walk(folder):
        for filename in dnames:
            if substring in filename:
                print(filename)
                for dirpath1, dnames1, fnames1 in os.walk(filename):
                    name_path = os.path.join(root, filename)
        hf_rest1 = h5py.File(folder, 'r')


def featurePipeline(data):
    n_features_per_channel = 2
    n_channels = 1
    n_samples_raw = data.shape[0]
    winsize = int(S_RATE * READ_LENGTH)
    n_samples_proc = int(n_samples_raw / winsize)

    data_out = np.zeros(
        (n_features_per_channel * n_channels, n_samples_proc))

    b, a = butter(FILTER_ORDER,
                  (LOWCUT / S_RATE / 2., HIGHCUT / S_RATE / 2.),
                  'bandpass')
    pipeline = Pipeline([
        Windower(int(S_RATE * WIN_SIZE)),
        Filter(b, a=a,
               overlap=0),
        FeatureExtractor([('wl', WaveformLength()), ('logvar', LogVar())]),
        Ensure2D(orientation='col')
    ])


    for sample in range(n_samples_proc):
        data_cur = data[sample * winsize:(sample + 1) * winsize]
        data_out[:, sample] = pipeline.process([data_cur]).reshape(-1, )

    return data_out

def saveDataPerGrasp(dict, prefix):
    for k in dict.keys():
        os.mkdir(prefix + str(k))
        outfile = prefix+str(k)+'\\emg_raw.npy'
        np.save(outfile, dict[k])

        figFile = prefix+str(k)+'\\emg_raw.jpg'
        plotGridSensors(dict[k], figFile)


if __name__ == '__main__':
    cp = ConfigParser()
    cp.read(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'config.ini'))
    # processing
    WIN_SIZE = cp.getfloat('processing', 'win_size')
    LOWCUT = cp.getfloat('processing', 'lowcut')
    HIGHCUT = cp.getfloat('processing', 'highcut')
    FILTER_ORDER = cp.getfloat('processing', 'filter_order')



    S_RATE = 2000.
    READ_LENGTH = cp.getfloat('hardware', 'read_length')
    # hf = h5py.File('C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\examples_code\\abstract_control\\data\\SN06\\calibration_20221010173547\\data_proc.hdf5', 'r')
    # hf = h5py.File('C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\data\\Kasia_rec3_05Aug\\lateral_20220805094517\\emg_raw.hdf5', 'r')
    # hf = h5py.File('C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\data\\a\\tripod_20220914192058\\emg_raw.hdf5', 'r')
    # hf = h5py.File('C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\data\\Kasia_Sep_15\\tripod_20220915143102\\emg_raw.hdf5','r')
    # hf = h5py.File('C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\data\\ztest\\rest_20220923192753\\emg_raw.hdf5','r')

    prefix = 'C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\data\\participanttestIris\\participanttestIris_day1_block2_20221129151721\\'
    hf = h5py.File(prefix+'emg_raw.hdf5', 'r')
    grasps_sequence =  [1,2,3,4,5,5,4,2,3,1,4,5,1,2,3,3,1,4,5,2,2,3,5,1,4]
    gr_unique = np.unique(grasps_sequence)


    arr = selectGrasps(grasps_sequence, gr_unique, hf)
    saveDataPerGrasp(arr, prefix)

    figFile = "power.jpg"
    plotGridSensors(arr, figFile)


