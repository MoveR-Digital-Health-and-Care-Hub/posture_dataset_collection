import numpy as np
import h5py
import helper_plot_functions as plotFunctions
import os


def selectGraspsManually(hf):

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



def selectGrasps(grasps_sequence, gr_unique, hf):
    ind = 0
    gr_index = np.zeros([len(gr_unique), 5])

    for gr in gr_unique:
        gr_index[gr-1] = [i for i, x in enumerate(grasps_sequence) if x == gr]
        ind = ind+1


    emgShape = hf.get(str('0'))
    emgDict = {}

    for gr in gr_unique:
        print('grasp ' + gr)
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


def saveDataPerGrasp(dict, prefix):
    for k in dict.keys():
        os.mkdir(prefix + 'grasp_' + str(k))
        outfile = prefix + 'grasp_' + str(k)+'\\emg_raw.npy'
        np.save(outfile, dict[k])

        figFile = prefix + 'grasp_' + str(k)+'\\emg_raw'
        plotFunctions.plotGridSensors(dict[k], figFile)


def concatenateDatasets(hf_target, hf_source, fromIndex1, toIndex1):
    len_db = len(hf_target.keys())

    for i in range(fromIndex1, toIndex1):
        if hf_source[str(i)].shape is not None:
            # print(i, hf_source[str(i)])
            # values1 = [hf_source[k] for k in hf_source.keys()
            # hf_temp = hf_source[str(i)]
            # hf_temp['block'] = len_db + 1
            hf_target.create_dataset(name=str(len_db+i), data=hf_source[str(i)])
        else:
            print(i, hf_source[str(i)].shape)



def concatenateDatasetsByIndex(hf_target, hf_source, indices):
    len_db = len(hf_target.keys())

    for i in indices:
        if hf_source[str(i)].shape is not None:
            # print(i, hf_source[str(i)])
            # values1 = [hf_source[k] for k in hf_source.keys()
            # hf_temp = hf_source[str(i)]
            # hf_temp['block'] = len_db + 1
            hf_target.create_dataset(name=str(len_db+i), data=hf_source[str(i)])
        else:
            print(i, hf_source[str(i)].shape)


def concatenateDataFromTrials(hf_rest, rmin, rmax):
    data = hf_rest.get(str(rmin-1))
    for i in range(rmin, rmax):
        n1 = hf_rest.get(str(i))
        data = np.concatenate([data, n1], axis=1)
    return data


def concatenateDataFromTrialsByIndex(hf_rest, indices):
    data = hf_rest.get(str(indices[0]))
    for i in indices[1:]:
        n1 = hf_rest.get(str(i))
        data = np.concatenate([data, n1], axis=1)
    return data

def plotSingleChannelH5PY(file, rmin, rmax, channel):
    hf = h5py.File(file, 'r+', locking=False)
    emg = concatenateDataFromTrials(hf, rmin, rmax)
    pltDatabyChannel(emg, channel)