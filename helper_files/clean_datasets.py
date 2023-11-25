import numpy as np
import pandas as pd
import h5py
import os
import shutil
from configparser import ConfigParser
import helper_plot_functions as plotFunctions


def concatenateDatasets(hf_target, hf_source, fromIndex1, toIndex1):
    len_db = len(hf_target.keys())
    print(str(len(hf_target.keys())) + " " + str(len(hf_source.keys())))

    for i in range(fromIndex1, toIndex1):
        if hf_source[str(i)].shape is not None:
            # values1 = [hf_source[k] for k in hf_source.keys()
            # hf_temp = hf_source[str(i)]
            # hf_temp['block'] = len_db + 1
            hf_target.create_dataset(name=str(len_db+i), data=hf_source[str(i)])
        else:
            print(i, hf_source[str(i)].shape)

    print("final size: " + str(len(hf_target.keys())))
    return hf_target

def fixTheRecording(fromFile1, fromIndex1, toIndex1, fromFile2, fromIndex2, toIndex2, toFile, prefix):
    prefixRest1 = prefix + fromFile1
    prefixRest2 = prefix + fromFile2
    saveRestFolder = prefix + toFile
    labels1 = prefixRest1 + 'trials.csv'
    labels2 = prefixRest2 + 'trials.csv'

    if not os.path.exists(saveRestFolder):
        os.mkdir(saveRestFolder)

    # create the csv file
    df1 = pd.read_csv(labels1)
    df2 = pd.read_csv(labels2)
    df = [df1.iloc[fromIndex1:toIndex1], df2.iloc[fromIndex2:toIndex2]]
    labels = pd.concat(df)

    selected_labels = labels[['trial_no', 'target_position', 'grasp', 'seq_index',  'block',  'trial']].copy()
    seq_index = np.concatenate([([i] * 5) for i in range(int(len(selected_labels)/5))], axis=0)
    selected_labels.reset_index()
    selected_labels.loc[:, 'seq_index'] = seq_index
    selected_labels.loc[:, 'block'] = seq_index

    selected_labels = selected_labels.reset_index(drop=True)

    selected_labels.to_csv(saveRestFolder+'trials.csv')

    # copy the parameters file
    src = prefixRest2 + 'recording_parameters.txt'
    dst = saveRestFolder + 'recording_parameters.txt'
    shutil.copyfile(src, dst)

    fileEnds = ['emg_raw.hdf5', 'emg_proc.hdf5', 'glove_raw.hdf5', 'glove_proc.hdf5']
    for f in fileEnds:
        rest1 = prefixRest1 + f
        rest2 = prefixRest2 + f

        hf_rest1 = h5py.File(rest1, 'r+',locking=False)
        hf_rest2 = h5py.File(rest2, 'r+',locking=False)

        restFile = saveRestFolder + f
        hf_target = h5py.File(restFile, 'w')

        hf_target = concatenateDatasets(hf_target, hf_rest1, fromIndex1, toIndex1)
        # hf_target.close()
        # hf_target = h5py.File(restFile, 'w')
        hf_target = concatenateDatasets(hf_target, hf_rest2, fromIndex2, toIndex2)

        hf_rest2.close()
        hf_rest1.close()
        hf_target.close()

    print("written in " + saveRestFolder)


if __name__ == '__main__':
    cp = ConfigParser()
    cp.read(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'clean_db_config.ini'))

    fromFile1 = cp.get('participant', 'fromFile1')
    fromFile2 = cp.get('participant', 'fromFile2')
    toFile = cp.get('participant', 'toFile')

    fromIndex1 = cp.getint('participant', 'fromIndex1')
    toIndex1 = cp.getint('participant', 'toIndex1')
    fromIndex2 = cp.getint('participant', 'fromIndex2')
    toIndex2 = cp.getint('participant', 'toIndex2')


    participant = cp.get('participant', 'participant_no')
    prefix = 'C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\data\\posture_dataset_15_8\\participant_'
    prefixParticipant = prefix+ str(participant) + '\\'

    fixTheRecording(fromFile1, fromIndex1, toIndex1, fromFile2, fromIndex2, toIndex2, toFile, prefixParticipant)