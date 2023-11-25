import numpy as np
import pandas as pd
import os
import h5py


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
    return hf_target


def fixTheRecording(fromFile1, fromFile2, toFile, prefix):
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
    df = [df1, df2]
    labels = pd.concat(df)

    # selected_labels = labels[['trial_no', 'target_position', 'grasp', 'seq_index',  'block',  'trial']].copy()
    # seq_index = np.concatenate([([i] * 5) for i in range(int(len(selected_labels)/5))], axis=0)
    # selected_labels.reset_index()
    # selected_labels.loc[:, 'seq_index'] = seq_index
    # selected_labels.loc[:, 'block'] = seq_index
    #
    # selected_labels = selected_labels.reset_index(drop=True)

    labels.to_csv(saveRestFolder+'trials.csv')

    # # copy the parameters file
    # src = prefixRest2 + 'recording_parameters.txt'
    # dst = saveRestFolder + 'recording_parameters.txt'
    # shutil.copyfile(src, dst)

    fileEnds = ['emg_raw.hdf5', 'emg_proc.hdf5', 'glove_raw.hdf5', 'glove_proc.hdf5']
    for f in fileEnds:

        rest1 = prefixRest1 + f
        rest2 = prefixRest2 + f

        hf_rest1 = h5py.File(rest1, 'r+',locking=False)
        hf_rest2 = h5py.File(rest2, 'r+',locking=False)

        restFile = saveRestFolder + f
        hf_target = h5py.File(restFile, 'w')


        hf_target = concatenateDatasets(hf_target, hf_rest1, 0, 150)
        # hf_target.close()
        # hf_target = h5py.File(restFile, 'w')
        hf_target = concatenateDatasets(hf_target, hf_rest2, 0, 150)

        hf_rest2.close()
        hf_rest1.close()
        hf_target.close()

    print("written in " + saveRestFolder)



if __name__ == '__main__':
    # prefix = 'C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\data\\posture_dataset_clean_csv\\'
    prefix = 'C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\data\\posture_dataset_15_8\\'

    for i in range(1,9):
        if i!=7:
            continue

        participant = 'participant_' + str(i)
        print(participant)
        prefixParticipant = prefix + str(participant) + '\\'
        for day in (1,2):

            fromFile1 = "participant" + str(i) + "_day" + str(day) + "_block" + str(1) + "\\"
            fromFile2 = "participant" + str(i) + "_day" + str(day) + "_block" + str(2) + "\\"
            toFile = "participant" + str(i) + "_day" + str(day) + "\\"

            fixTheRecording(fromFile1, fromFile2, toFile, prefixParticipant)

                #
                # prefix_folder = prefix + participant + '\\participant' + str(i) + '_day' + str(day) + '_block' + str(block)
                #
                # if os.path.isfile(prefix_folder + '\\trials1.csv'):
                #     os.remove(prefix_folder + '\\trials1.csv')
                # labels = prefix_folder + '\\trials.csv'
                # df1 = pd.read_csv(labels)
                #
                # df1 = df1.assign(row_number=range(len(df1)))
                #
                # cols = ['row_number', 'target_position', 'grasp', 'trial_no', 'block']
                # df1 = df1[cols]
                #
                # df1.to_csv(prefix_folder + '\\trials.csv')
                #
                # print(df1.head)
