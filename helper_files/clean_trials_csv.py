import numpy as np
import pandas as pd
import os

if __name__ == '__main__':
    prefix = 'C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\data\\posture_dataset_15_8\\'

    for i in range(1,9):
        if i !=7:
            continue
        participant = 'participant_' + str(i)
        for day in (1,2):
            for block in (1,2):
                prefix_folder = prefix + participant + '\\participant' + str(i) + '_day' + str(day) + '_block' + str(block)
                if os.path.isfile(prefix_folder + '\\trials1.csv'):
                    os.remove(prefix_folder + '\\trials1.csv')
                labels = prefix_folder + '\\trials.csv'
                df1 = pd.read_csv(labels)

                df1 = df1.assign(row_number=range(len(df1)))

                cols = ['row_number', 'target_position', 'grasp', 'trial_no', 'block']
                df1 = df1[cols]

                df1.to_csv(prefix_folder + '\\trials.csv')

                print(df1.head)


    # for i in range(1,9):
    #     participant = 'participant_' + str(i)
    #     for day in (1, 2):
    #         prefix_folder = prefix + participant + '\\participant' + str(i) + '_day' + str(day)
    #         labels = prefix_folder + '\\trials.csv'
    #         df1 = pd.read_csv(labels)
    #
    #         df1 = df1.assign(row_number=range(len(df1)))
    #
    #         cols = ['row_number', 'target_position', 'grasp', 'trial_no', 'block']
    #         df1 = df1[cols]
    #         if os.path.isfile(prefix_folder + '\\trials1.csv'):
    #             os.remove(prefix_folder + '\\trials1.csv')
    #         df1.to_csv(prefix_folder + '\\trials.csv')
    #
    #         print(df1.head)
