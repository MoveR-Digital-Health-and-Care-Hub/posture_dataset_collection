# Posture_dataset_collection

This repository includes:
- The code to record a surface EMG and kinematics dataset on a Windows machine: 
	- uses the axopy library
	- requires the Delsys Trigno Control Utility (for recording EMG)
	- requires a paired via bluetooth dataglove
- The recorded data from 8 subjects using the above code can be found in folder 'data'
	- emg_data.hdf5 contains the raw recordings from 16 EMG sensors. The file has the format of an HDF5 binary data file and is indexed by the trial number (total of 150 trials). The stored matrix per trial is of 16 x (2kHz*5sec) shape. The signal is processed upon recording, by applying a 4th order Butterworth filter
	-  glove_data.hdf5 file has a similar format and size to the emg_data.hdf5 file. It's an HDF5 binary data file of a shape of 18 x (2kHz*5sec) matrix. 
	- finger_data.hdf5 file is holding information on the position of each of the five fingers.
	- file recording_parameters.txt includes the configuration details of the recording.
	- trials.csv file contains the labeling information on target position and grasp per trial number.


# How to run the code
## Dependencies:
numpy scipy pandas sklearn  PyQT5 h5py matplotlib  seaborn

## Prepare Experiment:
- Config file:
- Check the amount of channels: 16
- Check processing parameters
- Change the current_movement appropriately
- Pseudo-randomised posture_sequence : the list of grasps sequences already used per subject are at file:  'participants_order.txt'
- Pick number of trials
- Trial Length and trial_interval time

## Run recording experiment:
python3 posture_data_collection.py —trigno (or noise)  --oscilloscopeOff (or oscilloscopeOn) --gloveOff (or gloveOn)




