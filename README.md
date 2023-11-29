# Posture_dataset_collection

This repository includes:
- The code to record a surface EMG and kinematics dataset on a Windows machine: 
	- uses axopy library
	- requires the Delsys Trigno Control Utility (for recording EMG)
	- requires a paired via bluetooth dataglove
- The recorded data from 8 subjects using the above code can be found in folder 'data'


# How to run the code
Dependencies:
numpy scipy pandas sklearn  PyQT5 h5py matplotlib  seaborn

Prepare Experiment:
Config file:
Check the amount of channels: 16
Check processing parameters
Change the current_movement appropriately
Pseudo-randomised posture_sequence : the list of grasps sequences already used per subject are at file:  C:\Users\Iris Kyranou\Documents\axopy_applications\Kasia_experiment\participants_order.txt
Pick number of trials
Trial Length and trial_interval time

Run recording experiment:
python3 posture_data_collection.py —trigno (or noise)  --oscilloscopeOff (or oscilloscopeOn) --gloveOff (or gloveOn)




