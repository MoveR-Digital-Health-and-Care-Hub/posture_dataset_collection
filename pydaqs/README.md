# pydaqs

A collection of wrapper functions for DAQ packages and libraries in Python.
Mainly intended for internal use within [IntellSensing Lab](http://www.intellsensing.com/).

The wrappers follow a simple protocol for data acqusition, which is compatible with [axopy](https://github.com/axopy/axopy). 
Each device implements a `read()` method which returns a numpy array with shape (`n_channels`, `samples_per_read`). This method needs to be called in a loop from the main application. The frequency the method is called needs to be at least equal to the rate data are streamed from the DAQ device.

Here is a minimal working example:

```python
from pydaqs.nidaq import Nidaq
dev = nidaq(channels=[0,1], rate=1000, samples_per_read=100)
dev.read()
dev.stop()
```


## Requirements
[Numpy](https://github.com/numpy/numpy) >= 1.11

## Hardware-specific package requirements
Tested versions in brackets. 
* MYO armband: [myo-python](https://github.com/NiklasRosenstein/myo-python) (v. 1.0.4)
* NI DAQs: [nidaqmx-python](https://github.com/ni/nidaqmx-python) (v. 0.5.7)
* Blackrock hardware (Cerebus, Neuroport): [CereLink](https://github.com/dashesy/CereLink) (v. 0.7.4)

Note that some DAQs also require proprietary software to be running concurrentely for data acquistion to work.

* Myo armband: [Myo connect](https://support.getmyo.com/hc/en-us/articles/360018409792-Myo-Connect-SDK-and-firmware-downloads) (v.1.0.1) and [Myo SDK](https://support.getmyo.com/hc/en-us/articles/360018409792-Myo-Connect-SDK-and-firmware-downloads) (v.0.9.0)
* Blackrock hardware (Cerebus, Neuroport): [Central Software Suite](https://blackrockmicro.com/technical-support/software-downloads/) (v. 7.0.4)
* Digitimer D360 (sampled with NI-DAQ): [D360 control software](https://digitimer.com/products/human-neurophysiology/isolated-amplifiers-emg-eeg/d360-8-channel-patient-amplifier/#Downloads) (v. 4.7.3.0)

## Notes
* Tested with Python >= 3.6
* When using the Myo band (and therefore myo-python), the libary needs to be initalised and also be told where the Myo SDK files are stored on the disk (i.e., `myo.init(sdk_path=...`). See example in the library [documentation](https://github.com/NiklasRosenstein/myo-python#example).
