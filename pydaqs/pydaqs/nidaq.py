import numpy as np
from nidaqmx.constants import AcquisitionType
from nidaqmx.task import Task

from .base import _BaseDAQ

__all__ = ['Nidaq']


class Nidaq(_BaseDAQ):
    """
    NIDAQ stream reader device.

    Parameters
    ----------
    channels : list or tuple
        Channels to use.
    rate : float
        Sampling rate.
    samples_per_read : int
        Number of samples per channel to read in each read operation.
    dev : str, optional
        Device name. By default, 1 is used.
    zero_based : bool, optional
        If ``True``, 0-based indexing is used for channel numbering. Default is
        ``True``.
    """

    def __init__(self, channels, rate, samples_per_read, dev='1',
                 zero_based=True):
        self.channels = channels
        self.rate = rate
        self.samples_per_read = samples_per_read
        self.dev = dev
        self.zero_based = zero_based

        self._initialize()

    def _initialize(self):
        self._task = Task()

        for channel in self.channels:
            if self.zero_based:
                channel_no = channel
            else:
                channel_no = channel - 1
            chan_name = 'Dev' + self.dev + '/ai' + str(channel_no)
            self._task.ai_channels.add_ai_voltage_chan(chan_name)

        self._task.timing.cfg_samp_clk_timing(
            rate=self.rate,
            sample_mode=AcquisitionType.FINITE)

    def start(self):
        """Tell the device to begin streaming data. Does not do anything."""
        pass

    def read(self):
        """
        Request a sample of data from the device.

        This is a blocking method, meaning it returns only once the requested
        number of samples are available.

        Returns
        -------
        data : ndarray, shape=(total_signals, num_samples)
            Data read from the device. Each channel is a row and each column
            is a point in time.
        """
        data = self._task.read(
            number_of_samples_per_channel=self.samples_per_read)
        return np.array(data)

    def stop(self):
        """Tell the device to stop streaming data."""
        self._task.close()

    def reset(self):
        """Reset the task."""
        self._initialize()
