from abc import ABC, abstractmethod
import socket
import struct
import numpy as np


class _BaseTrignoDaq(ABC):
    """
    Delsys Trigno wireless EMG system.

    Requires the Trigno Control Utility to be running.

    Warning: This class should not be used directly.
    Use derived classes instead.

    Parameters
    ----------
    host : str
        IP address the TCU server is running on.
    cmd_port : int
        Port of TCU command messages.
    data_port : int
        Port of TCU data access.
    rate : int
        Sampling rate of the data source.
    signals_per_channel : int
        Number of signals per channel.
    timeout : float
        Number of seconds before socket returns a timeout exception.

    Attributes
    ----------
    BYTES_PER_CHANNEL : int
        Number of bytes per sample per channel. EMG and accelerometer data
    TOTAL_NUM_CHANNELS : int
        Total number of channels.
    CMD_TERM : str
        Command string termination.

    Notes
    -----
    Implementation details can be found in the Delsys SDK reference:
    http://www.delsys.com/integration/sdk/
    """

    BYTES_PER_CHANNEL = 4
    TOTAL_NUM_CHANNELS = 16
    CMD_TERM = '\r\n\r\n'

    def __init__(self, host, cmd_port, data_port, timeout, signals_per_channel):
        self.host = host
        self.cmd_port = cmd_port
        self.data_port = data_port
        self.timeout = timeout
        self._signals_per_channel = signals_per_channel

        self.total_signals = \
            self._signals_per_channel * self.TOTAL_NUM_CHANNELS
        self._min_recv_size = self.total_signals * self.BYTES_PER_CHANNEL

        self._initialize()

    def _initialize(self):

        # create command socket and consume the servers initial response
        self._comm_socket = socket.create_connection(
            (self.host, self.cmd_port), self.timeout)
        self._comm_socket.recv(1024)

        # create the data socket
        self._data_socket = socket.create_connection(
            (self.host, self.data_port), self.timeout)

    def start(self):
        """
        Tell the device to begin streaming data.

        You should call ``read()`` soon after this, though the device typically
        takes about two seconds to send back the first batch of data.
        """
        self._send_cmd('START')

    def set_channels(self, channels):
        """
        Sets the channels to read from the device.

        Parameters
        ----------
        channels : list or tuple
            Sensor channels to use.
        """
        self.channels = set(channels)
        self.num_channels = len(channels)

    def read(self, num_samples):
        """
        Request a sample of data from the device.

        This is a blocking method, meaning it returns only once the requested
        number of samples are available.

        Parameters
        ----------
        num_samples : int
            Number of samples to read per channel.

        Returns
        -------
        data : ndarray, shape=(total_signals, num_samples)
            Data read from the device. Each channel is a row and each column
            is a point in time.
        """
        l_des = num_samples * self._min_recv_size
        l = 0
        packet = bytes()
        while l < l_des:
            try:
                packet += self._data_socket.recv(l_des - l)
            except socket.timeout:
                l = len(packet)
                packet += b'\x00' * (l_des - l)
                raise IOError("Device disconnected.")
            l = len(packet)

        data = np.asarray(
            struct.unpack('<' + 'f' * self.total_signals * num_samples, packet))
        data = np.transpose(data.reshape((-1, self.total_signals)))

        return data[self._signals_read_idx, :]

    def stop(self):
        """Tell the device to stop streaming data."""
        self._send_cmd('STOP')

    def close_connection(self):
        """Closes the connection to the Trigno Control Utility server."""
        self._comm_socket.close()

    def reset(self):
        """Restart the connection to the Trigno Control Utility server."""
        self.close_connection()
        self._initialize()

    def __del__(self):
        try:
            self.close_connection()
        except BaseException:
            pass

    def _send_cmd(self, command):
        self._comm_socket.send(self._cmd(command))
        resp = self._comm_socket.recv(128)
        self._validate(resp)

    @staticmethod
    def _cmd(command):
        return bytes("{}{}".format(command, _BaseTrignoDaq.CMD_TERM),
                     encoding='ascii')

    @staticmethod
    def _validate(response):
        s = str(response)
        if 'OK' not in s:
            print("warning: TrignoDaq command failed: {}".format(s))


class TrignoEMG(_BaseTrignoDaq):
    """
    Delsys Trigno wireless EMG system EMG data.

    Requires the Trigno Control Utility to be running.

    Parameters
    ----------
    channels : list or tuple
        Sensor channels to use. Each sensor has a single EMG
        channel.
    samples_per_read : int
        Number of samples per channel to read in each read operation.
    units : {'V', 'mV', 'normalized'}, optional
        Units in which to return data. If 'V', the data is returned in its
        un-scaled form (volts). If 'mV', the data is scaled to millivolt level.
        If 'normalized', the data is scaled by its maximum level so that its
        range is [-1, 1].
    host : str, optional
        IP address the TCU server is running on. By default, the device is
        assumed to be attached to the local machine.
    cmd_port : int, optional
        Port of TCU command messages.
    data_port : int, optional
        Port of TCU EMG data access. By default, 50041 is used, but it is
        configurable through the TCU graphical user interface.
    timeout : float, optional
        Number of seconds before socket returns a timeout exception.

    Attributes
    ----------
    rate : int
        Sampling rate in Hz.
    scaler : float
        Multiplicative scaling factor to convert the signals to the desired
        units.
    """

    def __init__(self, channels, samples_per_read, zero_based=False, units='V',
                 host='localhost', cmd_port=50040, data_port=50041, timeout=5):
        super(TrignoEMG, self).__init__(
            host=host, cmd_port=cmd_port, data_port=data_port,
            signals_per_channel=1, timeout=timeout)

        self.samples_per_read = samples_per_read
        self.zero_based = zero_based
        self.rate = 2000
        self.scaler = 1.
        if units == 'mV':
            self.scaler = 1000.
        elif units == 'normalized':
            # max range of EMG data is 11 mV
            self.scaler = 1 / 0.011

        self.set_channels(channels)

    def set_channels(self, channels):
        """
        Sets the channels to read from the device.

        Parameters
        ----------
        channels : list or tuple
            Sensor channels to use.
        """
        super(TrignoEMG, self).set_channels(channels=channels)
        if self.zero_based:
            self._signals_read_idx = list(self.channels)
        else:
            self._signals_read_idx = [channel - 1 for channel in self.channels]

    def read(self):
        """
        Request a sample of data from the device.

        This is a blocking method, meaning it returns only once the requested
        number of samples are available.

        Returns
        -------
        data : ndarray, shape=(num_channels, num_samples)
            Data read from the device. Each channel is a row and each column
            is a point in time.
        """
        data = super(TrignoEMG, self).read(self.samples_per_read)
        return self.scaler * data


class TrignoACC(_BaseTrignoDaq):
    """
    Delsys Trigno wireless EMG system accelerometer data.

    Requires the Trigno Control Utility to be running.

    Parameters
    ----------
    channels : list or tuple
        Sensor channels to use. Each sensor has three accelerometer
        channels.
    samples_per_read : int
        Number of samples per channel to read in each read operation.
    host : str, optional
        IP address the TCU server is running on. By default, the device is
        assumed to be attached to the local machine.
    cmd_port : int, optional
        Port of TCU command messages.
    data_port : int, optional
        Port of TCU accelerometer data access. By default, 50042 is used, but
        it is configurable through the TCU graphical user interface.
    timeout : float, optional
        Number of seconds before socket returns a timeout exception.

    Attributes
    ----------
    rate : int
        Sampling rate in Hz.
    """

    def __init__(self, channels, samples_per_read, zero_based=False,
                 host='localhost', cmd_port=50040, data_port=50042, timeout=5):
        super(TrignoACC, self).__init__(
            host=host, cmd_port=cmd_port, data_port=data_port,
            signals_per_channel=3, timeout=timeout)

        self.samples_per_read = samples_per_read
        self.zero_based = zero_based

        self.rate = 148.1

        self.set_channels(channels)

    def set_channels(self, channels):
        """
        Sets the channels to read from the device.

        Parameters
        ----------
        channels : list or tuple
            Sensor channels to use.
        """
        super(TrignoACC, self).set_channels(channels=channels)
        read_idx = np.zeros(0, dtype=int)
        if self.zero_based:
            channels_ = self.channels
        else:
            channels_ = [channel - 1 for channel in self.channels]

        for channel in channels_:
            read_idx = np.append(read_idx, np.arange(
                channel * self._signals_per_channel,
                (channel + 1) * self._signals_per_channel))

        self._signals_read_idx = read_idx

    def read(self):
        """
        Request a sample of data from the device.

        This is a blocking method, meaning it returns only once the requested
        number of samples are available.

        Returns
        -------
        data : ndarray, shape=(num_channels * 3, num_samples)
            Data read from the device. Each channel is a row and each column
            is a point in time.
        """
        data = super(TrignoACC, self).read(self.samples_per_read)
        return data
