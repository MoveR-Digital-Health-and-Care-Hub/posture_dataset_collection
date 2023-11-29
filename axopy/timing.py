"""Utilities for keeping track of time in a task."""

from __future__ import division
from PyQt5 import QtCore
from axopy.messaging import Transmitter, TransmitterBase


class Counter(TransmitterBase):
    """Counts to a given number then transmits a timeout event.

    Parameters
    ----------
    max_count : int
        Number of iterations to go through before transmitting the `timeout`
        event. Must be greater than 1.
    reset_on_timeout : bool, optional
        Specifies whether or not the timer should reset its count back to zero
        once the timeout event occurs. The default behavior is to reset.

    Attributes
    ----------
    count : int
        Current count.
    timeout : Transmitter
        Transmitted when ``max_count`` has been reached.

    Examples
    --------
    Basic usage:

    >>> from axopy.timing import Counter
    >>> timer = Counter(2)
    >>> timer.increment()
    >>> timer.count
    1
    >>> timer.progress
    0.5
    >>> timer.increment()
    >>> timer.count
    0
    """

    timeout = Transmitter()

    def __init__(self, max_count=1, reset_on_timeout=True):
        super(Counter, self).__init__()
        max_count = int(max_count)
        if max_count < 1:
            raise ValueError('max_count must be > 1')

        self.reset_on_timeout = reset_on_timeout

        self.max_count = max_count
        self.count = 0

    @property
    def progress(self):
        """Progress toward timeout, from 0 to 1."""
        return self.count / self.max_count

    def increment(self):
        """Increment the counter.

        If `max_count` is reached, the ``timeout`` event is transmitted. If
        `reset_on_timeout` has been set to True (default), the timer is also
        reset.
        """
        self.count += 1

        if self.count == self.max_count:
            if self.reset_on_timeout:
                self.reset()

            self.timeout.emit()

    def reset(self):
        """Resets the count to 0 to start over."""
        self.count = 0


class StepCounter(Counter):
    """Multiple step counter.
       Counts to a given number then transmits a timeout event.
       Transmits events at multiple values up to timeout.

    Parameters
    ----------
    max_count : int
        Number of iterations to go through before transmitting the `timeout`
        event. Must be greater than 1.
    reset_on_timeout : bool, optional
        Specifies whether or not the timer should reset its count back to zero
        once the timeout event occurs. The default behavior is to reset.

    Attributes
    ----------
    count : int
        Current count.
    timeout : Transmitter
        Transmitted when ``max_count`` has been reached.

    Examples
    --------
    Basic usage:

    >>> from axopy.timing import StepCounter
    >>> timer = StepCounter(3)
    >>> timer.add_step(1, function1)
    >>> timer.add_step(2, function2)
    >>> timer.increment()
    "function 1"
    >>> timer.increment()
    "function 2"
    >>> timer.increment()
    >>> timer.count
    0
    """

    timeout = Transmitter()

    # README: No way to create an array of pyqtSignal(s) ...
    # https://stackoverflow.com/questions/38506979/creating-an-array-of-pyqtsignal 
    step_max = 10
    for i in range(step_max):
        vars()['step' + str(i)] = Transmitter()

    def __init__(self, max_count=1, reset_on_timeout=True):
        super(StepCounter, self).__init__(max_count, reset_on_timeout)
        self.step_inc = 0
        self.step_count = []

    @staticmethod
    def _dummy():
        """A default method for add_step."""
        pass

    def add_step(self, count=0, event=_dummy):
        """Add a step if we have enough emitters."""
        if self.step_inc < self.step_max:
            getattr(self, 'step' + str(self.step_inc)).connect(event)
            self.step_count.append(count)
            self.step_inc += 1

    def increment(self):
        """Increment the counter.

        If a count is reached which is found in `step_count` the event
        associated with the count is transmitted.

        If `max_count` is reached, the ``timeout`` event is transmitted. If
        `reset_on_timeout` has been set to True (default), the timer is also
        reset.
        """
        self.count += 1

        if self.count in self.step_count:
            # _index = self.step_count.index(self.count)
            # getattr(self, 'step' + str(_index)).emit()
            _index = [i for i, x in enumerate(self.step_count)
                      if x == self.count]
            for i in _index:
                getattr(self, 'step' + str(i)).emit()

        if self.count == self.max_count:
            if self.reset_on_timeout:
                self.reset()
            self.timeout.emit()


class Timer(TransmitterBase):
    """Real-time one-shot timer.

    This is useful in situations where you want to wait for some amount of time
    and locking the timing to data acquisition updates is not important. For
    example, inserting a waiting period between trials of a task can be done by
    connecting the ``timeout`` transmitter to your task's
    :meth:`~axopy.task.Task.next_trial` method.

    Parameters
    ----------
    duration : float
        Duration of the timer, in seconds.

    Attributes
    ----------
    timeout : Transmitter
        Transmitted when the timer has finished.
    """

    timeout = Transmitter()

    def __init__(self, duration):
        super(Timer, self).__init__()
        self.duration = duration

        self._qtimer = QtCore.QTimer()
        self._qtimer.setInterval(int(1000*self.duration))
        self._qtimer.setSingleShot(True)
        self._qtimer.timeout.connect(self.timeout)

    def start(self):
        """Start the timer."""
        self._qtimer.start()

    def stop(self):
        """Stop the timer.

        If you stop the timer early, the timeout event won't be transmitted.
        """
        self._qtimer.stop()
