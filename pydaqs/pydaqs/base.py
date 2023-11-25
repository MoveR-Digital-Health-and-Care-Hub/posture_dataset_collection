from abc import ABC, abstractmethod

class _BaseDAQ(ABC):
    """
    Base class for DAQ stream reading devices.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    def __del__(self):
        try:
            self.stop()
        except BaseException:
            pass
