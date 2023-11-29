"""Task design containers."""

import numpy
import random
import pprint

__all__ = ['Design', 'Block', 'Trial', 'Array']


class Design(list):
    """Top-level task design container.

    The :class:`Design` is a list of :class:`Block` objects, which themselves
    are lists of :class:`Trial` objects.
    """

    def add_block(self):
        """Add a block to the design.

        Returns
        -------
        block : design.Block
            The created block.
        """
        block = Block(len(self))
        self.append(block)
        return block


class Block(list):
    """List of trials.

    Experiments often consist of a set of blocks, each containing the same set
    of trials in randomized order. You usually shouldn't need to create a block
    directly -- use :meth:`Design.add_block` instead.

    Parameters
    ----------
    index : int
        Index of the block in the design. This is required to pass along to
        each trial in the block, so that the trial knows which block it belongs
        to.
    """

    def __init__(self, index, *args, **kwargs):
        super(Block, self).__init__(*args, **kwargs)
        self.index = index

    def add_trial(self, attrs=None):
        """Add a trial to the block.

        A :class:`Trial` object is created and added to the block. You can
        optionally provide a dictionary of attribute name/value pairs to
        initialize the trial.

        Parameters
        ----------
        attrs : dict, optional
            Dictionary of attribute name/value pairs.

        Returns
        -------
        trial : Trial
            The trial object created. This can be used to add new attributes or
            arrays. See :class:`Trial`.
        """
        if attrs is None:
            attrs = {}

        attrs.update({'block': self.index, 'trial': len(self)})

        trial = Trial(attrs=attrs)
        self.append(trial)
        return trial

    def shuffle(self, reset_index=True, seed=None):
        """Shuffle the block's trials in random order.

        Parameters
        ----------
        reset_index : bool, optional
            Whether or not to set the ``trial`` attribute of each trial such
            that they remain in sequential order after shuffling. This is the
            default.
        seed : int, optional
            If provided, the random seed will be set to the specified value to
            ensure reproducible shuffling. Note that if you have multiple
            identical blocks and want to shuffle them differently, use a
            different seed value for each block.
        """
        if seed is not None:
            random.seed(seed)

        random.shuffle(self)
        if reset_index:
            for i, trial in enumerate(self):
                trial.attrs['trial'] = i


class Trial(object):
    """Container of trial data.

    There are two kinds of data typically needed during a trial: attributes and
    arrays. Attributes are scalar quantities or primitives like integers,
    floating point numbers, booleans, strings, etc. Arrays are NumPy arrays,
    useful for holding things like cursor trajectories.

    There are two primary purposes for each of these two kinds of data. First,
    it's useful to design a task with pre-determined values, such as the target
    location or the cursor trajectory to follow. The other purpose is to
    temporarily hold runtime data using the same interface, such as the final
    cursor position or the time-to-target.

    You shouldn't normally need to create a trial directly -- instead, use
    :meth:`Block.add_trial`.

    Attributes
    ----------
    attrs : dict
        Dictionary mapping attribute names to their values.
    arrays : dict
        Dictionary mapping array names to :class:`Array` objects, which contain
        the array data.
    """

    def __init__(self, attrs):
        self.attrs = attrs
        self.arrays = {}

    def add_array(self, name, **kwargs):
        """Add an array to the trial.

        Parameters
        ----------
        name : str
            Name of the array.
        kwargs : dict
            Keyword arguments passed along to :class:`Array`.
        """
        self.arrays[name] = Array(**kwargs)

    def add_bufferedarray(self, name, **kwargs):
        """Add a buffered array to the trial.

        Parameters
        ----------
        name : str
            Name of the array.
        kwargs : dict
            Keyword arguments passed along to :class:`BufferedArray`.
        """
        self.arrays[name] = BufferedArray(**kwargs)

    def __str__(self):
        return pprint.pformat(self.attrs)


class Array(object):
    """Trial array.

    The array is not much more than a NumPy array with a :meth:`stack` method
    for conveniently adding new data to the array. This is useful in cases
    where you iteratively collect new segments of data and want to concatenate
    them. For example, you could use an :class:`Array` to collect the samples
    from a data acquisition device as they come in.

    You usually don't need to create an array manually -- instead, use
    :meth:`Trial.add_array`.

    Parameters
    ----------
    data : ndarray, optional
        Data to initialize the array with. If ``None``, the first array passed
        to :meth:`stack` is used for initialization.
    stack_axis : int, optional
        Axis to stack the data along.
    dtype : str, optional
        Array data type. Default is 'f'.

    Attributes
    ----------
    data : ndarray, optional
        The NumPy array holding the data.
    """

    _stack_funcs = {0: numpy.vstack, 1: numpy.hstack, 2: numpy.dstack}

    def __init__(self, data=None, stack_axis=1, dtype='f'):
        self.data = data
        self.stack_axis = stack_axis
        self.dtype = dtype

    def stack(self, data):
        """Stack new data onto the array.

        Parameters
        ----------
        data : ndarray
            New data to add. The direction to stack along is specified in the
            array's constructor (stack_axis).
        """
        if self.data is None:
            self.data = data
        else:
            self.data = self._stack_funcs[self.stack_axis]([self.data, data])

    def clear(self):
        """Clears the buffer.

        Anything that was in the buffer is not retrievable.
        """
        self.data = None


class BufferedArray(object):
    """Trial array.

    The buffered array preallocates an array an insert method for adding new
    data. The size of the buffer must be set manually. This is useful in cases
    where you are iteratively collecting new segments of data and you would
    like to record for longer periods of time. For example, you could use a
    :class:`BufferedArray` to collect samples from a data acquisition device
    as they come in for rather a long time and things should not start to lag
    because memory has already been allocated.

    The attribute data is empty until :meth:`BufferedArray.set_data` is
    called. If the size of the buffer is exceeded data will be empty.

    You usually don't need to create an buffered array manually -- instead,
    use :meth:`Trial.add_bufferedarray`.

    Parameters
    ----------
    buffer_dims: tuple(int, int, int)
        Tuple to determine the size of the buffered array.

    insert_axis : int, optional
        Axis to insert the data along.

    dtype : str, optional
        Array data type. Default is 'f'.

    Attributes
    ----------
    data : ndarray, optional
        The NumPy array holding the data.
    """

    def __init__(self, buffer_dims=(8, 10), insert_axis=1, dtype='f'):
        self.insert_axis = insert_axis
        self.dtype = dtype
        self.buffer_dims = buffer_dims
        self.buffer = numpy.zeros(buffer_dims, self.dtype)
        self.pos = 0
        self.overflow = False
        self.data = numpy.empty((0, 0))

    def insert(self, data):
        """Insert new data into the buffer.

        Parameters
        ----------
        data : ndarray
            New data to add. The direction to insert on is specified in the
            array's constructor (insert_axis).
        """
        if self.overflow:
            return

        if data.ndim > 1:
            new_sample = data.shape[self.insert_axis]
        else:
            new_sample = 1

        new_pos = self.pos + new_sample
        if (new_pos > self.buffer_dims[self.insert_axis]):
            self.overflow = True
            return

        # about as stupid as Axopy's arbitrary stack dims
        idx = slice(self.pos, new_pos)
        if (self.insert_axis == 0):
            self.buffer[idx, :] = data
        elif(self.insert_axis == 1):
            self.buffer[:, idx] = data
        elif(self.insert_axis == 2):
            self.buffer[:, :, idx] = data

        self.pos += new_sample

    def clear(self):
        """Clears the buffer and resets the interator

        Anything that was in the buffer is not retrievable.
        """
        self.buffer.fill(0)
        self.pos = 0
        self.overflow = False
        self.data = numpy.empty((0, 0))

    def set_data(self):
        """Creates the data attribute.

        In the case of overflow data will be empty.
        """
        if self.overflow:
            return

        idx = slice(0, self.pos)
        if (self.insert_axis == 0):
            self.data = self.buffer[idx, :]
        elif(self.insert_axis == 1):
            self.data = self.buffer[:, idx]
        elif(self.insert_axis == 2):
            self.data = self.buffer[:, :, idx]