import os
import numpy as np
import random
import time
import pyqtgraph

from scipy.signal import butter

from argparse import ArgumentParser
from configparser import ConfigParser
from axopy.pipeline import (Callable, Windower, Filter, Pipeline,
                            FeatureExtractor, Ensure2D, Estimator, Block)
from axopy.features import mean_absolute_value, WaveformLength, LogVar, mean_value
from axopy.experiment import Experiment
from axopy import util
from axopy.task import Task, Oscilloscope
from axopy.daq import NoiseGenerator, DumbDaq
from axopy.timing import Counter, Timer
from axopy.gui.canvas import Canvas, Item, Circle, Cross, Line, Text, Rectangle
from axopy.gui.graph import SignalWidget
from axopy.gui.prompts import ImagePrompt

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QPoint, QSize
from PyQt5.QtGui import QPainterPath, QColor, QPalette, QPixmap, QFont

from cyberglove import CyberGlove
from PyQt5.QtWidgets import QLabel

import winsound



class Color(QtWidgets.QWidget):

    def __init__(self, color):
        super(Color, self).__init__()
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(color))
        self.setPalette(palette)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("My App")
        self.icon_size = (350, 350)
        # but.setFixedSize(500, 500)
        # but.setIconSize(QSize(200, 200))
        self.grid_size = (3,3)

        self.create_buttons()
        self.create_icons()

        layout = QtWidgets.QGridLayout()
        # layout.setSpacing(0)
        # layout.setHorizontalSpacing(0)
        # layout.setVerticalSpacing(0)
        layout.setContentsMargins(400, 0, 400, 0)
        self.init_buttons()
        for x in range(0, self.grid_size[0]):
            for y in range(0, self.grid_size[1]):
                index = y + self.grid_size[0]*x
                layout.addWidget(self.button_list[index], x, y)


        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)



    def create_buttons(self):
        buttons = [{'button_id': 1, 'button_name': 'button1'},
                    {'button_id': 2, 'button_name': 'button2'},
                    {'button_id': 3, 'button_name': 'button3'},
                    {'button_id': 4, 'button_name': 'button4'},
                    {'button_id': 5, 'button_name': 'button5'},
                    {'button_id': 6, 'button_name': 'button6'},
                    {'button_id': 7, 'button_name': 'button7'},
                    {'button_id': 8, 'button_name': 'button8'},
                    {'button_id': 9, 'button_name': 'button9'}
        ]

        self.button_list = []

        for i in buttons:
            but = QtWidgets.QToolButton()
            but.setFont(QFont('Times', 16))

            for k, v in i.items():
                setattr(but, k, v)
            self.button_list.append(but)

    def create_icons(self):
        movements = [{'icon_id': 1, 'icon_name': 'lateral', 'location': 'C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\pics\\lateral.jpg'},
                    {'icon_id': 2, 'icon_name': 'open', 'location': 'C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\pics\\open.jpg'},
                    {'icon_id': 3, 'icon_name': 'power', 'location': 'C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\pics\\power.jpg'},
                    {'icon_id': 4, 'icon_name': 'pointer', 'location': 'C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\pics\\pointer.jpg'},
                    {'icon_id': 5, 'icon_name': 'rest', 'location': 'C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\pics\\rest.jpg'},
                    {'icon_id': 6, 'icon_name': 'tripod', 'location': 'C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\pics\\tripod.jpg'}
        ]

        movementsGrey = [{'icon_id': 1, 'icon_name': 'lateral',
                      'location': 'C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\pics\\lateral_g.jpg'},
                     {'icon_id': 2, 'icon_name': 'open',
                      'location': 'C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\pics\\open_g.jpg'},
                     {'icon_id': 3, 'icon_name': 'power',
                      'location': 'C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\pics\\power_g.jpg'},
                     {'icon_id': 4, 'icon_name': 'pointer',
                      'location': 'C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\pics\\pointer_g.jpg'},
                     {'icon_id': 5, 'icon_name': 'rest',
                      'location': 'C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\pics\\rest_g.jpg'},
                     {'icon_id': 6, 'icon_name': 'tripod',
                      'location': 'C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\pics\\tripod_g.jpg'}
                     ]

        self.bg_icon = QtGui.QIcon('C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\pics\\bg.jpg')
        self.next_target =  QtGui.QIcon('C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment\\pics\\next_target.jpg')

        self.movements_dict = {}

        for i in movements:
            mov = QtGui.QIcon(i['location'])
            for k, v in i.items():
                setattr(mov, k, v)
            self.movements_dict[i['icon_name']] = mov

        self.movements_dict_grey = {}

        for i in movementsGrey:
            mov = QtGui.QIcon(i['location'])
            for k, v in i.items():
                setattr(mov, k, v)
            self.movements_dict_grey[i['icon_name']] = mov

    def init_buttons(self):
        for i in range(0, len(self.button_list)):
            if i+1 == int(POSTURE_SEQ[0]):
                #self.button_list[i].setIcon(self.next_target)
                self.button_list[i].setIcon(self.movements_dict_grey[CUR_MOVEMENT])
                self.button_list[i].setText("next grip: " + CUR_MOVEMENT)

            else:
                self.button_list[i].setIcon(self.bg_icon)
            self.button_list[i].setStyleSheet('QToolButton {background-color: #d3d3d3; color: black; font: bold}')
            self.button_list[i].setIconSize(QtCore.QSize(*self.icon_size))
            self.button_list[i].setIconSize(QtCore.QSize(*self.icon_size))
            self.button_list[i].setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)


class Scope(pyqtgraph.Qt.QtGui.QWidget):
    def __init__(self, channel_names, yrange=None):
        # open new window and set properties
        QtWidgets.QWidget.__init__(self)
        layout = QtWidgets.QGridLayout()
        # layout.setSpacing(10)
        self.setLayout(layout)
        self.oscilloscope = SignalWidget(channel_names=channel_names, yrange=yrange)
        layout.addWidget(self.oscilloscope, 0, 0)
        self.setWindowFlag(QtCore.Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)
        self.show()

    def plot(self, data):
        self.oscilloscope.plot(data)


class _BaseTask(Task):
    """Base experimental task.

    Implements the processing pipeline, the daqstream and the trial counter.
    """

    def __init__(self):
        super(_BaseTask, self).__init__()
        self.pipeline = self.make_pipeline()
        if OSCILLOSCOPE_ON:
            self.cur_proc_data = {'emg': None, 'glove': None, 'monitor': None}
        else:
            self.cur_proc_data = {'emg': None, 'glove': None}

    def make_pipeline(self):
        # Multiple feature extraction could also be implemented using a
        # parallel pipeline and a block that joins multiple outputs.
        if EMG_or_ACC == 1:
            pipeline_raw = Pipeline()
            return pipeline_raw

        b, a = butter(FILTER_ORDER,
                     (LOWCUT / S_RATE / 2., HIGHCUT / S_RATE / 2.),
                      'bandpass')
        pipeline_raw = Pipeline([
            Windower(int(S_RATE * WIN_SIZE)),
            Filter(b, a=a,
                   overlap=(int(S_RATE * WIN_SIZE) -
                            int(S_RATE * READ_LENGTH))),
            Ensure2D(orientation='col')
        ])

        pipeline = Pipeline([
            Windower(int(S_RATE * WIN_SIZE)),
            Filter(b, a=a,
                   overlap=(int(S_RATE * WIN_SIZE) -
                            int(S_RATE * READ_LENGTH))),
            FeatureExtractor([('wl', WaveformLength()), ('logvar', LogVar())],
                             n_channels=len(CHANNELS)),
            Ensure2D(orientation='col')
        ])

        return pipeline_raw

    def make_emg_pipeline(self):
        b, a = butter(FILTER_ORDER,
                      (LOWCUT / S_RATE / 2., HIGHCUT / S_RATE / 2.),
                      'bandpass')
        pipeline = Pipeline([
            Windower(int(S_RATE * WIN_SIZE)),
            Filter(b, a=a,
                   overlap=(int(S_RATE * WIN_SIZE) -
                            int(S_RATE * READ_LENGTH))),
            FeatureExtractor([('wl', WaveformLength()), ('logvar', LogVar())],
                             n_channels=len(CHANNELS)),
            Ensure2D(orientation='col')
        ])

        return pipeline

    def make_monitor_pipeline(self):

        pipeline = Pipeline([
            Windower(4000),
        ])

        return pipeline

    def prepare_daq(self, daqstream):
        self.daqstream = daqstream
        self.daqstream['master'].start()
        self.daqstream['emg'].start()
        self.daqstream['glove'].start()

    def reset(self):
        # self.timer.reset()
        if OSCILLOSCOPE_ON:
            self.cur_proc_data = {'emg': None, 'glove': None, 'monitor': None}
        else:
            self.cur_proc_data = {'emg': None, 'glove': None}

    def key_press(self, key):
        super(_BaseTask, self).key_press(key)
        if key == util.key_escape:
            self.finish()

    def finish(self):
        # self.daqstream.stop()
        self.writer.write(self.trial)
        self.daqstream['master'].stop()
        self.daqstream['emg'].stop()
        self.daqstream['glove'].stop()
        self.finished.emit()

    def update_emg(self, data):
        data_proc = self.pipeline['emg'].process(data)
        if OSCILLOSCOPE_ON:
            mon_data = self.pipeline['monitor'].process(data)
            self.cur_proc_data['monitor'] = mon_data
        self.cur_proc_data['emg'] = data_proc
        self.trial.arrays['emg_raw'].stack(data)

    def update_glove(self, data):
        if GLOVE_REC_ON:
            data_proc = self.pipeline['glove'].process(data)
            self.cur_proc_data['glove'] = data_proc
            self.trial.arrays['glove_raw'].stack(data)

    def image_path(self, grip):
        """Returns the path for specified grip. """
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'pics',
            grip + '.jpg')
        print(path)
        return path

    def run_trial(self, trial):
        self.reset()
        trial.add_array('emg_raw', stack_axis=1)
        trial.add_array('emg_proc', stack_axis=1)
        trial.add_array('glove_raw', stack_axis=1)
        trial.add_array('glove_proc', stack_axis=1)
        # self.pipeline.clear()

        self.pipeline['emg'].clear()
        self.pipeline['glove'].clear()
        if OSCILLOSCOPE_ON:
            self.pipeline['monitor'].clear()

        # self.connect(self.daqstream.updated, self.update)
        self.connect_all()

    def connect_all(self):
        self.connect(self.daqstream['master'].updated, self.update_master)
        self.connect(self.daqstream['emg'].updated, self.update_emg)
        self.connect(self.daqstream['glove'].updated, self.update_glove)

    def disconnect_all(self):
        self.disconnect(self.daqstream['master'].updated, self.update_master)
        self.disconnect(self.daqstream['emg'].updated, self.update_emg)
        self.disconnect(self.daqstream['glove'].updated, self.update_glove)


class DataCollection(_BaseTask):

    def __init__(self):
        super(DataCollection, self).__init__()
        positions = (1, 2, 3, 4, 5, 6, 7, 8, 9)
        self.previous_position = positions[0]
        self.first_run = True
        self.remaining_trials = N_TRIALS-1
        self.position_index = 0
        if OSCILLOSCOPE_ON:
            self.pipeline = {
                'emg': self.make_emg_pipeline(),
                'glove': self.make_glove_pipeline(),
                'monitor':self.make_monitor_pipeline()
            }
        else:
            self.pipeline = {
                'emg': self.make_emg_pipeline(),
                'glove': self.make_glove_pipeline()
            }

    def make_glove_pipeline(self):
        pipeline = Pipeline([
            Windower(int(GLOVE_S_RATE * WIN_SIZE)),
            Callable(mean_value),
            Callable(lambda x: np.dot(x, GLOVE_FINGER_MAP)),
            Ensure2D(orientation='col')
        ])

        return pipeline

    def prepare_storage(self, storage):
        dtime = time.strftime('%Y%m%d%H%M%S', time.localtime())

        FILENAME = participant + '_' + 'day' + str(DAY) + '_' + 'block' + str(BLOCK) + '_' +dtime
        PARAM_FILE = 'data\\' + participant + '\\' + FILENAME + '\\recording_parameters.txt'

        self.writer = storage.create_task(FILENAME)  # CUR_MOVEMENT + '_' + dtime + '\\')

        with open(PARAM_FILE, 'w') as f:
            f.write('Participant: ' + participant + '\n')
            f.write('Day: ' + str(DAY) + '\n')
            f.write('Block: ' + str(BLOCK) + '\n')
            f.write('Configuration: ' + CONFIGURATION + '\n')
            f.write('Time of Recording: ' + dtime + '\n')
            f.write('----------------------------------------------------\n')
            f.write('Window Size: ' + str(WIN_SIZE) + '\n')
            f.write('Lowcut Frequency: ' + str(LOWCUT) + '\n')
            f.write('Highcut Frequency: ' + str(HIGHCUT) + '\n')
            f.write('Filter Order: ' + str(FILTER_ORDER) + '\n')
            f.write('Number of Trials: ' + str(N_TRIALS) + '\n')
            f.write('Trial interval time (sec): ' + str(TRIAL_INTERVAL) + '\n')
            f.write('Trial length time (sec): ' + str(TRIAL_LENGTH) + '\n')
            f.write('Channels No: ' + str(len(CHANNELS)) + '\n')


    def calc_coordinates(self, A, B, n_steps=10):
        a_x, a_y = A
        b_x, b_y = B
        n = n_steps + 1
        step_width_x = (b_x - a_x) / n
        step_width_y = (b_y - a_y) / n
        return [(a_x + i * step_width_x, a_y + i * step_width_y) for i in range(1, n)]

    def prepare_design(self, design):
        # target_positions = {"1": (-0.67, 0.67), "2":(0, 0.67), "3":(0.67, 0.67),
        #                     "4": (-0.67, 0.0), "5":(0, 0.0), "6":(0.67, 0.0),
        #                     "7": (-0.67, -0.67), "8":(0, -0.67), "9":(0.67, -0.67)}

        #for mov in GRASPS_SEQ:
        for tar in range(0,len(POSTURE_SEQ)):
            cur_tar = POSTURE_SEQ[tar]
            CUR_MOVEMENT = GRASPS_SEQ[tar]
            # x, y = target_positions[tar]
            block = design.add_block()
            for trial in range(N_TRIALS):
                block.add_trial(attrs={
                    'trial_no': trial,
                    'target_position': POSTURE_SEQ[tar],
                    'grasp': CUR_MOVEMENT,
                    'seq_index':tar
                })

    def update_master(self):

        # This the "master" Daq, i.e. an update happens when master daq is
        # updated. The check is used to ensure that updates start only after
        # the two streams have started providing data.
        if not any(elem is None for elem in self.cur_proc_data.values()):
            emg_proc = self.cur_proc_data['emg'].copy()
            glove_proc = self.cur_proc_data['glove'].copy()
            if OSCILLOSCOPE_ON:
                monitor_proc = self.cur_proc_data['monitor'].copy()
                self.oscilloscope.plot(monitor_proc)

            self.trial.arrays['emg_proc'].stack(emg_proc)
            self.trial.arrays['glove_proc'].stack(glove_proc)

            # self.timer.increment()

    def prepare_graphics(self, container):
        self.mw = MainWindow()
        if OSCILLOSCOPE_ON:
            self.oscilloscope = Scope(channel_names=CHANNELS, yrange=(-0.0001, 0.0001))
        container.set_widget(self.mw)

    def beep_to_start(self, when):
        if when=="rec_start":
            frequency = 400
            duration = 100
            winsound.Beep(frequency, duration)
        elif when=="rec_stop":
            frequency = 300
            duration = 100
            winsound.Beep(frequency, duration)

    def run_trial(self, trial):
        self.reset()

        self.position = int(self.trial.attrs['target_position']) - 1
        self.grasp = graspsDict[(self.trial.attrs['grasp'])]
        # print(self.previous_position, self.position+1)

        if (int(self.previous_position) != int(self.position+1)):
            self.mw.button_list[self.previous_position-1].setIcon(self.mw.bg_icon)
            self.mw.button_list[self.previous_position-1].setText("")
            self.previous_position = self.position+1

            # self.mw.button_list[self.position].setIcon(self.mw.next_target)
            # self.mw.button_list[self.position].setText("")

        self.mw.button_list[self.position].setIcon(self.mw.movements_dict[self.grasp])
        self.mw.button_list[self.position].setText(self.grasp + ' grip: [' + str(N_TRIALS - self.remaining_trials) + '/' + str(N_TRIALS) + ']')
        self.beep_to_start("rec_start")
        self.mw.button_list[self.position].setStyleSheet('QToolButton {background-color: #d3d3d3; color: green; font: bold}')

        self.timer = Timer(TRIAL_LENGTH)
        self.timer.timeout.connect(self.finish_trial)
        self.timer.start()

        super(DataCollection, self).run_trial(trial)

    def update(self, data):
        print('in update')
        data_proc = self.pipeline.process(data)
        emg_proc = self.cur_proc_data['emg'].copy()
        glove_proc = self.cur_proc_data['glove'].copy()

        self.trial.arrays['data_raw'].stack(data)
        self.trial.arrays['data_proc'].stack(data_proc)
        self.trial.arrays['emg_proc'].stack(emg_proc)
        self.trial.arrays['glove_proc'].stack(glove_proc)

        self.timer.increment()

    def finish_trial(self):
        self.disconnect_all()


        self.mw.button_list[self.position].setIcon(self.mw.movements_dict['rest'])

        self.mw.button_list[self.position].setText("rest")
        self.mw.button_list[self.position].setStyleSheet('QToolButton {background-color: #d3d3d3; color: red; font: bold}')

        self.writer.write(self.trial)
        self.disconnect_all()

        if CUR_MOVEMENT == "rest":
            self.wait_timer = Timer(0.0)
        else:
            self.beep_to_start("rec_stop")
            self.wait_timer = Timer(TRIAL_INTERVAL - VISUAL_DELAY)
        self.wait_timer.timeout.connect(self.finish_position)
        self.wait_timer.start()


    def finish_position(self):
        self.mw.button_list[self.position].setIcon(self.mw.bg_icon)
        self.mw.button_list[self.position].setText("")
        if int(self.trial.attrs['trial_no']) == N_TRIALS-1:
            self.position_index = self.position_index + 1
            #self.mw.button_list[int(POSTURE_SEQ[self.position_index])-1].setIcon(self.mw.next_target)

            print(self.position_index)
            print((self.trial.attrs['seq_index'])+1)

            if (int(self.position_index) == len(POSTURE_SEQ)):
                self.disconnect_all()
                return
            else:
                self.mw.button_list[int(POSTURE_SEQ[self.position_index]) - 1].setIcon(self.mw.movements_dict_grey[graspsDict[GRASPS_SEQ[(self.trial.attrs['seq_index'])+1]]])
                self.mw.button_list[int(POSTURE_SEQ[self.position_index])-1].setText("next grip: " + graspsDict[GRASPS_SEQ[(self.trial.attrs['seq_index'])+1]])
                self.remaining_trials = N_TRIALS

        self.disconnect_all()
        self.wait_timer = Timer(VISUAL_DELAY)
        self.wait_timer.timeout.connect(self.next_trial)
        self.wait_timer.start()
        self.remaining_trials = self.remaining_trials-1


if __name__ == '__main__':
    # dev = NoiseGenerator(rate=2000, num_channels=4, read_size=200)
    #

    graspsDict = {
        "1": "power",
        "2": "lateral",
        "3": "tripod",
        "4": "pointer",
        "5": "open",
        "6": "rest",
    }

    parser = ArgumentParser()

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--trigno', action='store_true')
    source.add_argument('--noise', action='store_true')

    oscill = parser.add_mutually_exclusive_group(required=True)
    oscill.add_argument('--oscilloscopeOn', action='store_true')
    oscill.add_argument('--oscilloscopeOff', action='store_true')

    glove = parser.add_mutually_exclusive_group(required=True)
    glove.add_argument('--gloveOn', action='store_true')
    glove.add_argument('--gloveOff', action='store_true')

    args = parser.parse_args()

    cp = ConfigParser()
    cp.read(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'config.ini'))

    # hardware
    READ_LENGTH = cp.getfloat('hardware', 'read_length')
    CHANNELS = list(map(int, (cp.get('hardware', 'channels').split(','))))
    GLOVE_PORT = cp.get('hardware', 'glove_port')
    GLOVE_S_RATE = 40.
    GLOVE_FINGER_MAP = np.loadtxt(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     'map.csv'),
        delimiter=',')
    cal_path = os.path.join('C:\\Users\\Iris Kyranou\\Documents\\axopy_applications\\Kasia_experiment',
                            'glove_calibration.cal')

    # processing
    WIN_SIZE = cp.getfloat('processing', 'win_size')
    LOWCUT = cp.getfloat('processing', 'lowcut')
    HIGHCUT = cp.getfloat('processing', 'highcut')
    FILTER_ORDER = cp.getfloat('processing', 'filter_order')

    # experiment
    MOVEMENTS = cp.get('experiment', 'movements').split(',')
    GRASPS_SEQ = cp.get('experiment', 'grasps_sequence').split(',')
    CUR_MOVEMENT  = graspsDict[GRASPS_SEQ[0]]
    POSTURE_SEQ =  cp.get('experiment', 'posture_sequence').split(',')
    EMG_or_ACC = cp.getint('experiment', 'emg_or_acc')
    PARTICIPANT = cp.get('experiment', 'participant')

    BLOCK = cp.getint('experiment', 'block')
    DAY = cp.getint('experiment', 'day')
    CONFIGURATION = cp.get('experiment', 'configuration')

    # control
    TRIAL_INTERVAL = cp.getfloat('control', 'trial_interval')
    TRIAL_LENGTH = cp.getfloat('control', 'trial_length')
    N_TRIALS = cp.getint('control', 'n_trials')
    #N_BLOCKS = len(MOVEMENTS)


    VISUAL_DELAY = cp.getfloat('experiment', 'visual_delay')  # used to show the next movement slightly earlier than the recording starts, to minimise user reaction delay

    remaining_trials = N_TRIALS

    OSCILLOSCOPE_ON = 0
    GLOVE_REC_ON = 0

    participant = "participant" + PARTICIPANT




    if args.trigno:
        from pytrigno.pytrigno import TrignoEMG, TrignoACC
        if EMG_or_ACC == 0:
            print('recording EMG')
            S_RATE = 2000.
            dev_emg = TrignoEMG(channels=CHANNELS, zero_based=False,
                            samples_per_read=int(S_RATE * READ_LENGTH))
        else:
            print('recording ACC')
            S_RATE = 148.
            dev_emg = TrignoACC(channels=CHANNELS, zero_based=False,
                            samples_per_read=int(S_RATE * READ_LENGTH))
    elif args.noise:
        from axopy.daq import NoiseGenerator
        S_RATE = 2000.
        dev_emg = NoiseGenerator(rate=S_RATE, num_channels=8, amplitude=10.0,
                             read_size=int(S_RATE * READ_LENGTH))

    if args.oscilloscopeOn:
        OSCILLOSCOPE_ON = 1



    dev_master = DumbDaq(rate=S_RATE,
                         read_size=int(S_RATE * READ_LENGTH))


    if args.gloveOn:
        GLOVE_REC_ON = 1
        dev_glove = CyberGlove(n_df=18, s_port=GLOVE_PORT, cal_path=cal_path,
                               samples_per_read=1)

    else:
        GLOVE_REC_ON = 0
        from axopy.daq import NoiseGenerator
        S_RATE = 2000.
        dev_glove = NoiseGenerator(rate=S_RATE, num_channels=8, amplitude=10.0,
                                 read_size=int(S_RATE * READ_LENGTH))


    daq = {'master': dev_master, 'emg': dev_emg, 'glove': dev_glove}



    Experiment(daq=daq, subject=participant).run(DataCollection())

    # Experiment(daq=dev, allow_overwrite=True).run(
    #     DataCollection())
