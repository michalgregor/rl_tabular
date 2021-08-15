#!/usr/bin/env python3
# -*- coding: utf-8 -*-
VERSION = "0.1"

from .dp import value_iteration
from .value_functions import (
    compute_action_value, compute_state_value,
    StateValueTable, ActionValueTable
)

from .control import vtable_control, qtable_control, action_value_control
from .states import collect_states
from .policy import EpsGreedyPolicy, RandomPolicy
from .replay_buffer import ReplayBuffer
from .trainer import Trainer, EndEpisodeSignal, EndTrainingSignal
from .td import QLearning, TDLearning, SARSA
from .schedule import Schedule, LinearSchedule, ExponentialSchedule
from .utils import seed