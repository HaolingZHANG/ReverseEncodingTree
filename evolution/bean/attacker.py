import copy
import random
from enum import Enum

import numpy


class AttackType(Enum):
    Normal = 0
    Reverse = 1
    Gaussian = 2
    Zerout = 3


# noinspection PyPep8Naming
class CartPole_v0_Attacker(object):

    def __init__(self, current_state=False, beta=0.25, epsilon=0.3, attack_type=AttackType.Normal,
                 normal_max=0.1, normal_min=0.05, gaussian_peak=0.2):
        """
        initialize attacker for the game name CartPole v0.

        :param current_state: current state of attack, is attack or not.
        :param beta: rate for the situation of the observed networks.
        :param epsilon: rate of confrontation sample.
        :param attack_type: the operation type of attack.
        :param normal_max: the maximum value in the normal attack type.
        :param normal_min: the minimum value in the normal attack type.
        :param gaussian_peak: the gaussian peak in the gaussian attack type.
        """
        self.current_state = current_state
        self.beta = beta
        self.epsilon = epsilon
        self.attack_type = attack_type

        # initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        # discount factor for DQN agent
        self.gamma = 0.99
        # reverse flag 0% == 100% (dilute noise level, become a half)
        self.reverse_flag = True

        self.normal_max = normal_max
        self.normal_min = normal_min
        self.gaussian_max = gaussian_peak

    def attack(self, original_observation, need_attack=True):
        """
        attack by requested attacker.

        :param original_observation: original observation of Reinforcement Learning.
        :param need_attack: whether the current situation requires attack.

        :return: observation under attack.
        """
        attack_observation = copy.deepcopy(original_observation)
        if need_attack:
            size = len(attack_observation)
            if self.attack_type == AttackType.Normal:
                for index, value in enumerate(numpy.random.normal(self.normal_max, self.normal_min, size)):
                    if random.randint(0, 1) == 1:
                        attack_observation[index] += value
                    else:
                        attack_observation[index] -= value
            elif self.attack_type == AttackType.Reverse:
                if self.reverse_flag:
                    # reverse the ray tracer, but the keep the 2-dim velocity
                    attack_observation = attack_observation[::-1][: size]
                    self.reverse_flag = False
                else:
                    self.reverse_flag = True
            elif self.attack_type == AttackType.Gaussian:
                attack_observation = numpy.random.normal(numpy.mean(attack_observation[: size]),
                                                         self.gaussian_max, size)
            elif self.attack_type == AttackType.Zerout:
                attack_observation[0: size - 1] = 0

        return numpy.array(attack_observation)
