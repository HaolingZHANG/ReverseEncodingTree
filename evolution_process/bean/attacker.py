import copy
from enum import Enum

import numpy


# noinspection PyPep8Naming
class ATTACK_TYPE(Enum):
    Reverse = 1
    GaussianAvg = 2
    Zerout = 3


class CartPole_v0_Attacker(object):

    def __init__(self, current_state=False, beta=0.25, epsilon=0.3, attack_type=ATTACK_TYPE.Reverse):
        """
        initialize attacker for the game name CartPole v0.

        :param current_state: current state of attack, is attack or not.
        :param beta: rate for the situation of the observed networks.
        :param epsilon: rate of confrontation sample.
        :param attack_type: the operation type of attack.
        """
        self.current_state = current_state
        self.beta = beta
        self.epsilon = epsilon
        self.attack_type = attack_type

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        # discount factor for DQN agent
        self.gamma = 0.99

    def attack(self, original_observation, need_attack=False):
        """
        attack by requested attacker.

        :param original_observation: original observation of Reinforcement Learning.
        :param need_attack: whether the current situation requires attack.

        :return: observation under attack.
        """
        attack_observation = copy.deepcopy(original_observation)
        if need_attack:
            size = len(attack_observation)
            if self.attack_type == ATTACK_TYPE.Reverse:
                # reverse the ray tracer, but the keep the 2-dim velocity
                attack_observation = attack_observation[::-1][: size]
            elif self.attack_type == ATTACK_TYPE.GaussianAvg:
                numpy.random.normal(numpy.mean(attack_observation[: size]), 0.1, size)
            elif self.attack_type == ATTACK_TYPE.Zerout:
                attack_observation[0: size - 1] = 0

        return numpy.array(attack_observation)
