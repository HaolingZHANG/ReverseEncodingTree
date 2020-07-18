import copy
import random
from enum import Enum

import numpy


# noinspection PyPep8Naming
class ATTACK_TYPE(Enum):
    Normal = 0
    Reverse = 1
    Gaussian = 2
    Zerout = 3


# noinspection PyPep8Naming
class CartPole_v0_Attacker(object):

    def __init__(self, current_state=False, beta=0.25, epsilon=0.3, attack_type=ATTACK_TYPE.Normal,
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
            if self.attack_type == ATTACK_TYPE.Normal:
                for index, value in enumerate(numpy.random.normal(self.normal_max, self.normal_min, size)):
                    if random.randint(0, 1) == 1:
                        attack_observation[index] += value
                    else:
                        attack_observation[index] -= value
            elif self.attack_type == ATTACK_TYPE.Reverse:
                if self.reverse_flag:
                    # reverse the ray tracer, but the keep the 2-dim velocity
                    attack_observation = attack_observation[::-1][: size]
                    self.reverse_flag = False
                else:
                    self.reverse_flag = True
            elif self.attack_type == ATTACK_TYPE.Gaussian:
                attack_observation = numpy.random.normal(numpy.mean(attack_observation[: size]),
                                                         self.gaussian_max, size)
            elif self.attack_type == ATTACK_TYPE.Zerout:
                attack_observation[0: size - 1] = 0

        return numpy.array(attack_observation)


class Clever(object):

    def __init__(self, trained_model, maximum_perturbation, estimate_iterations):

        self.trained_model = trained_model
        self.maximum_perturbation = maximum_perturbation
        self.estimate_iterations = estimate_iterations

        self.scores = []

    def calculate_by_dataset(self, data_set, batch_size, sample_number,
                             expected_labels=None, satisfiable_labels=None, expected_results=None, results_ranges=None,
                             is_full=False, chosen_pair=None):

        if is_full:
            if expected_labels is not None and satisfiable_labels is not None:
                for expect_label, data in zip(expected_labels, data_set):
                    self.score_in_classification(expect_label, data,
                                                 satisfiable_labels, batch_size, sample_number)
            elif expected_results is not None and results_ranges is not None:
                for expected_result, data in zip(expected_results, data_set):
                    self.score_in_regression(expected_result, data,
                                             results_ranges, batch_size, sample_number)
            else:
                raise ValueError("no dataset.")

        else:
            if chosen_pair is None:
                chosen_pair = random.randint(0, len(expected_labels) - 1)

            if expected_labels is not None and satisfiable_labels is not None:
                self.score_in_regression(expected_labels[chosen_pair], data_set[chosen_pair],
                                         satisfiable_labels, batch_size, sample_number)
            elif expected_results is not None and results_ranges is not None:
                self.score_in_regression(expected_results[chosen_pair], data_set[chosen_pair],
                                         results_ranges, batch_size, sample_number)
            else:
                raise ValueError("no dataset.")

    def calculate_by_environment(self, environment, episode_generation, input_type, output_type):
        pass

    def score_in_classification(self, expected_label, data, satisfiable_labels, batch_size, sample_number):
        clever_scores = []
        for index, target_label in enumerate(filter(lambda label: label != expected_label, satisfiable_labels)):
            # calculate perturbed set.
            perturbed_set = {}
            for batch_index in range(batch_size):
                predicted_labels = []
                for sample_index in range(sample_number):
                    n_data = numpy.array(data)
                    n_noise = numpy.random.uniform(-self.maximum_perturbation, self.maximum_perturbation, n_data.shape)
                    perturbed_data = (n_data + n_noise).tolist()
                    predicted_labels.append(self.trained_model.activate(perturbed_data))

                expected_values = numpy.array(predicted_labels)[:, satisfiable_labels.index(expected_label)]
                target_values = numpy.array(predicted_labels)[:, satisfiable_labels.index(target_label)]

                perturbed_values = expected_values - target_values

                # TODO gradient of what (population or the winner genome)?
                perturbed_gradient = numpy.array([0 for _ in range(len(data))])

                reshape_value = 1
                for value in perturbed_gradient.shape[1:]:
                    reshape_value *= value

                if reshape_value > 1:
                    perturbed_gradient = perturbed_gradient.reshape((sample_number, reshape_value))

                norm_1 = numpy.linalg.norm(perturbed_gradient, axis=1, ord=1, keepdims=True)
                norm_2 = numpy.linalg.norm(perturbed_gradient, axis=1, ord=2, keepdims=True)
                norm_i = numpy.linalg.norm(perturbed_gradient, axis=1, ord=numpy.inf, keepdims=True)

                perturbed_set[batch_index] = [numpy.max(norm_1), numpy.max(norm_2), numpy.max(norm_i)]

            location_estimate = self.fit_reverse_weibull_distribution(perturbed_set)

            predicted_label = self.trained_model.activate(data)
            expected_value = predicted_label[satisfiable_labels.index(expected_label)]
            target_value = predicted_label[satisfiable_labels.index(target_label)]
            difference = expected_value - target_value

            clever_scores.append(min(difference / location_estimate, self.maximum_perturbation))

        self.scores.append(clever_scores)

    def score_in_regression(self, expected_results, data_set, results_ranges, batch_size, sample_number):
        # TODO
        pass

    def score_in_reinforcement(self):
        # TODO
        pass

    def fit_reverse_weibull_distribution(self, perturbed_set):
        self.estimate_iterations = 1
        return 1

    def get_scores(self):
        pass
