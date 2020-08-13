import random
from enum import Enum
import math
import numpy
import logging

from neat.nn import feed_forward, recurrent


class LearnType(Enum):
    Supervised = 1
    Reinforced = 2


class NetType(Enum):
    FeedForward = 1
    Recurrent = 2


class EvalType(Enum):
    EulerDistance = 1
    HammingDistance = 2
    ManhattanDistance = 3


class TypeCorrect(Enum):
    List = 1
    Value = 2


class FitDevice(object):

    def __init__(self, method, network_type=NetType.FeedForward):
        """
        Initialize the evolution calculation and type of network.

        :param method: evolution process, see /evolution/methods/
        :param network_type: type of network created by genome.
        """
        logging.info("Initialize the evolution process calculation.")
        self.method = method
        self.network_type = network_type

        self.learn_type = None

        self.dataset = None

        self.environment = None
        self.episode_steps = None
        self.episode_generation = None
        self.input_type = None
        self.output_type = None
        self.attacker = None
        self.noise_level = None

    def set_environment(self, environment, episode_steps, episode_generation,
                        input_type, output_type,
                        attacker=None, noise_level=None):
        """
        Set the environment of Reinforcement Learning in gym library.

        :param environment: environment of Reinforcement Learning in gym library.
        :param episode_steps: maximum episode steps.
        :param episode_generation: evaluate by the minimum of episode rewards.
        :param input_type: type of input, TYPE_CORRECT.List or TYPE_CORRECT.Value.
        :param output_type: type of output, TYPE_CORRECT.List or TYPE_CORRECT.Value.
        :param attacker: noise attacker for observation.
        :param noise_level: noise level of attacker.
        """
        logging.info("Obtain the environment.")
        if self.dataset is None:
            self.environment = environment
            self.episode_steps = episode_steps
            self.episode_generation = episode_generation
            self.learn_type = LearnType.Reinforced
            self.input_type = input_type
            self.output_type = output_type
            self.attacker = attacker
            self.noise_level = noise_level
        elif self.learn_type is None:
            logging.warning("Do not enter data repeatedly!")
        else:
            logging.warning("You have expect data set in Supervised Learning!")

    def set_dataset(self, dataset):
        """
        Set the dataset of Supervised Learning.

        :param dataset: dataset, including inputs and expected outputs, type is {"i": data, "o": data}.
        """
        if self.environment is None:
            self.dataset = dataset
            self.learn_type = LearnType.Supervised
        elif self.learn_type is None:
            logging.warning("Do not enter data repeatedly!")
        else:
            logging.warning("You have environment in Reinforcement Learning!")

    def genomes_fitness(self, genomes, config):
        """
        Calculate the evolution process of genomes.

        :param genomes: genomes of NEAT.
        :param config: configure of genome.
        """
        for genome_id, genome in genomes:
            if genome.fitness is None:
                self.genome_fitness(genome, config)

    def genome_fitness(self, genome, config):
        """
        Calculate the evolution process of genome.

        :param genome: genome of NEAT.
        :param config: configure of genome.
        """
        if self.learn_type == LearnType.Supervised:
            eval("self._genome_in_supervised")(genome, config)
        else:
            eval("self._genome_in_reinforced")(genome, config)

    def _genome_in_supervised(self, genome, config):
        """
        Calculate evolution of genome in Supervised Learning.

        :param genome: one genome in current generation.
        :param config: generated configure of network by genome.
        """
        network = self.generated_network(genome, config)

        obtain_outputs = []
        for current_input in self.dataset.get("i"):
            obtain_outputs.append(network.activate(current_input))

        genome.fitness = self.method.calculate(learn_type=self.learn_type,
                                               obtain_outputs=obtain_outputs,
                                               expected_outputs=self.dataset.get("o"))
    
    def _genome_in_reinforced(self, genome, config):
        """
        Calculate evolution of genome in Reinforcement Learning.

        :param genome: one genomes in current generation.
        :param config: generated configures of network by genome.
        """
        network = self.generated_network(genome, config)

        has_attack = self.attacker is not None and self.noise_level is not None

        episode_recorder = []
        # tasks many episodes for the genome in case it is lucky.
        for episode in range(self.episode_generation):
            accumulative_recorder = 0
            attack_count = 0
            observation = self.environment.reset()
            for step in range(self.episode_steps):
                # set attack if has attack.
                if has_attack and random.randint(0, 100) < self.noise_level * 100:
                    if type(observation) is not numpy.ndarray:
                        observation = numpy.array([observation])
                    actual_observation = self.attacker.attack(observation)
                    attack_count += 1
                else:
                    actual_observation = observation

                # check input type
                if self.input_type == TypeCorrect.List and type(actual_observation) is not numpy.ndarray:
                    actual_observation = numpy.array([actual_observation])
                if self.input_type == TypeCorrect.Value and type(actual_observation) is numpy.ndarray:
                    actual_observation = actual_observation[0]

                action_values = network.activate(actual_observation)
                action = numpy.argmax(action_values)

                # check output type
                if self.output_type == TypeCorrect.List and type(action) is not numpy.ndarray:
                    action = numpy.array([action])
                if self.output_type == TypeCorrect.Value and type(action) is numpy.ndarray:
                    action = action[0]

                current_observation, reward, done, _ = self.environment.step(action)
                accumulative_recorder += reward

                if done:
                    # if has_attack:
                    #     print("with: ", round(attack_count / float(step + 1), 2), "% attack.")
                    break
                else:
                    observation = current_observation
            episode_recorder.append(accumulative_recorder)

        genome.fitness = self.method.calculate(learn_type=self.learn_type,
                                               episode_recorder=episode_recorder,
                                               episode_steps=self.episode_steps)
    
    def generated_network(self, genome, config):
        """
        Obtain a network from genome and its configure.

        :param genome: all genomes in current generation.
        :param config: generated configures of network by genome.

        :return: generated network.
        """
        if self.network_type == NetType.FeedForward:
            return feed_forward.FeedForwardNetwork.create(genome, config)
        elif self.network_type == NetType.Recurrent:
            return recurrent.RecurrentNetwork.create(genome, config)

        return None


class FitProcess(object):

    def __init__(self, init_fitness=None, eval_type=EvalType.EulerDistance):
        """
        Initialize the hyper-parameters.

        :param init_fitness: initialize fitness in Distance.
        :param eval_type: distance type for evaluation.
        """
        self.init_fitness = init_fitness
        self.eval_type = eval_type

    def _update(self, previous_fitness, output, expected_output):
        """
        Update the current fitness.

        :param previous_fitness: previous fitness.
        :param output: actual outputs in Supervised Learning.
        :param expected_output: expected outputs in Supervised Learning.

        :return: current fitness.
        """
        record = 0
        for value, expected_value in zip(output, expected_output):
            if self.eval_type == EvalType.EulerDistance:
                record += math.sqrt(math.pow(value - expected_value, 2))
            elif self.eval_type == EvalType.HammingDistance:
                record += abs(1 - int(value == expected_value))
            elif self.eval_type == EvalType.ManhattanDistance:
                record += math.pow(value - expected_value, 2)

        return previous_fitness - record

    def calculate(self, learn_type,
                  obtain_outputs=None, expected_outputs=None,
                  episode_recorder=None, episode_steps=None):
        """
        Calculate the current fitness.

        :param learn_type: learning type of tasks, Supervised or Reinforced.
        :param obtain_outputs: actual outputs in Supervised Learning.
        :param expected_outputs: expected outputs in Supervised Learning.
        :param episode_recorder: episode recorder in Reinforcement Learning.
        :param episode_steps: episode steps in Reinforcement Learning.

        :return: current fitness.
        """
        if learn_type == LearnType.Supervised:
            if self.init_fitness is None:
                raise Exception("No init fitness value!")
            current_fitness = self.init_fitness
            for output, expected_output in zip(obtain_outputs, expected_outputs):
                current_fitness = self._update(current_fitness, output, expected_output)
            return current_fitness
        else:
            return numpy.min(episode_recorder) / float(episode_steps)
