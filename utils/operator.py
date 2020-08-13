import datetime
import logging
import pickle
import random
import numpy
from enum import Enum

from neat import population, checkpoint, statistics, reporting

from ReverseEncodingTree.evolution.evolutor import LearnType, TypeCorrect
from ReverseEncodingTree.utils import visualize


class ReporterType(Enum):
    Statistic = 1
    Stdout = 2
    Checkpoint = 3


class Operator(object):

    def __init__(self, config, fitter, node_names,
                 max_generation=None, checkpoint_value=-1, stdout=False, output_path=None):
        """
        Initialize the operator of NeuroEvolution.

        :param config: configures of NEAT.
        :param fitter: fitter of NEAT.
        :param node_names: node information for display the obtained network.
        :param max_generation: generations (iteration times).
        :param checkpoint_value: the point to save the current state.
        :param output_path: parent path for save file of displaying the genome and checkpoint.
        :param stdout: whether output the log.
        """
        # load configuration.
        self._config = config

        self._fitter = fitter
        self._node_names = node_names
        self._max_generation = max_generation
        self._output_path = output_path

        # create the population by configuration, which is the top-level object for a NEAT tasks.
        self._population = population.Population(self._config)

        self._checkpoint_value = checkpoint_value
        if self._checkpoint_value >= 0:
            # create the check point reporter.
            self._checkpoint_reporter = checkpoint.Checkpointer(generation_interval=checkpoint_value,
                                                                filename_prefix=output_path + "neat-checkpoint-")
            self._population.add_reporter(self._checkpoint_reporter)

        # create the stdout reporter.
        self._stdout = stdout
        self._stdout_reporter = reporting.StdOutReporter(stdout)
        self._population.add_reporter(self._stdout_reporter)

        # create the statistics reporter.
        self._statistics_reporter = statistics.StatisticsReporter()
        self._population.add_reporter(self._statistics_reporter)

        # best genome after training.
        self._winner = None
        self._obtain_success = False

    def obtain_winner(self):
        """
        Obtain the winning genome (network).
        """
        self._winner = self._population.run(self._fitter.genomes_fitness, self._max_generation)
        if self._winner.fitness >= self._config.fitness_threshold:
            self._obtain_success = True

    def get_best_genome(self):
        """
        get best genome.

        :return: best network.
        """
        return self._winner

    def get_final_genomes(self):
        """
        get final genomes (population)

        :return: final genomes
        """
        genomes = []
        for genome_index, genome in self._population.population.items():
            genomes.append(genome)

        genomes.sort(reverse=True, key=lambda g: g.fitness)

        return genomes

    def get_best_network(self):
        """
        get the winning network.

        :return: generated network.
        """
        if self._winner is None:
            logging.error("Please obtain winner first!")
        return self._fitter.generated_network(self._winner, self._config)

    def get_reporter(self, reporter_type=None):
        """
        get the choose reporter.

        :return: reporter.
        """
        if reporter_type == ReporterType.Statistic:
            return self._statistics_reporter
        elif reporter_type == ReporterType.Stdout:
            return self._stdout_reporter
        elif reporter_type == ReporterType.Checkpoint:
            return self._checkpoint_reporter

        return {"statistics": self._statistics_reporter,
                "stdout": self._stdout_reporter,
                "checkpoint": self._checkpoint_reporter}

    def display_genome(self, filename, node_names=None, genome=None, config=None, reporter=None):
        """
        display the genome.

        :param filename: file name of the output.
        :param node_names: node information for display the obtained network.
        :param genome: genome of network.
        :param config: configures of NEAT.
        :param reporter: statistic reporter
        """
        if node_names is None:
            node_names = self._node_names
        if genome is None:
            genome = self._winner
        if config is None:
            config = self._config
        if reporter is None:
            reporter = self._statistics_reporter

        visualize.draw_network(config, genome, True, node_names=node_names,
                               parent_path=self._output_path, filename=filename)
        visualize.plot_statistics(reporter, y_log=False, show=True,
                                  parent_path=self._output_path, filename=filename)
        visualize.plot_species(reporter, show=True,
                               parent_path=self._output_path, filename=filename)

    def evaluation(self, dataset=None, environment=None,
                   input_type=None, output_type=None, attacker=None, noise_level=None,
                   run_minutes=1):
        """
        evaluate the network by testing dataset or environment.

        :param dataset: testing dataset in Supervised Learning.
        :param environment: environment in Reinforcement Learning.
        :param input_type: type of input, TYPE_CORRECT.List or TYPE_CORRECT.Value.
        :param output_type: type of output, TYPE_CORRECT.List or TYPE_CORRECT.Value.
        :param attacker: noise attacker for observation.
        :param noise_level: noise level of attacker.
        :param run_minutes: running minutes in Reinforcement Learning.
        :return: result in Supervised Learning.
        """
        if self._fitter.learn_type == LearnType.Supervised:
            obtain_outputs = []
            for current_input in dataset.get("i"):
                obtain_outputs.append(self._winner.activate(current_input))

            right = 0
            for obtain_output, expected_output in zip(obtain_outputs, dataset.get("o")):
                is_right = True
                for obtain_value, expected_value in zip(obtain_output, expected_output):
                    if obtain_value != expected_value:
                        is_right = False
                        break
                right += int(is_right)

            return right, len(dataset.get("o"))

        elif self._fitter.learn_type == LearnType.Reinforced:
            has_attack = attacker is not None and noise_level is not None

            network = self.get_best_network()
            start_time = datetime.datetime.now()
            while True:
                observation = environment.reset()
                attack_count = 0
                total_count = 0
                while True:
                    environment.render()

                    # set attack if has attack.
                    if has_attack and random.randint(0, 100) < noise_level * 100:
                        if type(observation) is not numpy.ndarray:
                            observation = numpy.array([observation])
                        actual_observation = attacker.attack(observation)
                        attack_count += 1
                    else:
                        actual_observation = observation

                    # check input type
                    if input_type == TypeCorrect.List and type(actual_observation) is not numpy.ndarray:
                        actual_observation = numpy.array([actual_observation])
                    if input_type == TypeCorrect.Value and type(actual_observation) is numpy.ndarray:
                        actual_observation = actual_observation[0]

                    action_values = network.activate(actual_observation)
                    action = numpy.argmax(action_values)

                    # check output type
                    if output_type == TypeCorrect.List and type(action) is not numpy.ndarray:
                        action = numpy.array([action])
                    if output_type == TypeCorrect.Value and type(action) is numpy.ndarray:
                        action = action[0]

                    _, _, done, _ = environment.step(action)

                    if done:
                        if has_attack:
                            print("with: ", round(attack_count / float(total_count + 1), 2), "% attack.")
                        break

                    total_count += 1

                if (datetime.datetime.now() - start_time).seconds > run_minutes * 60:
                    environment.close()
                    break

            return None

    def get_actual_generation(self):
        """
        get actual generation from stdout reporter.

        :return: actual generation and whether obtain success.
        """
        return self._stdout_reporter.generation, self._obtain_success

    def save_best_genome(self, path):
        """
        save the best genome by file.

        :param path: file path.
        """
        with open(path, "wb") as file:
            pickle.dump(self._winner, file)

    def save_best_network(self, path):
        """
        save the network created by best genome to file.

        :param path: file path.
        """
        with open(path, "wb") as file:
            pickle.dump(self.get_best_network(), file)

    def load_best_genome(self, path):
        """
        load best genome from file.

        :param path: file path.
        """
        with open(path, "rb") as file:
            self._winner = pickle.load(file)

    def reset(self):
        while True:
            # noinspection PyBroadException
            try:
                # re-create the population by configuration, which is the top-level object for a NEAT tasks.
                self._population = population.Population(self._config)
                break
            except Exception:
                print("re-create again!")

        if self._checkpoint_value >= 0:
            # create the check point reporter.
            self._checkpoint_reporter = checkpoint.Checkpointer(generation_interval=self._checkpoint_value,
                                                                filename_prefix=self._output_path + "neat-checkpoint-")
            self._population.add_reporter(self._checkpoint_reporter)

        # create the stdout reporter.
        self._stdout_reporter = reporting.StdOutReporter(self._stdout)
        self._population.add_reporter(self._stdout_reporter)

        # create the statistics reporter.
        self._statistics_reporter = statistics.StatisticsReporter()
        self._population.add_reporter(self._statistics_reporter)

        # best genome after training.
        self._winner = None
        self._obtain_success = False
