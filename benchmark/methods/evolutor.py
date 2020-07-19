import random

import math
import numpy
import numpy as np
import numpy.linalg as la

from sklearn.cluster import KMeans

from ReverseEncodingTree.benchmark.methods import inherent
from ReverseEncodingTree.benchmark.methods.inherent import Population, Recorder


class GA(object):

    def __init__(self, size, scope, start_position, stride, mutate_rate=0.5):
        """
        initialize the evolution method based on simple genetic algorithm.

        :param size: population size.
        :param scope: scope of landscape.
        :param start_position: starting position of the population.
        :param stride: individual maximum moving step.
        :param mutate_rate: individual mutation rate.
        """
        self.population = Population()
        self.population.create_by_local(size=size, start_position=start_position, stride=stride, scope=scope)
        self.recorder = Recorder()

        self.scope = scope
        self.mutate_rate = mutate_rate
        self.stride = stride
        self.size = size

    def evolute(self, terrain, generations, evolute_type):
        """
        evolution process.

        :param terrain: landscape.
        :param generations: count of generations.
        :param evolute_type: evolution direction, go big or small.
        """
        self.population.fitness(terrain)
        self.recorder.add_population(self.population)

        for generation in range(generations):
            self.population.save_by_sort(self.size, save_type=evolute_type)
            new_population = Population()
            save_count = int(self.size * (1 - self.mutate_rate))

            for index in range(save_count):
                new_population.add_by_individual(self.population.get_individual(index))

            for index in range(save_count, self.size):
                individual = self.population.get_individual(index)
                new_individual = []
                for attribute_index in range(len(individual) - 1):
                    attribute = individual[attribute_index] + random.randint(-self.stride, self.stride)
                    if attribute < 0:
                        attribute = 0
                    elif attribute >= self.scope:
                        attribute = self.scope - 1
                    new_individual.append(attribute)

                new_individual.append(terrain[int(new_individual[0])][int(new_individual[1])])
                new_population.add_by_individual(new_individual)

            self.population = new_population
            self.recorder.add_population(self.population)
            print("generation = " + str(generation))


class PBIL(object):

    def __init__(self, size, scope, learn_rate=0.1):
        """
        initialize the evolution method based on Population-Based Incremental Learning.

        :param size: population size.
        :param scope: scope of landscape.
        :param learn_rate: learn rate of probability matrix.
        """
        self.population = Population()
        self.population.create_by_global(size=size, scope=scope)
        self.recorder = Recorder()

        self.probability_matrix = [[0.5 for _ in range(scope)] for _ in range(scope)]
        self.learn_rate = learn_rate

        self.scope = scope
        self.size = size

    def evolute(self, terrain, generations, evolute_type):
        """
        evolution process.

        :param terrain: landscape.
        :param generations: count of generations.
        :param evolute_type: evolution direction, go big or small.
        """
        self.population.fitness(terrain)
        self.recorder.add_population(self.population)

        for generation in range(generations):
            self._update_probability_matrix(evolute_type)
            self._next_generation()
            self.population.fitness(terrain)
            self.recorder.add_population(self.population)
            print("generation = " + str(generation))

    def _update_probability_matrix(self, evolute_type):
        """
        update the probability matrix by population.

        :param evolute_type: evolution direction, go big or small.
        """
        best_individual = self.population.get_individual(0)
        mean_fitness = best_individual[2]

        if evolute_type == inherent.MAX:
            # obtain best individual
            for index in range(1, self.size):
                individual = self.population.get_individual(index)
                mean_fitness += individual[2]
                if best_individual[2] < individual[2]:
                    best_individual = individual

            mean_fitness /= self.size
            self.probability_matrix[best_individual[0]][best_individual[1]] += 0.5
            max_probability = self.probability_matrix[best_individual[0]][best_individual[1]]

            # update probability matrix
            for index in range(1, self.size):
                individual = self.population.get_individual(index)
                if individual[2] >= mean_fitness:
                    self.probability_matrix[individual[0]][individual[1]] += \
                        self.learn_rate * (self.probability_matrix[best_individual[0]][best_individual[1]] - 0.5)
                    if max_probability < self.probability_matrix[individual[0]][individual[1]]:
                        max_probability = self.probability_matrix[individual[0]][individual[1]]
                else:
                    self.probability_matrix[individual[0]][individual[1]] -= \
                        self.learn_rate * (self.probability_matrix[best_individual[0]][best_individual[1]] - 0.5)

            # adjust value range
            if max_probability > 1:
                for row in range(len(self.probability_matrix)):
                    for col in range(len(self.probability_matrix)):
                        self.probability_matrix[row][col] /= max_probability

        else:
            # obtain best individual
            for index in range(1, self.size):
                individual = self.population.get_individual(index)
                mean_fitness += individual[2]
                if best_individual[2] > individual[2]:
                    best_individual = individual

            mean_fitness /= self.size
            self.probability_matrix[best_individual[0]][best_individual[1]] += 0.5
            max_probability = self.probability_matrix[best_individual[0]][best_individual[1]]

            # update probability matrix
            for index in range(1, self.size):
                individual = self.population.get_individual(index)
                if individual[2] <= mean_fitness:
                    self.probability_matrix[individual[0]][individual[1]] += \
                        self.learn_rate * (self.probability_matrix[best_individual[0]][best_individual[1]] - 0.5)
                    if max_probability < self.probability_matrix[individual[0]][individual[1]]:
                        max_probability = self.probability_matrix[individual[0]][individual[1]]
                else:
                    self.probability_matrix[individual[0]][individual[1]] -= \
                        self.learn_rate * (self.probability_matrix[best_individual[0]][best_individual[1]] - 0.5)

            # adjust value range
            if max_probability > 1:
                for row in range(len(self.probability_matrix)):
                    for col in range(len(self.probability_matrix)):
                        self.probability_matrix[row][col] /= max_probability

    def _next_generation(self):
        while True:
            chooser_matrix = list(numpy.random.random(self.scope ** 2) <=
                                  list(numpy.reshape(numpy.array(self.probability_matrix), (1, self.scope ** 2))[0]))

            if sum(chooser_matrix) >= self.size:
                created_population = Population()
                indices = []
                for index, value in enumerate(chooser_matrix):
                    if value:
                        indices.append(index)
                for choose in random.sample(indices, self.size):
                    col = choose % self.scope
                    row = int((choose - col) / self.scope)
                    created_population.add_by_individual([row, col, 0])

                self.population = created_population
                break


class CMAES(object):

    def __init__(self, size, scope, start_position, stride, elite_rate=0.3):
        """
        initialize the evolution method based on simple genetic algorithm.

        :param size: population size.
        :param scope: scope of landscape.
        :param start_position: starting position of the population.
        """
        self.population = Population()
        self.population.create_by_normal(size, start_position, stride, scope)
        self.recorder = Recorder()

        self.scope = scope
        self.elite_rate = elite_rate
        self.size = size

    def evolute(self, terrain, generations, evolute_type):
        """
        evolution process.

        :param terrain: landscape.
        :param generations: count of generations.
        :param evolute_type: evolution direction, go big or small.
        """
        self.population.fitness(terrain)
        self.recorder.add_population(self.population)

        for generation in range(generations):
            self.population.save_by_sort(self.size, save_type=evolute_type)
            current_individuals = [[], []]
            elite_individuals = [[], []]

            for index in range(self.size):
                individual = self.population.get_individual(index)
                if index <= int(self.size * self.elite_rate):
                    elite_individuals[0].append(individual[0])
                    elite_individuals[1].append(individual[1])
                current_individuals[0].append(individual[0])
                current_individuals[1].append(individual[1])

            elite_individuals = numpy.array(elite_individuals)
            current_individuals = numpy.array(current_individuals)
            centered = elite_individuals - current_individuals.mean(1, keepdims=True)
            c = (centered @ centered.T) / (int(self.size * self.elite_rate) - 1)
            w, e = la.eigh(c)
            new_individuals = None
            while True:
                try:
                    new_individuals = elite_individuals.mean(1, keepdims=True) + (e @ np.diag(np.sqrt(w)) @
                                                                                  np.random.normal(size=(2, self.size)))
                except ValueError:
                    continue
                break

            new_population = Population()
            for position in new_individuals.T:
                if position[0] < 0:
                    position[0] = 0
                elif position[0] >= self.scope:
                    position[0] = self.scope - 1
                if position[1] < 0:
                    position[1] = 0
                elif position[1] >= self.scope:
                    position[1] = self.scope - 1
                new_population.add_by_individual([int(position[0]), int(position[1]), 0])

            new_population.fitness(terrain)
            self.population = new_population

            self.recorder.add_population(self.population)
            print("generation = " + str(generation))


class BI(object):

    def __init__(self, size, init_interval, min_interval, scope):
        """
        initialize the evolution method based on binary search.

        :param size: population size.
        :param init_interval: minimum interval between two individuals at initialization.
        :param min_interval: minimum interval between two individuals at each iteration.
        :param scope: scope of landscape.
        """
        self.population = Population()
        self.population.create_by_global(size=size, scope=scope, interval=init_interval)
        self.recorder = Recorder()
        self.scope = scope
        self.min_interval = min_interval
        self.size = size

    def evolute(self, terrain, generations, evolute_type):
        """
        evolution process.

        :param terrain: landscape.
        :param generations: count of generations.
        :param evolute_type: evolution direction, go big or small.
        """
        self.population.fitness(terrain)
        self.recorder.add_population(self.population)

        for generation in range(generations):
            created_population = Population()

            method = KMeans(n_clusters=self.size, max_iter=self.size)
            feature_matrices = []
            for index in range(self.size):
                feature_matrices.append(self.population.get_individual(index)[:-1])
            method.fit(feature_matrices)
            genome_clusters = [[] for _ in range(self.size)]
            for index, cluster_index in enumerate(method.labels_):
                genome_clusters[cluster_index].append(self.population.get_individual(index))

            for index in range(self.size):
                individual_1 = genome_clusters[index][0]
                for another_index in range(index + 1, self.size):
                    individual_2 = genome_clusters[another_index][0]

                    if evolute_type == inherent.MAX:
                        if math.sqrt(math.pow(individual_1[0] - individual_2[0], 2) +
                                     math.pow(individual_1[1] - individual_2[1], 2)) > self.min_interval:
                            if individual_1[2] > individual_2[2]:
                                new_individual = self._find_around(individual_1, [self.population, created_population],
                                                                   self.min_interval * 2, self.size)
                                if new_individual is not None:
                                    created_population.add_by_individual(new_individual)
                            elif individual_1[2] < individual_2[2]:
                                new_individual = self._find_center(individual_1, individual_2,
                                                                   [self.population, created_population])
                                if new_individual is not None:
                                    created_population.add_by_individual(new_individual)
                    else:
                        if math.sqrt(math.pow(individual_1[0] - individual_2[0], 2) +
                                     math.pow(individual_1[1] - individual_2[1], 2)) > self.min_interval:
                            if individual_1[2] > individual_2[2]:
                                new_individual = self._find_center(individual_1, individual_2,
                                                                   [self.population, created_population])
                                if new_individual is not None:
                                    created_population.add_by_individual(new_individual)
                            elif individual_1[2] < individual_2[2]:
                                new_individual = self._find_around(individual_1, [self.population, created_population],
                                                                   self.min_interval * 2, self.size)
                                if new_individual is not None:
                                    created_population.add_by_individual(new_individual)

            created_population.fitness(terrain)

            self.population.merge_population(created_population)
            self.recorder.add_population(self.population)
            self.population.save_by_sort(self.size, save_type=evolute_type)
            print("generation = " + str(generation))

    def _find_center(self, individual_1, individual_2, remains):
        """
        look for an individual between two individuals.

        :param individual_1: one individual.
        :param individual_2: another individual.

        :param remains: remain population(s).

        :return: created individual (if it is remain).
        """
        new_row = int(round((individual_1[0] + individual_2[0]) / 2))
        new_col = int(round((individual_1[1] + individual_2[1]) / 2))
        is_created = True
        for remain in remains:
            if remain.population is not None:
                if not remain.meet_interval([new_row, new_col], self.min_interval):
                    is_created = False

        return [new_row, new_col, 0] if is_created else None

    def _find_around(self, individual, remains, search_scope, search_count):
        """
        look for an individual around the requested individual.

        :param individual: one individual
        :param remains: remain population(s).
        :param search_scope: searchable scope.
        :param search_count: number of times to search.

        :return: created individual (if it is remain).
        """
        count = 0
        while True:
            new_row = individual[0] + random.randint(-search_scope, search_scope)
            new_col = individual[1] + random.randint(-search_scope, search_scope)

            is_created = True
            if not (0 <= new_row < self.scope and 0 <= new_col < self.scope):
                is_created = False
            for remain in remains:
                if remain.population is not None:
                    if not remain.meet_interval([new_row, new_col], self.min_interval):
                        is_created = False
            if is_created:
                return [new_row, new_col, 0]

            count += 1
            if count >= search_count:
                return None
