import copy
import random
import math
import numpy

MAX = 1
MIN = -1

GRADIENT = 1
POPULATION = -1


class Population(object):

    def __init__(self):
        """
        initialize the population.
        """
        self.population = None

    def create_by_population(self, old_population):
        """
        create the population by old population.

        :param old_population: previous population.
        """
        self.population = copy.deepcopy(old_population)

    def create_by_normal(self, size, start_position, stride, scope):
        self.population = [[], [], []]
        populations_list = numpy.random.normal(0, stride, (2, size * size))
        for position in populations_list.T:
            actual_position = [int(start_position[0] + position[0]), int(start_position[1] + position[1])]
            if actual_position[0] in self.population[0] and actual_position[1] in self.population[1]:
                continue
            elif 0 <= actual_position[0] < scope and 0 <= actual_position[1] < scope:
                self.population[0].append(actual_position[0])
                self.population[1].append(actual_position[1])
                self.population[2].append(0)
                if len(self.population[0]) == size:
                    break

    def create_by_local(self, size, start_position, stride, scope):
        """
        create the population by local random.

        :param size: population size.
        :param start_position: center of local population.
        :param stride: length of step (random scope or side length).
        :param scope: scope or side length of landscape.
        """
        self.population = [[], [], []]
        while len(self.population[0]) < size:
            x = start_position[0] + random.randint(-stride * 3, stride * 3)
            y = start_position[1] + random.randint(-stride * 3, stride * 3)
            if x in self.population[0] and y in self.population[1]:
                continue
            elif 0 <= x < scope and 0 <= y < scope:
                self.population[0].append(x)
                self.population[1].append(y)
                self.population[2].append(0)

    def create_by_global(self, size, scope, interval=0):
        """
        create the population by global random.

        :param size: population size.
        :param scope: scope or side length of landscape.
        :param interval: minimum interval between individuals.
        """
        self.population = [[], [], []]
        while len(self.population[0]) < size:
            actual_row = random.randint(0, scope - 1)
            actual_col = random.randint(0, scope - 1)
            if interval > 0:
                if self.meet_interval([actual_row, actual_col], interval):
                    self.population[0].append(actual_row)
                    self.population[1].append(actual_col)
                    self.population[2].append(0)
            else:
                self.population[0].append(actual_row)
                self.population[1].append(actual_col)
                self.population[2].append(0)

    def add_by_individual(self, individual):
        """
        insert individual in population.

        :param individual: one individual.
        """
        if self.population is None:
            self.population = [[] for _ in range(len(individual))]

        for index, attribute in enumerate(individual):
            self.population[index].append(attribute)

    def merge_population(self, new_remain):
        """
        merge another population in this population.

        :param new_remain: another population.
        """
        for index, attributes in enumerate(new_remain.population):
            self.population[index] += attributes

    def save_by_sort(self, save_count, save_type=MAX):
        """
        save the population by fitness sort.

        :param save_count: count of best preserved individuals.
        :param save_type: type of fitness (save better or worse).
        """
        population = numpy.array(copy.deepcopy(self.population))
        if save_type == MIN:
            population = list(population.T[numpy.lexsort(population)].T)
        else:
            population = list(population.T[numpy.lexsort(-population)].T)

        self.population = []
        for attributes in population:
            self.population.append(list(attributes)[: save_count])

    def get_individual(self, index):
        """
        get individual by index.

        :param index: index in population.

        :return: requested individual.
        """
        individual = []
        for attribute_index in range(len(self.population)):
            individual.append(self.population[attribute_index][index])
        return individual

    def fitness(self, terrain, population=None):
        """
        obtain fitness of each individuals by landscape.

        :param terrain: landscape of phenotype.
        :param population: external population if requested.
        """
        if population is None:
            for index, (x, y) in enumerate(zip(self.population[0], self.population[1])):
                self.population[2][index] = terrain[x][y]
        else:
            for index, (x, y) in enumerate(zip(population[0], population[1])):
                population[2][index] = terrain[x][y]

    def sort_index(self, sort_type=MAX):
        """
        sort individuals by fitness.

        :param sort_type: type of fitness (MAX or MIN).

        :return: sorted index list.
        """
        if sort_type == MIN:
            return list(numpy.argsort(numpy.array(self.population[-1])))
        else:
            return list(numpy.argsort(-numpy.array(self.population[-1])))

    def meet_interval(self, individual, min_interval):
        """
        check whether the new individual meets the distance requirements.

        :param individual: the new individual.
        :param min_interval: minimum interval between every two individual.

        :return: check result.
        """
        min_distance = -1
        for p_row, p_col in zip(self.population[0], self.population[1]):
            distance = math.sqrt(math.pow(p_row - individual[0], 2) + math.pow(p_col - individual[1], 2))
            if distance < min_distance or min_distance == -1:
                min_distance = distance
        return (min_distance > min_interval) or (min_distance == -1)

    def get_all_individuals(self):
        """
        get all individuals from the population.

        :return: individual list.
        """
        return self.population

    def __str__(self):
        """
        show all individuals.

        :return: string of all individual
        """
        individuals = []
        for index in range(self.population[0]):
            individuals.append(self.get_individual(index))

        return str(individuals)


class Recorder(object):

    def __init__(self):
        """
        initialize the recorder to record the changing process of population.
        """
        self.recorder = []
        self.record_type = None

    def add_population(self, remain):
        """
        add new population to the recorder.

        :param remain: remain population.
        """
        if self.record_type is None:
            self.record_type = POPULATION

        if self.record_type is POPULATION:
            self.recorder.append(copy.deepcopy(remain.population))
        else:
            print("Input error record type!")
            exit(1)

    def get_result(self):
        """
        get the result from recorder.
        
        :return: recorder.
        """
        return self.recorder
