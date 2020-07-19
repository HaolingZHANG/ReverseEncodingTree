import math
import numpy
import numpy.linalg as la
from neat import DefaultReproduction
from neat.config import DefaultClassConfig, ConfigParameter

from evolution.bean.genome import GlobalGenome


class Reproduction(DefaultReproduction):

    def __init__(self, config, reporters, stagnation):
        super().__init__(config, reporters, stagnation)
        self.genome_config = None
        self.genome_type = None

    @classmethod
    def parse_config(cls, param_dict):
        """
        add init and min distance in config.

        :param param_dict: parameter dictionary.

        :return: config.
        """
        return DefaultClassConfig(param_dict, [ConfigParameter('elite_rate', float, 0.2)])

    def create_new(self, genome_type, genome_config, num_genomes):
        """
        create new genomes by type, config, and number.

        :param genome_type: genome type.
        :param genome_config: genome config.
        :param num_genomes: number of new genomes.

        :return: new genomes.
        """
        if genome_config.num_hidden + genome_config.num_inputs + genome_config.num_outputs > genome_config.max_node_num:
            raise Exception("config: max_node_num must larger than num_inputs + num_outputs + num_hidden")

        self.genome_config = genome_config
        self.genome_type = genome_type

        new_genomes = {}

        for created_index in range(num_genomes):
            key = next(self.genome_indexer)
            genome = genome_type(key)
            while True:
                genome.configure_new(genome_config)
                min_distance = float("inf")
                for index, new_genome in new_genomes.items():
                    current_distance = genome.distance(new_genome, genome_config)
                    if min_distance > current_distance:
                        min_distance = current_distance
                if min_distance > 0:
                    break
            new_genomes[key] = genome
            self.ancestors[key] = tuple()

        return new_genomes

    def reproduce(self, config, species, pop_size, generation):
        """
        handles creation of genomes, either from scratch or by sexual or asexual reproduction from parents.

        :param config: genome config.
        :param species: genome species.
        :param pop_size: population size.
        :param generation: generation of population.

        :return: new population.
        """

        # obtain all genomes from species.
        current_genomes = []
        for i, value in species.species.items():
            members = value.members
            for key, individual in members.items():
                current_genomes.append(individual)

        # sort members in order of descending fitness.
        current_genomes.sort(reverse=True, key=lambda g: g.fitness)

        # create feature matrices and calculate speed list and avg adjusted fitness
        elite_individuals = []
        current_individuals = []
        avg_adjusted_fitness = 0
        length = len(current_genomes[0].feature_matrix) * len(current_genomes[0].feature_matrix[0])
        for index, genome in enumerate(current_genomes):
            if index < int(self.reproduction_config.elite_rate * pop_size):
                avg_adjusted_fitness += genome.fitness / pop_size
                elite_individuals.append(numpy.array(genome.feature_matrix).reshape((1, length))[0])
            current_individuals.append(numpy.array(genome.feature_matrix).reshape((1, length))[0])

        elite_individuals = numpy.array(elite_individuals).T
        current_individuals = numpy.array(current_individuals).T

        self.reporters.info("Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness))

        # adopt feature matrices based on covariance matrix
        centered = elite_individuals - current_individuals.mean(1, keepdims=True)
        c = (centered @ centered.T) / (int(pop_size * self.reproduction_config.elite_rate) - 1)
        w, e = la.eigh(c)

        new_individuals = None
        while True:
            try:
                # TODO nan problem
                new_individuals = elite_individuals.mean(1, keepdims=True) + \
                                  (e @ numpy.diag(numpy.sqrt(w)) @ numpy.random.normal(size=(length, pop_size)))
            except ValueError:
                continue
            break

        new_individuals = new_individuals.T
        new_population = {}
        for index, individual in enumerate(new_individuals):
            feature_matrix = []
            for row in individual.reshape((int(math.sqrt(len(individual))), int(math.sqrt(len(individual))) + 1)):
                feature_matrix.append(list(row))
            genome = GlobalGenome(index)
            genome.feature_matrix_new(feature_matrix, self.genome_config)
            new_population[index] = genome

        return new_population
