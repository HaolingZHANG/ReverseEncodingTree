"""
Name: NEAT evoluted by Binary Search with Trinet

Function(s):
Reproduction by Binary Search with Trinet.
"""
import numpy
from neat.config import DefaultClassConfig, ConfigParameter

from evolution_process.bean.genome import create_center_new
from evolution_process.methods import bi


class Reproduction(bi.Reproduction):

    @classmethod
    def parse_config(cls, param_dict):
        """
        add init and min distance in config.

        :param param_dict: parameter dictory.

        :return: config.
        """
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('elitism', int, 0),
                                   ConfigParameter('survival_threshold', float, 0.2),
                                   ConfigParameter('min_species_size', int, 2),
                                   ConfigParameter('init_distance', float, 5),
                                   ConfigParameter('min_distance', float, 0.2),
                                   ConfigParameter('search_count', int, 1)])

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

        # calculate average adjusted fitness
        avg_adjusted_fitness = 0
        for genome in current_genomes:
            avg_adjusted_fitness += genome.fitness / pop_size
        self.reporters.info("Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness))

        # sort members in order of descending fitness.
        current_genomes.sort(reverse=True, key=lambda x: x.fitness)

        if len(current_genomes) > pop_size:
            current_genomes = current_genomes[:pop_size]

        # calculate distance matrix of TriNet.
        distance_matrix = [[float("inf") for _ in range(pop_size)] for _ in range(pop_size)]
        for index_1 in range(pop_size):
            genome_1 = current_genomes[index_1]
            for index_2 in range(index_1 + 1, pop_size):
                genome_2 = current_genomes[index_2]
                distance = genome_1.distance(genome_2, self.genome_config)

                if distance > self.reproduction_config.min_distance:
                    distance_matrix[index_1][index_2] = distance
                    distance_matrix[index_2][index_1] = distance

        new_genomes = []

        # create new genome in TriNet.
        for row, one_list in enumerate(distance_matrix):
            cal_list = numpy.array([[i for i in range(len(one_list))], one_list])
            sort_list = cal_list.T[numpy.lexsort(cal_list)].T
            for index in sort_list[0][:3]:
                genome_1 = current_genomes[row]
                genome_2 = current_genomes[int(index)]

                new_genome = create_center_new(genome_1, genome_2, self.genome_config, pop_size + len(new_genomes))
                if new_genome is not None:
                    is_input = True
                    for check_genome in current_genomes + new_genomes:
                        if new_genome.distance(check_genome, self.genome_config) \
                                < self.reproduction_config.min_distance:
                            is_input = False

                    if is_input:
                        new_genomes.append(new_genome)
                        break

        # aggregate final population
        new_population = {}
        for index, genome in enumerate(current_genomes + new_genomes):
            genome.key = index
            new_population[index] = genome

        return new_population
