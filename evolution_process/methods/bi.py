"""
Name: NEAT evoluted by Binary Search

Function(s):
Reproduction by Binary Search and Random Near Search.
"""
import math
from statistics import mean

from neat import DefaultReproduction
from neat.config import DefaultClassConfig, ConfigParameter
from six import itervalues, iteritems


class Reproduction(DefaultReproduction):

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
                                   ConfigParameter('search_count', int, 0)])

    def create_new(self, genome_type, genome_config, num_genomes):
        """
        create new genomes by type, config, and number.

        :param genome_type: genome type.
        :param genome_config: genome config.
        :param num_genomes: number of new genomes.

        :return: new genomes.
        """

        new_genomes = {}
        distance_matrix = [[float("inf") for _ in range(num_genomes - 1)] for _ in range(num_genomes - 1)]

        for created_index in range(num_genomes):
            key = next(self.genome_indexer)
            genome = genome_type(key)
            count = 0
            while True:
                genome.configure_new(genome_config)
                min_distance = float("inf")
                for check_index in range(1, created_index):
                    current_distance = genome.distance(new_genomes[check_index], genome_config)
                    distance_matrix[created_index - 1][check_index - 1] = current_distance
                    distance_matrix[check_index - 1][created_index - 1] = current_distance
                    if min_distance > current_distance:
                        min_distance = current_distance
                if min_distance >= self.reproduction_config.init_distance:
                    break

                count += 1

                if count > self.reproduction_config.search_count:
                    raise Exception('init_distance is too large for the whole landscape.')

            new_genomes[key] = genome
            self.ancestors[key] = tuple()

        print("distance:")
        for i in distance_matrix:
            print(i)
        return new_genomes

    def reproduce(self, config, species, pop_size, generation):
        """
        Handles creation of genomes, either from scratch or by sexual or asexual reproduction from parents.

        :param config: genome config.
        :param species: genome species.
        :param pop_size: population size.
        :param generation: generation of population.

        :return: new population.
        """
        print(species.species)
        for key, value in species.species.items():
            print(str(key) + " | " + str(value.members))
        # TODO

        new_population = {}

        return new_population
