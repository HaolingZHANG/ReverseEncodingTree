"""
Name: NEAT evoluted by Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

Reference:

Function(s):
Reproduction by Covariance Matrix Adaptation Evolution Strategy.
"""

from neat import DefaultReproduction


class Reproduction(DefaultReproduction):

    def create_new(self, genome_type, genome_config, num_genomes):
        pass

    @staticmethod
    def compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size):
        pass

    def reproduce(self, config, species, pop_size, generation):
        pass
