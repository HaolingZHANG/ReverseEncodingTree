"""
Name: NEAT evoluted by Binary Search

Function(s):
Reproduction by Binary Search and Random Near Search.
"""

from neat import DefaultReproduction


class Reproduction(DefaultReproduction):

    def __init__(self, config, reporters, stagnation):
        super().__init__(config, reporters, stagnation)

    def create_new(self, genome_type, genome_config, num_genomes):
        pass

    @staticmethod
    def compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size):
        pass

    def reproduce(self, config, species, pop_size, generation):
        pass
