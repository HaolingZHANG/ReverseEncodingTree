"""
Name: NEAT evoluted by Estimation of Distribution Algorithm, Population-Based Incremental Learning.

Reference:

Function(s):
Reproduction by Population-Based Incremental Learning.
"""

from neat import DefaultReproduction


class Reproduction(DefaultReproduction):

    def __init__(self, config, reporters, stagnation, learn_rate=0.1, negative_learn_rate=0.075,
                 best_update_rate=0.5, worse_update_rate=0.5):
        super().__init__(config, reporters, stagnation)
        self._learn_rate = learn_rate
        self._negative_learn_rate = negative_learn_rate
        self._best_update_rate = best_update_rate
        self._worse_update_rate = worse_update_rate

    @staticmethod
    def compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size):
        pass

    def reproduce(self, config, species, pop_size, generation):
        pass
